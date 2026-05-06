"""Integration tests for all three memory repositories.

Requires a live Postgres (local docker-compose or Supabase); auto-skipped
when unreachable via the ``migrated_db`` fixture from conftest.
"""

from __future__ import annotations

import math
from uuid import uuid4

import pytest

from career_coach.db import get_pool
from career_coach.memory.episodic import EpisodicRepo
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo
from career_coach.models.user_model import EvidenceDraft, FactUpdate

pytestmark = pytest.mark.asyncio


# -------- helpers -------------------------------------------------------


async def _make_user(pool_dsn: str) -> ...:  # type: ignore[type-arg]
    """Insert a throwaway user for an isolated test."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            "INSERT INTO users (display_name, is_test) VALUES ($1, TRUE) RETURNING user_id",
            f"test_{uuid4().hex[:8]}",
        )
    return user_id


# -------- Tier 1: structured facts --------------------------------------


async def test_structured_repo_upsert_and_get_all(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = StructuredFactsRepo()

    await repo.upsert(user_id, FactUpdate(key="age", value=19))
    await repo.upsert(user_id, FactUpdate(key="location", value="Berlin"))

    facts = await repo.get_all(user_id)
    assert facts == {"age": 19, "location": "Berlin"}


async def test_structured_repo_upsert_overwrites(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = StructuredFactsRepo()

    await repo.upsert(user_id, FactUpdate(key="location", value="Paris"))
    await repo.upsert(user_id, FactUpdate(key="location", value="Berlin"))

    facts = await repo.get_all(user_id)
    assert facts["location"] == "Berlin"


async def test_structured_repo_bulk_upsert(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = StructuredFactsRepo()

    await repo.bulk_upsert(
        user_id,
        [
            FactUpdate(key="age", value=20),
            FactUpdate(key="goal", value="software engineer"),
        ],
    )

    facts = await repo.get_all(user_id)
    assert facts["age"] == 20
    assert facts["goal"] == "software engineer"


# -------- Tier 2: episodic memory ---------------------------------------


async def test_episodic_repo_session_and_turn(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = EpisodicRepo()

    session = await repo.create_session(user_id, session_theory="initial theory")
    assert session.session_id is not None
    assert session.session_theory == "initial theory"

    turn_id = await repo.save_turn(
        session_id=session.session_id,
        user_id=user_id,
        user_message="What career suits me?",
        assistant_message="Tell me more about your interests.",
        intent_packet={"turn_intent": "explore"},
        flow_used="B",
        critic_verdicts=[{"verdict": "pass"}],
        tokens_used={"understander": 50, "coach": 200},
    )
    assert turn_id is not None

    recent = await repo.get_recent(session.session_id, limit=5)
    assert len(recent) == 1
    assert recent[0].user_message == "What career suits me?"
    assert recent[0].intent == "explore"


async def test_episodic_repo_get_recent_order(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = EpisodicRepo()
    session = await repo.create_session(user_id)

    for i in range(4):
        await repo.save_turn(
            session_id=session.session_id,
            user_id=user_id,
            user_message=f"msg {i}",
            assistant_message=f"reply {i}",
            intent_packet=None,
            flow_used="A",
            critic_verdicts=None,
            tokens_used=None,
        )

    recent = await repo.get_recent(session.session_id, limit=3)
    assert len(recent) == 3
    # Must be in ascending (chronological) order.
    indices = [t.turn_index for t in recent]
    assert indices == sorted(indices)
    # Newest 3 only (turns 1, 2, 3).
    assert indices[0] == 1


async def test_episodic_repo_embedding_round_trip(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = EpisodicRepo()
    session = await repo.create_session(user_id)

    turn_id = await repo.save_turn(
        session_id=session.session_id,
        user_id=user_id,
        user_message="vector test",
        assistant_message="ok",
        intent_packet=None,
        flow_used="A",
        critic_verdicts=None,
        tokens_used=None,
    )

    # 384-dim unit vector for the test.
    dim = 384
    embedding = [1.0 / math.sqrt(dim)] * dim
    await repo.save_turn_embedding(turn_id, embedding)

    # Searching with the same vector should find the turn we just inserted.
    results = await repo.search_similar(user_id, embedding, limit=5)
    found_ids = [r.turn.turn_id for r in results]
    assert turn_id in found_ids
    # Cosine distance of identical vectors must be essentially 0.
    match = next(r for r in results if r.turn.turn_id == turn_id)
    assert match.distance < 1e-4


async def test_episodic_repo_update_session_theory(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = EpisodicRepo()
    session = await repo.create_session(user_id, "old theory")

    await repo.update_session_theory(session.session_id, "new theory")

    updated = await repo.get_session(session.session_id)
    assert updated is not None
    assert updated.session_theory == "new theory"


# -------- Tier 3: semantic / hypotheses ---------------------------------


async def test_semantic_repo_create_and_retrieve(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = SemanticRepo()

    h_id = await repo.create_hypothesis(
        user_id,
        statement="User gravitates toward analytical work.",
        confidence=0.4,
        open_questions=["Does this hold outside academia?"],
    )
    assert h_id is not None

    active = await repo.get_active(user_id)
    assert any(h.hypothesis_id == h_id for h in active)
    hyp = next(h for h in active if h.hypothesis_id == h_id)
    assert hyp.statement == "User gravitates toward analytical work."
    assert hyp.open_questions == ["Does this hold outside academia?"]


async def test_semantic_repo_append_evidence(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = SemanticRepo()
    h_id = await repo.create_hypothesis(user_id, "test hyp", confidence=0.3)

    evidence = EvidenceDraft(
        source_type="user_statement",
        excerpt="I prefer theory over practice.",
        weight=0.7,
    )
    e_id = await repo.append_evidence(h_id, evidence)
    assert e_id is not None

    # Evidence stored — verify via raw query.
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT weight, excerpt FROM hypothesis_evidence WHERE evidence_id = $1",
            e_id,
        )
    assert row is not None
    assert abs(row["weight"] - 0.7) < 1e-6


async def test_semantic_repo_update_confidence(migrated_db: str) -> None:
    user_id = await _make_user(migrated_db)
    repo = SemanticRepo()
    h_id = await repo.create_hypothesis(user_id, "test hyp 2", confidence=0.2)

    await repo.update_confidence(h_id, 0.85, status="active")

    active = await repo.get_active(user_id)
    hyp = next(h for h in active if h.hypothesis_id == h_id)
    assert abs(hyp.confidence - 0.85) < 1e-6
