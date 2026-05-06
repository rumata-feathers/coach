"""Integration tests for the distillation job.

Seeds a user with evidence rows (bypassing the Profiler background task for
speed), then runs distillation and asserts that hypotheses are created.

Uses a MockLLMClient to control the distillation LLM output deterministically.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from career_coach.db import get_pool
from career_coach.jobs.distillation import run_distillation
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.memory.episodic import EpisodicRepo
from career_coach.memory.semantic import SemanticRepo
from career_coach.models.user_model import EvidenceDraft

pytestmark = pytest.mark.asyncio


async def _make_user(name: str):  # type: ignore[no-untyped-def]
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "INSERT INTO users (display_name, is_test) VALUES ($1, TRUE) RETURNING user_id", name
        )


async def _seed_user_with_evidence(user_id, n_evidence: int = 5) -> list:  # type: ignore[no-untyped-def]
    """Create a session, one turn, then queue evidence rows via SemanticRepo."""
    episodic = EpisodicRepo()
    semantic = SemanticRepo()

    session = await episodic.create_session(user_id)
    turn_id = await episodic.save_turn(
        session_id=session.session_id,
        user_id=user_id,
        user_message="I really enjoy working on hard maths problems.",
        assistant_message="That's a strong signal for quantitative fields.",
        intent_packet={"turn_intent": "explore"},
        flow_used="B",
        critic_verdicts=None,
        tokens_used=None,
    )

    evidence_ids = []
    for i in range(n_evidence):
        eid = await semantic.queue_evidence(
            EvidenceDraft(
                source_type="user_statement",
                excerpt=f"I love maths and logic puzzles (signal {i}).",
                weight=0.7,
                hypothesis_hint="analytical_aptitude",
            ),
            turn_id=turn_id,
        )
        evidence_ids.append(eid)
    return evidence_ids


def _distillation_response_json(evidence_ids: list) -> str:
    """Return a realistic distillation LLM response for all evidence_ids."""
    first, rest = evidence_ids[0], evidence_ids[1:]
    matches = [
        {
            "evidence_id": str(first),
            "action": "create_new",
            "statement": "User has a strong analytical aptitude and preference for quantitative reasoning.",
            "initial_confidence": 0.4,
            "confidence_delta": 0.1,
        }
    ]
    # All remaining evidence match the newly created hypothesis
    for eid in rest:
        matches.append(
            {
                "evidence_id": str(eid),
                "action": "create_new",
                "statement": "User has a strong analytical aptitude and preference for quantitative reasoning.",
                "initial_confidence": 0.4,
                "confidence_delta": 0.05,
            }
        )
    return json.dumps({"matches": matches})


async def test_distillation_creates_at_least_one_hypothesis(migrated_db: str) -> None:
    user_id = await _make_user(f"distil_{uuid4().hex[:8]}")
    evidence_ids = await _seed_user_with_evidence(user_id, n_evidence=5)

    mock = MockLLMClient(default_response=_distillation_response_json(evidence_ids))
    factory = LLMFactory()
    factory.register_client("huggingface", mock)

    summary = await run_distillation(user_id, factory=factory)

    assert summary["processed"] == 5
    assert summary["created"] == 1  # Only one unique hypothesis created

    # Verify hypothesis row exists in DB
    semantic = SemanticRepo()
    hyps = await semantic.get_active(user_id)
    assert len(hyps) >= 1
    assert any("analytical" in h.statement.lower() for h in hyps)


async def test_distillation_matches_existing_hypothesis(migrated_db: str) -> None:
    """Evidence queued after an existing hypothesis should match, not create new."""
    user_id = await _make_user(f"distil_match_{uuid4().hex[:8]}")

    semantic = SemanticRepo()
    existing_hyp_id = await semantic.create_hypothesis(
        user_id,
        "User prefers analytical work over interpersonal tasks.",
        confidence=0.5,
    )

    evidence_ids = await _seed_user_with_evidence(user_id, n_evidence=2)

    match_response = json.dumps(
        {
            "matches": [
                {
                    "evidence_id": str(evidence_ids[0]),
                    "action": "match_existing",
                    "hypothesis_id": str(existing_hyp_id),
                    "confidence_delta": 0.1,
                },
                {
                    "evidence_id": str(evidence_ids[1]),
                    "action": "match_existing",
                    "hypothesis_id": str(existing_hyp_id),
                    "confidence_delta": 0.05,
                },
            ]
        }
    )
    mock = MockLLMClient(default_response=match_response)
    factory = LLMFactory()
    factory.register_client("huggingface", mock)

    summary = await run_distillation(user_id, factory=factory)

    assert summary["matched"] == 2
    assert summary["created"] == 0

    # Confidence should have been bumped by 0.1 + 0.05 = 0.15
    pool = await get_pool()
    async with pool.acquire() as conn:
        conf = await conn.fetchval(
            "SELECT confidence FROM hypotheses WHERE hypothesis_id = $1", existing_hyp_id
        )
    assert float(conf) == pytest.approx(0.65, abs=0.01)


async def test_distillation_noop_when_no_evidence(migrated_db: str) -> None:
    user_id = await _make_user(f"distil_empty_{uuid4().hex[:8]}")
    mock = MockLLMClient()
    factory = LLMFactory()
    factory.register_client("huggingface", mock)

    summary = await run_distillation(user_id, factory=factory)

    assert summary["processed"] == 0
    assert len(mock.calls) == 0  # No LLM call when nothing to process
