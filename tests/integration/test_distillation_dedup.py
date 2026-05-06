"""Integration test: cross-run hypothesis deduplication in the distillation job.

Verifies spec §6.4 — before creating a new hypothesis the distillation job
asks a cheap LLM call whether an existing hypothesis already covers the same
trait. If yes, the evidence is redirected to the existing hypothesis instead
of spawning a duplicate.

Two test scenarios:
  1. Direct dedup — pre-create a hypothesis, run distillation once with a mock
     that proposes a similar new one; assert the dedup check fires and
     ``created == 0``.

  2. Two-run dedup — no pre-existing hypothesis; run distillation twice on
     similar evidence; assert active hypothesis count ≤ 2 after the second run
     (i.e., dedup prevented an exact duplicate accumulating).

Uses a real Postgres (skipped otherwise) and MockLLMClient (no HF token needed).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_user(name: str) -> object:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "INSERT INTO users (display_name, is_test) VALUES ($1, TRUE) RETURNING user_id", name
        )


async def _seed_evidence(user_id: object, n: int = 3) -> list[object]:
    """Create a session + turn, queue n evidence rows, return evidence IDs."""
    episodic = EpisodicRepo()
    semantic = SemanticRepo()

    session = await episodic.create_session(user_id)
    turn_id = await episodic.save_turn(
        session_id=session.session_id,
        user_id=user_id,
        user_message="I prefer working alone on hard problems.",
        assistant_message="Noted.",
        intent_packet={"turn_intent": "explore"},
        flow_used="B",
        critic_verdicts=None,
        tokens_used=None,
    )

    evidence_ids = []
    for i in range(n):
        eid = await semantic.queue_evidence(
            EvidenceDraft(
                source_type="user_statement",
                excerpt=f"I prefer solo analytical work, signal {i}",
                weight=0.6,
                hypothesis_hint="prefers_solo_work",
            ),
            turn_id=turn_id,
        )
        evidence_ids.append(eid)
    return evidence_ids


def _main_distillation_response(evidence_ids: list[object], statement: str) -> str:
    """Return a distillation LLM response that proposes create_new for all evidence."""
    first = evidence_ids[0]
    rest = evidence_ids[1:]
    matches: list[dict] = [
        {
            "evidence_id": str(first),
            "action": "create_new",
            "statement": statement,
            "initial_confidence": 0.35,
            "confidence_delta": 0.08,
        }
    ]
    for eid in rest:
        matches.append(
            {
                "evidence_id": str(eid),
                "action": "create_new",
                "statement": statement,
                "initial_confidence": 0.35,
                "confidence_delta": 0.05,
            }
        )
    return json.dumps({"matches": matches})


def _dedup_yes_response(existing_id: object) -> str:
    """Return a dedup-check response that says 'duplicate, use this ID'."""
    return json.dumps(
        {
            "duplicate": True,
            "existing_id": str(existing_id),
            "reason": "Same underlying preference for solo / individual work.",
        }
    )


def _dedup_no_response() -> str:
    """Return a dedup-check response that says 'not a duplicate'."""
    return json.dumps({"duplicate": False, "existing_id": None, "reason": "Distinct trait."})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_distillation_dedup_prevents_duplicate_hypothesis(
    migrated_db: str,
) -> None:
    """Pre-create a hypothesis, then run distillation with a similar create_new
    proposal. Dedup check must fire, redirect to the existing hypothesis,
    and leave ``created == 0``.
    """
    user_id = await _make_user(f"dedup_direct_{uuid4().hex[:6]}")
    semantic = SemanticRepo()

    # Pre-create hypothesis
    existing_hyp_id = await semantic.create_hypothesis(
        user_id,
        "User strongly prefers working alone on intellectually demanding problems.",
        confidence=0.45,
    )

    evidence_ids = await _seed_evidence(user_id, n=2)

    # Distillation proposes a semantically similar (but differently worded) hypothesis.
    # The dedup check should intercept and redirect to existing_hyp_id.
    similar_statement = "User has a marked preference for solo analytical work over collaboration."
    mock = MockLLMClient()
    mock.queue(
        # Call 1: main distillation response
        _main_distillation_response(evidence_ids, similar_statement),
        # Call 2: dedup check for the first create_new proposal
        _dedup_yes_response(existing_hyp_id),
    )
    factory = LLMFactory()
    factory.register_client("huggingface", mock)

    summary = await run_distillation(user_id, factory=factory)

    # Dedup must have fired and redirected
    assert summary["created"] == 0, (
        f"Expected created==0 (dedup should have redirected), got {summary['created']}"
    )
    assert summary["deduped"] >= 1, (
        f"Expected deduped>=1, got {summary['deduped']}"
    )

    # Still only the original hypothesis in the DB
    hyps = await semantic.get_active(user_id)
    assert len(hyps) == 1, f"Expected 1 active hypothesis (no duplicate), got {len(hyps)}"
    assert hyps[0].hypothesis_id == existing_hyp_id


async def test_distillation_two_runs_count_stays_low(
    migrated_db: str,
) -> None:
    """Run distillation twice on similar evidence; assert active hypothesis
    count stays ≤ 2. The second run's create_new proposals should be deduped
    against the hypothesis created in the first run.
    """
    user_id = await _make_user(f"dedup_tworun_{uuid4().hex[:6]}")
    semantic = SemanticRepo()

    statement = "User prefers deep solo work on abstract problems."

    # ── First run ────────────────────────────────────────────────────────────
    evidence_ids_1 = await _seed_evidence(user_id, n=2)

    # First run: no existing hypotheses → dedup check skipped → hypothesis created
    mock1 = MockLLMClient(default_response=_main_distillation_response(evidence_ids_1, statement))
    factory1 = LLMFactory()
    factory1.register_client("huggingface", mock1)

    summary1 = await run_distillation(user_id, factory=factory1)
    assert summary1["created"] == 1, f"First run should create 1 hypothesis, got {summary1}"

    # Fetch the ID of the newly created hypothesis
    hyps_after_run1 = await semantic.get_active(user_id)
    assert len(hyps_after_run1) == 1
    first_hyp_id = hyps_after_run1[0].hypothesis_id

    # ── Second run ───────────────────────────────────────────────────────────
    evidence_ids_2 = await _seed_evidence(user_id, n=2)

    # Second run: proposes the same statement again → dedup check fires →
    # mock says "yes, duplicate of first_hyp_id" → redirected
    similar_statement_2 = "User has a deep preference for solitary analytical problem-solving."
    mock2 = MockLLMClient()
    mock2.queue(
        # Main distillation response
        _main_distillation_response(evidence_ids_2, similar_statement_2),
        # Dedup check response (called once, for first create_new in the batch)
        _dedup_yes_response(first_hyp_id),
    )
    factory2 = LLMFactory()
    factory2.register_client("huggingface", mock2)

    summary2 = await run_distillation(user_id, factory=factory2)
    assert summary2["created"] == 0, (
        f"Second run should not create any new hypotheses (dedup), got {summary2}"
    )
    assert summary2["deduped"] >= 1, f"Expected deduped>=1 on second run, got {summary2}"

    # Total active hypothesis count must still be 1
    hyps_after_run2 = await semantic.get_active(user_id)
    assert len(hyps_after_run2) <= 2, (
        f"Expected ≤2 active hypotheses after 2 runs, got {len(hyps_after_run2)}"
    )
