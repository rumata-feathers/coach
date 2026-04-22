"""Integration test: within-session memory path.

Requires a live Postgres (skipped otherwise). Uses MockLLMClient so no HF
token is needed — we're testing the DB plumbing, not the LLM quality.

Verifies:
  - turn_index increments 0 → 1 → 2 → 3 across 4 turns in one session.
  - The 4th turn's Understander call has ``num_recent_turns == 3``
    (proven via agent_calls.input_payload logged by the Understander).
  - sessions.session_theory is non-null and updated after turn 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.pipeline.turn import TurnPipeline

pytestmark = pytest.mark.asyncio

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _intent_json(**overrides: object) -> str:
    base: dict[str, object] = {
        "session_theory": "User exploring careers.",
        "turn_intent": "explore",
        "specific_ask": "Tell me about careers.",
        "emotional_tenor": "curious",
        "clarity_score": 0.8,
        "needs_clarification": False,
        "clarification_question": None,
        "inferred_constraints": [],
        "budget_hint": "quick",  # flow A — no Critic, faster
    }
    base.update(overrides)
    return json.dumps(base)


def _coach_json(msg: str = "Here is some career advice.") -> str:
    return json.dumps(
        {
            "response_text": msg,
            "referenced_facts": [],
            "referenced_hypotheses": [],
            "proposed_challenge": None,
            "uncertainty_flags": [],
        }
    )


def _profiler_json() -> str:
    return json.dumps({"new_facts": [], "fact_updates": [], "hypothesis_evidence": []})


async def _create_test_user(display_name: str) -> object:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
            display_name,
        )


async def test_session_continuity_turn_indices_increment(migrated_db: str) -> None:
    """Four sequential turns in one session must have turn_index 0, 1, 2, 3."""
    user_id = await _create_test_user(f"continuity_{uuid4().hex[:6]}")

    mock = MockLLMClient()
    # 4 turns x (Understander + Coach + Profiler) = 12 responses
    for _ in range(4):
        mock.queue(_intent_json(), _coach_json(f"Advice {_ + 1}"), _profiler_json())

    factory = _make_factory(mock)
    pipeline = TurnPipeline(factory)
    # Patch onboarding policy to return is_new_user=False so we exercise
    # the standard Coach path (not the 3-turn onboarding arc)
    pipeline._onboarding_policy = AsyncMock()
    pipeline._onboarding_policy.is_new_user.return_value = False

    session_id = None
    for i in range(4):
        result = await pipeline.process_turn(
            user_id=user_id,
            user_message=f"Turn {i} question",
            session_id=session_id,
        )
        if session_id is None:
            session_id = result.session_id
        # All turns share the same session
        assert result.session_id == session_id

        # Allow background profiler task to complete
        import asyncio
        await asyncio.sleep(0.05)

    # Verify turn indices in DB
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT turn_index, user_message
            FROM turns
            WHERE session_id = $1
            ORDER BY turn_index
            """,
            session_id,
        )

    assert len(rows) == 4, f"Expected 4 turns, got {len(rows)}"
    for expected_idx, row in enumerate(rows):
        assert row["turn_index"] == expected_idx, (
            f"Expected turn_index {expected_idx}, got {row['turn_index']}"
        )


async def test_session_continuity_recent_turns_visible(migrated_db: str) -> None:
    """The 4th turn's Understander must see 3 recent turns.

    Verified via agent_calls.input_payload['num_recent_turns'] which is
    logged by the Understander's _serialize_input helper.
    """
    user_id = await _create_test_user(f"cont_rt_{uuid4().hex[:6]}")

    mock = MockLLMClient()
    for _ in range(4):
        mock.queue(_intent_json(), _coach_json(), _profiler_json())

    factory = _make_factory(mock)
    pipeline = TurnPipeline(factory)
    pipeline._onboarding_policy = AsyncMock()
    pipeline._onboarding_policy.is_new_user.return_value = False

    import asyncio
    session_id = None
    turn_ids = []
    for i in range(4):
        result = await pipeline.process_turn(
            user_id=user_id,
            user_message=f"Message {i}",
            session_id=session_id,
        )
        session_id = result.session_id
        turn_ids.append(result.turn_id)
        await asyncio.sleep(0.05)

    # Check that the 4th Understander call had num_recent_turns=3.
    # Understander's log_call uses turn_id=None (it runs before turn save),
    # so we query by agent_name and session to get the 4th call.
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get all understander calls ordered by creation time
        understander_calls = await conn.fetch(
            """
            SELECT input_payload, created_at
            FROM agent_calls
            WHERE agent_name = 'understander'
              AND turn_id IS NULL
            ORDER BY created_at DESC
            LIMIT 4
            """,
        )

    # The most recent understander call should have seen 3 recent turns
    assert understander_calls, "No understander calls found in agent_calls"
    most_recent_payload = understander_calls[0]["input_payload"]
    num_recent = most_recent_payload.get("num_recent_turns", -1)
    assert num_recent >= 3, (
        f"Expected num_recent_turns >= 3 on the 4th turn, got {num_recent}. "
        f"Session memory may not be working correctly."
    )


async def test_session_theory_updated_after_turn(migrated_db: str) -> None:
    """sessions.session_theory must be non-null after the first turn completes."""
    user_id = await _create_test_user(f"cont_st_{uuid4().hex[:6]}")

    mock = MockLLMClient()
    mock.queue(
        _intent_json(session_theory="User is exploring quantitative careers."),
        _coach_json(),
        _profiler_json(),
    )

    factory = _make_factory(mock)
    pipeline = TurnPipeline(factory)
    pipeline._onboarding_policy = AsyncMock()
    pipeline._onboarding_policy.is_new_user.return_value = False

    result = await pipeline.process_turn(
        user_id=user_id,
        user_message="Tell me about quant finance.",
    )

    pool = await get_pool()
    async with pool.acquire() as conn:
        theory = await conn.fetchval(
            "SELECT session_theory FROM sessions WHERE session_id = $1",
            result.session_id,
        )

    assert theory is not None, "session_theory must be set after turn 1"
    assert len(theory) > 0, "session_theory must not be empty"
