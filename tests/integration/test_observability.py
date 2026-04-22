"""Integration test: retry_count and fallback_reason land in agent_calls.

Requires a live Postgres (skipped otherwise). Does NOT require a real HF
token — uses MockLLMClient injected directly into the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.memory.episodic import Session
from career_coach.pipeline.turn import TurnPipeline

pytestmark = pytest.mark.asyncio

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _intent_json() -> str:
    return json.dumps(
        {
            "session_theory": "User exploring career options.",
            "turn_intent": "explore",
            "specific_ask": "Should I study CS?",
            "emotional_tenor": "curious",
            "clarity_score": 0.8,
            "needs_clarification": False,
            "clarification_question": None,
            "inferred_constraints": [],
            "budget_hint": "quick",  # flow A — no Critic
        }
    )


def _coach_json() -> str:
    return json.dumps(
        {
            "response_text": "CS is a solid choice.",
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


@pytest_asyncio.fixture()
async def pipeline_with_db(migrated_db: str) -> TurnPipeline:
    """Return a TurnPipeline wired against the real DB but with a MockLLMClient."""
    mock = MockLLMClient(default_response=_coach_json())
    factory = _make_factory(mock)
    pipeline = TurnPipeline(factory)
    return pipeline


async def test_retry_exhausted_written_to_agent_calls(migrated_db: str) -> None:
    """When both LLM attempts return empty JSON, agent_calls row must have
    retry_count=1 and fallback_reason='retry_exhausted'.

    Uses flow A (no Critic) so only Understander and Coach are called.
    We make the Understander return empty twice → retry_exhausted.
    Coach returns valid JSON → no fallback.
    """
    user_id = await _create_test_user(f"obs_test_{uuid4().hex[:6]}")

    mock = MockLLMClient()
    # Understander: empty → empty (both attempts fail → fallback)
    # Coach: valid on first attempt (flow A, no Critic)
    # Profiler: valid
    mock.queue(
        "",          # Understander attempt 1 — empty → triggers retry
        "",          # Understander attempt 2 — empty → fallback
        _coach_json(),   # Coach
        _profiler_json(),  # Profiler (background)
    )
    factory = _make_factory(mock)
    pipeline = TurnPipeline(factory)

    pool = await get_pool()
    fake_session = Session(session_id=uuid4(), user_id=user_id, session_theory=None)
    pipeline._episodic = AsyncMock()
    pipeline._episodic.get_session.return_value = fake_session
    pipeline._episodic.create_session.return_value = fake_session
    pipeline._episodic.get_recent.return_value = []
    pipeline._episodic.save_turn.return_value = uuid4()
    pipeline._episodic.update_session_theory.return_value = None
    pipeline._structured = AsyncMock()
    pipeline._structured.get_all.return_value = {}
    pipeline._structured.bulk_upsert.return_value = None
    pipeline._semantic = AsyncMock()
    pipeline._semantic.get_active.return_value = []
    pipeline._semantic.queue_evidence = AsyncMock(return_value=None)

    await pipeline.process_turn(
        user_id=user_id,
        user_message="Should I study CS?",
    )

    # Wait briefly for the background Profiler task to complete.
    import asyncio
    await asyncio.sleep(0.1)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT retry_count, fallback_reason
            FROM agent_calls
            WHERE agent_name = 'understander'
            ORDER BY created_at DESC
            LIMIT 1
            """,
        )

    assert row is not None, "No understander row found in agent_calls"
    assert row["retry_count"] == 1, (
        f"Expected retry_count=1, got {row['retry_count']}"
    )
    assert row["fallback_reason"] == "retry_exhausted", (
        f"Expected fallback_reason='retry_exhausted', got {row['fallback_reason']!r}"
    )
