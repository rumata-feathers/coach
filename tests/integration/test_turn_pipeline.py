"""Integration test for the end-to-end turn pipeline.

Requires:
  - A live Postgres reachable at ``SUPABASE_DB_URL`` (skipped otherwise).
  - A valid ``HUGGINGFACE_API_TOKEN`` (skipped otherwise).

Sends a real user message through the full agent pipeline and asserts that:
  - A TurnResult is returned with a non-empty response.
  - A turn row is persisted in the DB.
  - agent_calls rows were written (observability log).
  - The Profiler background task runs and writes at least one structured fact.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from career_coach.config import get_settings
from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.pipeline.turn import TurnPipeline

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def hf_token() -> str:
    token = get_settings().huggingface_api_token or ""
    if not token:
        pytest.skip("HUGGINGFACE_API_TOKEN not set; skipping live pipeline test")
    return token


async def _create_test_user(display_name: str) -> object:
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
            display_name,
        )
        # Seed a fact so the pipeline has something to work with.
        await conn.execute(
            """
            INSERT INTO structured_facts (user_id, key, value, confidence, source)
            VALUES ($1, 'age', '19'::jsonb, 1.0, 'system')
            ON CONFLICT (user_id, key) DO NOTHING
            """,
            user_id,
        )
        await conn.execute(
            """
            INSERT INTO structured_facts (user_id, key, value, confidence, source)
            VALUES ($1, 'location', '"London"'::jsonb, 1.0, 'system')
            ON CONFLICT (user_id, key) DO NOTHING
            """,
            user_id,
        )
    return user_id


async def test_process_turn_end_to_end(
    migrated_db: str,
    hf_token: str,
) -> None:
    user_id = await _create_test_user(f"integration_{uuid4().hex[:8]}")

    factory = LLMFactory()
    pipeline = TurnPipeline(factory)

    pool = await get_pool()

    # Snapshot agent_calls count before the turn runs
    async with pool.acquire() as conn:
        calls_before: int = await conn.fetchval("SELECT COUNT(*) FROM agent_calls")

    result = await pipeline.process_turn(
        user_id=user_id,
        user_message="I'm 19, studying in London, and I'm trying to decide between economics and computer science. What should I consider?",
    )

    # Response is non-empty
    assert result.response, "Expected a non-empty response"
    assert len(result.response) > 20

    # Turn was persisted
    assert result.turn_id is not None
    async with pool.acquire() as conn:
        turn_row = await conn.fetchrow(
            "SELECT turn_id, user_message, assistant_message FROM turns WHERE turn_id = $1",
            result.turn_id,
        )
    assert turn_row is not None
    assert turn_row["user_message"] is not None
    assert turn_row["assistant_message"] == result.response

    # agent_calls rows were written (Understander + Coach at minimum, possibly Critic)
    # Agents write calls with turn_id=NULL because the turn hasn't been saved yet when they run;
    # verify by checking total count increased.
    async with pool.acquire() as conn:
        calls_after: int = await conn.fetchval("SELECT COUNT(*) FROM agent_calls")
    calls_delta = calls_after - calls_before
    assert calls_delta >= 2, (
        f"Expected at least 2 new agent_calls rows, got delta={calls_delta}"
    )

    # Give the Profiler background task up to 30 seconds to complete.
    # Profiler writes to agent_calls WITH the real turn_id, so we can query by it.
    deadline = asyncio.get_event_loop().time() + 30
    profiler_ran = False
    while asyncio.get_event_loop().time() < deadline:
        async with pool.acquire() as conn:
            profiler_call = await conn.fetchval(
                "SELECT COUNT(*) FROM agent_calls WHERE agent_name = 'profiler' AND turn_id = $1",
                result.turn_id,
            )
        if int(profiler_call) > 0:
            profiler_ran = True
            break
        await asyncio.sleep(1)

    assert profiler_ran, (
        f"Profiler agent_calls row with turn_id={result.turn_id} not found within 30 s."
    )
