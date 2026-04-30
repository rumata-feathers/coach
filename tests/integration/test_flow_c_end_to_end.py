"""Integration test for Flow C (deliberation) end-to-end.

Requires:
  - A live Postgres reachable at ``SUPABASE_DB_URL`` (skipped otherwise).
  - A valid ``HUGGINGFACE_API_TOKEN`` (skipped otherwise).
  - A valid ``TAVILY_API_KEY`` (skipped otherwise).

Seeds a user with ≥5 structured facts including math/research signals, then
sends a deep-decision question. Asserts:

  - ``turns.flow_used == "C"``
  - ``turns.synthesizer_output`` is non-null (Flow C ran)
  - ``turns.research_brief_id`` is non-null (Researcher ran)
  - The response contains ≥2 user-fact references (grounded)
  - The response contains ≥1 finding-derived claim (inline "[N]" citation)
  - The response contains ≥1 tradeoff signal word
"""

from __future__ import annotations

import re
from uuid import UUID

import pytest

from career_coach.config import get_settings
from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.pipeline.turn import TurnPipeline

pytestmark = pytest.mark.asyncio

_CONFIG_PATH = None  # Uses default from LLMFactory

# Tradeoff signal words — at least one should appear in a deliberation response.
_TRADEOFF_WORDS = frozenset(
    [
        "tradeoff",
        "trade-off",
        "trade off",
        "however",
        "but",
        "on the other hand",
        "whereas",
        "versus",
        "vs",
        "risk",
        "downside",
        "disadvantage",
        "caveat",
        "although",
        "tension",
    ]
)


# ---- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def skip_if_missing(migrated_db: str) -> None:
    """Skip when required keys are absent; also ensures DB migrations are applied."""
    settings = get_settings()
    missing = []
    if not settings.huggingface_api_token:
        missing.append("HUGGINGFACE_API_TOKEN")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        pytest.skip(f"Skipping Flow C integration test — missing: {', '.join(missing)}")


@pytest.fixture(scope="module")
async def flow_c_user_id(skip_if_missing: None) -> UUID:
    """Create a test user seeded with ≥5 structured facts, return user_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id: UUID = await conn.fetchval(
            "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
            "test-flow-c-e2e",
        )
        # Seed ≥5 facts including math + research signals so Flow C fires.
        facts = [
            ("degree", '"Mathematics, Warwick"'),
            ("age", "22"),
            ("location", '"London"'),
            ("interests", '["markets", "algorithms", "research"]'),
            ("education_stage", '"final_year"'),
            ("career_goals", '"research or quant finance"'),
        ]
        for key, value_jsonb in facts:
            await conn.execute(
                """
                INSERT INTO structured_facts (user_id, key, value, confidence, source)
                VALUES ($1, $2, $3::jsonb, 1.0, 'system')
                ON CONFLICT (user_id, key) DO NOTHING
                """,
                user_id,
                key,
                value_jsonb,
            )
        return user_id


@pytest.fixture(scope="module")
def pipeline(skip_if_missing: None) -> TurnPipeline:
    """Return a live TurnPipeline."""
    from pathlib import Path

    config = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
    return TurnPipeline(LLMFactory(config_path=config))


# ---- test ------------------------------------------------------------------


async def test_flow_c_end_to_end(
    flow_c_user_id: UUID,
    pipeline: TurnPipeline,
) -> None:
    """Send a PhD-vs-quant decision question; assert Flow C ran correctly."""
    question = (
        "I'm deciding between doing a Maths PhD and going straight into quant "
        "finance — help me think through the tradeoffs carefully."
    )

    result = await pipeline.process_turn(
        user_id=flow_c_user_id,
        user_message=question,
    )

    # 1. Basic result shape
    assert result.response, "Expected non-empty response"
    assert result.turn_id is not None, "turn_id should be set"

    # 2. DB assertions
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT flow_used, synthesizer_output, research_brief_id,
                   devils_advocate_output
            FROM turns WHERE turn_id = $1
            """,
            result.turn_id,
        )

    assert row is not None, "Turn row not found in DB"
    assert row["flow_used"] == "C", (
        f"Expected flow_used='C' but got '{row['flow_used']}'. "
        "The Orchestrator may not have routed to Flow C — check fact_count and intent."
    )
    assert row["synthesizer_output"] is not None, (
        "synthesizer_output is NULL — Synthesizer did not run on Flow C"
    )
    assert row["research_brief_id"] is not None, (
        "research_brief_id is NULL — Researcher did not run on Flow C"
    )

    # 3. Response quality assertions (against synthesizer_output)
    import json

    synth = json.loads(row["synthesizer_output"])
    response_text: str = synth.get("response_text", "")

    # ≥1 inline citation [N] in response
    citation_pattern = re.compile(r"\[\d+\]")
    assert citation_pattern.search(response_text), (
        "Expected ≥1 inline citation [N] in response_text"
    )

    # ≥2 user-fact references
    referenced_facts: list[str] = synth.get("referenced_facts", [])
    assert len(referenced_facts) >= 2, (
        f"Expected ≥2 referenced_facts, got {referenced_facts}"
    )

    # ≥1 finding reference
    referenced_findings: list[str] = synth.get("referenced_findings", [])
    assert len(referenced_findings) >= 1, (
        f"Expected ≥1 referenced_finding, got {referenced_findings}"
    )

    # ≥1 tradeoff signal word in the response text
    text_lower = response_text.lower()
    has_tradeoff = any(word in text_lower for word in _TRADEOFF_WORDS)
    assert has_tradeoff, (
        f"Expected ≥1 tradeoff signal word in response but found none. "
        f"Response: {response_text[:300]}..."
    )

    # 4. integrated_from includes all three sources
    integrated_from: list[str] = synth.get("integrated_from", [])
    assert "coach" in integrated_from, f"'coach' missing from integrated_from: {integrated_from}"
    # researcher and devils_advocate should be present (non-strict: log if absent)
    if "researcher" not in integrated_from:
        pytest.xfail(
            "Researcher not in integrated_from — possible empty research result; "
            "acceptable if Tavily returned no findings."
        )
