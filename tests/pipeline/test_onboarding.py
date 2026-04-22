"""Unit tests for :class:`OnboardingPolicy` and :class:`OnboardingPipeline`.

All DB and LLM calls are mocked — no real API or database needed.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import CoachOutput
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import TurnSummary
from career_coach.pipeline.onboarding import OnboardingPipeline, OnboardingPolicy

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

_FAKE_USER_ID = uuid4()
_FAKE_SESSION_ID = uuid4()


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _intent() -> IntentPacket:
    return IntentPacket(
        session_theory="User asking about careers.",
        turn_intent="explore",
        specific_ask="Should I study economics?",
        emotional_tenor="curious",
        clarity_score=0.8,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=[],
        budget_hint="standard",
    )


def _onboarder_json(response_text: str = "Tell me about yourself?") -> str:
    return json.dumps(
        {
            "response_text": response_text,
            "referenced_facts": [],
            "referenced_hypotheses": [],
            "proposed_challenge": None,
            "uncertainty_flags": [],
        }
    )


def _make_summary(index: int, user_msg: str, asst_msg: str) -> TurnSummary:
    from datetime import datetime

    return TurnSummary(
        turn_id=uuid4(),
        turn_index=index,
        user_message=user_msg,
        assistant_message=asst_msg,
        intent="explore",
        created_at=datetime.now(UTC),
    )


def _mock_pool(fetchval_side_effects: list[int]) -> MagicMock:
    """Return a mock asyncpg pool whose connection's fetchval returns the given values."""
    mock_conn = AsyncMock()
    mock_conn.fetchval.side_effect = fetchval_side_effects

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)

    mock_get_pool = AsyncMock(return_value=pool)
    return mock_get_pool


# ---- OnboardingPolicy -------------------------------------------------------


async def test_onboarding_policy_new_user_few_facts() -> None:
    """User with < 5 facts and < 3 turns is new."""
    policy = OnboardingPolicy()
    with patch(
        "career_coach.pipeline.onboarding.get_pool",
        new=_mock_pool([2, 1]),  # fact_count=2, turn_count=1
    ):
        result = await policy.is_new_user(_FAKE_USER_ID)
    assert result is True


async def test_onboarding_policy_graduated_by_facts() -> None:
    """User with >= 5 facts graduates from onboarding regardless of turn count."""
    policy = OnboardingPolicy()
    with patch(
        "career_coach.pipeline.onboarding.get_pool",
        new=_mock_pool([5]),  # fact_count=5 → short-circuits immediately
    ):
        result = await policy.is_new_user(_FAKE_USER_ID)
    assert result is False


async def test_onboarding_policy_graduated_by_turns() -> None:
    """User with < 5 facts but >= 3 turns graduates from onboarding."""
    policy = OnboardingPolicy()
    with patch(
        "career_coach.pipeline.onboarding.get_pool",
        new=_mock_pool([3, 3]),  # fact_count=3, turn_count=3
    ):
        result = await policy.is_new_user(_FAKE_USER_ID)
    assert result is False


# ---- OnboardingPipeline ------------------------------------------------------


async def test_onboarding_pipeline_returns_coach_output() -> None:
    """process_turn must return a valid CoachOutput."""
    mock = MockLLMClient(default_response=_onboarder_json("Tell me about yourself!"))

    pipeline = OnboardingPipeline(_make_factory(mock))
    with (
        patch.object(pipeline, "_get_onboarding_turn_index", return_value=0),
        patch.object(pipeline, "log_call", new=AsyncMock()),
    ):
        output = await pipeline.process_turn(
            user_id=_FAKE_USER_ID,
            session_id=_FAKE_SESSION_ID,
            user_message="Should I study economics?",
            intent_packet=_intent(),
            user_facts={"age": 19},
            recent_turns=[],
        )

    assert isinstance(output, CoachOutput)
    assert output.response_text == "Tell me about yourself!"


async def test_onboarding_pipeline_turn_index_0_uses_context_probe() -> None:
    """Turn index 0 prompt must mention context-gathering language."""
    mock = MockLLMClient(default_response=_onboarder_json())

    pipeline = OnboardingPipeline(_make_factory(mock))
    with (
        patch.object(pipeline, "_get_onboarding_turn_index", return_value=0),
        patch.object(pipeline, "log_call", new=AsyncMock()),
    ):
        await pipeline.process_turn(
            user_id=_FAKE_USER_ID,
            session_id=_FAKE_SESSION_ID,
            user_message="Should I study economics?",
            intent_packet=_intent(),
            user_facts={},
            recent_turns=[],
        )

    prompt = mock.calls[0].messages[0].content
    assert "Turn 0" in prompt
    assert "context" in prompt.lower() or "studying" in prompt.lower()


async def test_onboarding_pipeline_turn_index_1_uses_values_probe() -> None:
    """Turn index 1 prompt must mention the forced-choice values scenario."""
    mock = MockLLMClient(default_response=_onboarder_json())

    pipeline = OnboardingPipeline(_make_factory(mock))
    with (
        patch.object(pipeline, "_get_onboarding_turn_index", return_value=1),
        patch.object(pipeline, "log_call", new=AsyncMock()),
    ):
        await pipeline.process_turn(
            user_id=_FAKE_USER_ID,
            session_id=_FAKE_SESSION_ID,
            user_message="I just finished an internship at a bank.",
            intent_packet=_intent(),
            user_facts={"age": 19},
            recent_turns=[_make_summary(0, "I study maths", "Tell me more!")],
        )

    prompt = mock.calls[0].messages[0].content
    assert "Turn 1" in prompt
    # The values probe must mention the forced-choice scenario
    assert "two careers" in prompt.lower() or "scenario" in prompt.lower()


async def test_onboarding_pipeline_retries_on_empty_response() -> None:
    """First call returns empty → retry → second call returns valid JSON."""
    mock = MockLLMClient()
    mock.queue("", _onboarder_json("Context probe response."))

    pipeline = OnboardingPipeline(_make_factory(mock))
    with (
        patch.object(pipeline, "_get_onboarding_turn_index", return_value=0),
        patch.object(pipeline, "log_call", new=AsyncMock()),
    ):
        output = await pipeline.process_turn(
            user_id=_FAKE_USER_ID,
            session_id=_FAKE_SESSION_ID,
            user_message="Should I study economics?",
            intent_packet=_intent(),
            user_facts={},
            recent_turns=[],
        )

    assert len(mock.calls) == 2
    assert isinstance(output, CoachOutput)
    assert output.response_text == "Context probe response."
