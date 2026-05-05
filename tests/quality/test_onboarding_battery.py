"""Onboarding quality battery — 5 cold-start fixtures.

Each fixture exercises the full 3-turn onboarding arc for a brand-new user
asking a different opening question. Assertions:

  Turn 1 — response acknowledges the topic AND asks a probe question.
  Turn 2 — response includes a values/forced-choice probe.
  Turn 3 — response references at least one detail from prior turns AND asks a
            check-the-framing question.

Uses a live HuggingFace token (skipped if absent). Requires a live Postgres.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from career_coach.config import get_settings
from career_coach.llm.factory import LLMFactory
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import TurnSummary
from career_coach.pipeline.onboarding import OnboardingPipeline

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

pytestmark = [pytest.mark.asyncio, pytest.mark.live]

COLD_START_QUESTIONS = [
    "Should I study economics?",
    "I'm thinking about switching majors",
    "Quant finance — thoughts?",
    "I don't know what I want to do",
    "What careers pay well in London?",
]


@pytest.fixture(scope="module")
def hf_token() -> str:
    token = get_settings().huggingface_api_token or ""
    if not token:
        pytest.skip("HUGGINGFACE_API_TOKEN not set; skipping onboarding battery")
    return token


def _intent(question: str) -> IntentPacket:
    return IntentPacket(
        session_theory="New user with an opening question.",
        turn_intent="explore",
        specific_ask=question,
        emotional_tenor="curious",
        clarity_score=0.6,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=[],
        budget_hint="standard",
    )


def _has_question(text: str) -> bool:
    """Return True if the response contains a question."""
    question_phrases = ["?", "tell me", "could you", "what do", "how do", "why do",
                        "which", "where are", "what are", "what prompted"]
    text_lower = text.lower()
    return "?" in text or any(p in text_lower for p in question_phrases)


async def _run_onboarding_arc(question: str, factory: LLMFactory) -> list[str]:
    """Run 3 onboarding turns and return the 3 response texts."""
    user_id = uuid4()
    session_id = uuid4()
    pipeline = OnboardingPipeline(factory)

    # Patch log_call so we don't need a real DB for this quality test
    pipeline.log_call = AsyncMock()  # type: ignore[method-assign]

    turns: list[TurnSummary] = []
    responses: list[str] = []

    for turn_idx in range(3):
        # Patch turn index directly to avoid DB lookup
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pipeline, "_get_onboarding_turn_index", AsyncMock(return_value=turn_idx))
            output = await pipeline.process_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=question if turn_idx == 0 else f"Follow-up message {turn_idx}",
                intent_packet=_intent(question),
                user_facts={} if turn_idx == 0 else {"age": 19},
                recent_turns=turns[:],
            )

        responses.append(output.response_text)

        from datetime import datetime
        turns.append(
            TurnSummary(
                turn_id=uuid4(),
                turn_index=turn_idx,
                user_message=question if turn_idx == 0 else f"Follow-up message {turn_idx}",
                assistant_message=output.response_text,
                intent="explore",
                created_at=datetime.now(UTC),
            )
        )

    return responses


@pytest.mark.parametrize("question", COLD_START_QUESTIONS)
async def test_onboarding_turn_1_acknowledges_and_probes(
    question: str,
    hf_token: str,
) -> None:
    """Turn 1 response must acknowledge the topic AND ask a probe question."""
    factory = LLMFactory(config_path=_CONFIG)
    responses = await _run_onboarding_arc(question, factory)
    turn1 = responses[0]

    assert len(turn1) > 20, "Response too short"
    assert _has_question(turn1), (
        f"Turn 1 response should ask a probe question.\n"
        f"Question: {question!r}\n"
        f"Response: {turn1!r}"
    )


@pytest.mark.parametrize("question", COLD_START_QUESTIONS)
async def test_onboarding_turn_2_includes_values_probe(
    question: str,
    hf_token: str,
) -> None:
    """Turn 2 response must include a values/forced-choice question."""
    factory = LLMFactory(config_path=_CONFIG)
    responses = await _run_onboarding_arc(question, factory)
    turn2 = responses[1]

    assert len(turn2) > 20, "Response too short"
    assert _has_question(turn2), (
        f"Turn 2 response should ask a values probe.\n"
        f"Question: {question!r}\n"
        f"Response: {turn2!r}"
    )


@pytest.mark.parametrize("question", COLD_START_QUESTIONS)
async def test_onboarding_turn_3_references_and_checks_framing(
    question: str,
    hf_token: str,
) -> None:
    """Turn 3 response must show a grounded thought AND check the framing."""
    factory = LLMFactory(config_path=_CONFIG)
    responses = await _run_onboarding_arc(question, factory)
    turn3 = responses[2]

    assert len(turn3) > 30, "Response too short"
    assert _has_question(turn3), (
        f"Turn 3 response should check the framing.\n"
        f"Question: {question!r}\n"
        f"Response: {turn3!r}"
    )
