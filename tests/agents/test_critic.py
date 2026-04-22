"""Unit tests for :class:`Critic` with :class:`MockLLMClient`."""

from __future__ import annotations

import json
from pathlib import Path

from career_coach.agents.critic import Critic
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import CoachOutput, CriticInput, CriticVerdict
from career_coach.models.intent import IntentPacket

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _intent() -> IntentPacket:
    return IntentPacket(
        session_theory="test",
        turn_intent="decide",
        specific_ask="Should I study economics?",
        emotional_tenor="uncertain",
        clarity_score=0.8,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=[],
        budget_hint="standard",
    )


def _pass_verdict() -> str:
    return json.dumps(
        {
            "verdict": "pass",
            "failure_modes": [],
            "specific_complaints": [],
            "suggested_fix": None,
        }
    )


def _reject_verdict(modes: list[str], complaints: list[str], fix: str) -> str:
    return json.dumps(
        {
            "verdict": "reject",
            "failure_modes": modes,
            "specific_complaints": complaints,
            "suggested_fix": fix,
        }
    )


def _coach_output(
    response_text: str = "Grounded response.",
    referenced_facts: list[str] | None = None,
) -> CoachOutput:
    return CoachOutput(
        response_text=response_text,
        referenced_facts=referenced_facts or ["age", "location"],
        referenced_hypotheses=[],
    )


def _critic_input(coach_out: CoachOutput | None = None) -> CriticInput:
    return CriticInput(
        coach_output=coach_out or _coach_output(),
        user_facts={"age": 19, "location": "London"},
        active_hypotheses=[],
        intent_packet=_intent(),
    )


async def test_critic_returns_pass_verdict() -> None:
    mock = MockLLMClient(default_response=_pass_verdict())
    critic = Critic(_make_factory(mock))
    verdict = await critic.run(_critic_input())

    assert isinstance(verdict, CriticVerdict)
    assert verdict.verdict == "pass"
    assert verdict.failure_modes == []


async def test_critic_returns_reject_verdict() -> None:
    mock = MockLLMClient(
        default_response=_reject_verdict(
            modes=["generic"],
            complaints=["Response could apply to any student."],
            fix="Reference the user's specific location and age.",
        )
    )
    critic = Critic(_make_factory(mock))
    verdict = await critic.run(_critic_input())

    assert verdict.verdict == "reject"
    assert "generic" in verdict.failure_modes
    assert verdict.suggested_fix is not None


async def test_critic_uses_haiku_model() -> None:
    mock = MockLLMClient(default_response=_pass_verdict())
    critic = Critic(_make_factory(mock))
    await critic.run(_critic_input())

    assert mock.calls[0].model == "Qwen/Qwen3-32B"


async def test_critic_fails_open_on_malformed_json() -> None:
    """A broken Critic response must not block the user — fail open with pass."""
    mock = MockLLMClient(default_response="definitely not json")
    critic = Critic(_make_factory(mock))
    verdict = await critic.run(_critic_input())

    assert verdict.verdict == "pass"


async def test_critic_prompt_contains_response_text() -> None:
    mock = MockLLMClient(default_response=_pass_verdict())
    critic = Critic(_make_factory(mock))
    coach_out = _coach_output(response_text="Unique phrase XYZ123 in the response.")
    await critic.run(_critic_input(coach_out=coach_out))

    prompt = mock.calls[0].messages[0].content
    assert "Unique phrase XYZ123" in prompt


async def test_critic_prompt_contains_specific_ask() -> None:
    mock = MockLLMClient(default_response=_pass_verdict())
    critic = Critic(_make_factory(mock))
    await critic.run(_critic_input())

    prompt = mock.calls[0].messages[0].content
    assert "Should I study economics?" in prompt


async def test_critic_retries_on_empty_response() -> None:
    """First LLM call returns empty string; retry returns valid JSON."""
    mock = MockLLMClient()
    mock.queue("", _pass_verdict())  # empty first, valid second
    critic = Critic(_make_factory(mock))

    verdict = await critic.run(_critic_input())

    assert len(mock.calls) == 2, "Expected exactly 2 LLM calls (original + retry)"
    assert isinstance(verdict, CriticVerdict)
    assert verdict.verdict == "pass"
