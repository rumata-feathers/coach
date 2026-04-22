"""Unit tests for :class:`Coach` with :class:`MockLLMClient`."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from career_coach.agents.coach import Coach
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import CoachInput, CoachOutput
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import Hypothesis

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _intent(specific_ask: str = "Should I study economics?") -> IntentPacket:
    return IntentPacket(
        session_theory="User weighing degree options.",
        turn_intent="decide",
        specific_ask=specific_ask,
        emotional_tenor="uncertain",
        clarity_score=0.75,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=["UK undergrad"],
        budget_hint="standard",
    )


def _good_output_json(hyp_id: str | None = None) -> str:
    hyp_refs = [str(hyp_id)] if hyp_id else []
    return json.dumps(
        {
            "response_text": (
                "Based on what I know about you — you're 19, studying in London at "
                "undergrad level — economics could be a strong fit if you enjoy "
                "abstract thinking over applied work. The London job market for "
                "economics graduates is broad, though salaries vary widely "
                "(I'd put that in the uncertain column)."
            ),
            "referenced_facts": ["age", "location"],
            "referenced_hypotheses": hyp_refs,
            "proposed_challenge": None,
            "uncertainty_flags": ["salary_ranges_vary_widely"],
        }
    )


def _make_input(critic_feedback: str | None = None) -> CoachInput:
    return CoachInput(
        intent_packet=_intent(),
        user_facts={"age": 19, "location": "London", "education_stage": "undergrad"},
        active_hypotheses=[],
        recent_turns=[],
        critic_feedback=critic_feedback,
    )


async def test_coach_returns_valid_output() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    coach = Coach(_make_factory(mock))
    output = await coach.run(_make_input())

    assert isinstance(output, CoachOutput)
    assert output.response_text
    assert "age" in output.referenced_facts


async def test_coach_uses_sonnet_model() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    coach = Coach(_make_factory(mock))
    await coach.run(_make_input())

    assert mock.calls[0].model == "Qwen/Qwen3-235B-A22B"


async def test_coach_requests_json_format() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    coach = Coach(_make_factory(mock))
    await coach.run(_make_input())

    assert mock.calls[0].response_format == "json"


async def test_coach_includes_critic_feedback_in_prompt() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    coach = Coach(_make_factory(mock))
    await coach.run(_make_input(critic_feedback="Response was too generic."))

    prompt = mock.calls[0].messages[0].content
    assert "Response was too generic." in prompt


async def test_coach_falls_back_on_malformed_json() -> None:
    mock = MockLLMClient(default_response="not json at all")
    coach = Coach(_make_factory(mock))
    output = await coach.run(_make_input())

    assert isinstance(output, CoachOutput)
    assert output.response_text  # fallback message set


async def test_coach_escalation_skips_llm() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    coach = Coach(_make_factory(mock))
    output = await coach.run_escalation(_make_input())

    # Escalation path must NOT call the LLM.
    assert len(mock.calls) == 0
    assert isinstance(output, CoachOutput)
    assert "more context" in output.response_text.lower()


async def test_coach_prompt_contains_user_facts() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    coach = Coach(_make_factory(mock))
    await coach.run(_make_input())

    prompt = mock.calls[0].messages[0].content
    assert "London" in prompt
    assert "undergrad" in prompt


async def test_coach_prompt_contains_hypotheses() -> None:
    hyp = Hypothesis(
        hypothesis_id=uuid4(),
        statement="User prefers analytical work.",
        confidence=0.6,
        status="active",
    )
    mock = MockLLMClient(default_response=_good_output_json(hyp_id=str(hyp.hypothesis_id)))
    coach = Coach(_make_factory(mock))
    input_data = CoachInput(
        intent_packet=_intent(),
        user_facts={"age": 19},
        active_hypotheses=[hyp],
        recent_turns=[],
    )
    await coach.run(input_data)

    prompt = mock.calls[0].messages[0].content
    assert "User prefers analytical work." in prompt
