"""Unit tests for :class:`DevilsAdvocate` using :class:`MockLLMClient`.

All tests are purely in-memory — no DB, no real LLM. Persistence is skipped
because :meth:`DevilsAdvocate.run` logs via ``agent_calls`` which swallows
pool errors in test environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from career_coach.agents.devils_advocate import DevilsAdvocate
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import (
    CoachOutput,
    DevilsAdvocateInput,
    DevilsAdvocateOutput,
)
from career_coach.models.intent import IntentPacket

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _make_da(mock: MockLLMClient) -> DevilsAdvocate:
    return DevilsAdvocate(factory=_make_factory(mock))


def _intent(ask: str = "Should I go into quant finance?") -> IntentPacket:
    return IntentPacket(
        session_theory="User deciding between quant and SWE.",
        turn_intent="decide",
        specific_ask=ask,
        emotional_tenor="curious",
        clarity_score=0.8,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=["UK-based", "maths degree"],
        budget_hint="deep",
    )


def _coach_output(response: str = "Quant finance is a strong choice for you.") -> CoachOutput:
    return CoachOutput(
        response_text=response,
        referenced_facts=["degree", "location"],
        referenced_hypotheses=[],
        proposed_challenge=None,
        uncertainty_flags=[],
    )


def _da_disagrees_json() -> str:
    return json.dumps(
        {
            "counter_points": [
                {
                    "point": "Your degree is from a non-target university.",
                    "reasoning": (
                        "Most quant roles at top firms recruit almost exclusively from "
                        "Oxford, Cambridge, Imperial, and UCL. Your Warwick degree is "
                        "strong but places you at a disadvantage in the initial screening."
                    ),
                    "severity": "high",
                    "source_type": "user_profile",
                },
                {
                    "point": "You have no competition or Olympiad track record.",
                    "reasoning": (
                        "Quant firms treat competition history (IMO, ICPC, Putnam) as a "
                        "proxy for mathematical horsepower. Without it, you will need "
                        "exceptional internship performance to compensate."
                    ),
                    "severity": "med",
                    "source_type": "user_profile",
                },
            ],
            "blind_spots": [
                "The Coach did not address quant role types — dev quant vs research quant "
                "have very different profiles."
            ],
            "risks": [
                {
                    "scenario": (
                        "You pass first-round screening but fail the technical interview "
                        "due to lack of competition-level problem-solving practice."
                    ),
                    "likelihood": "med",
                    "impact": "high",
                }
            ],
            "agrees_with_coach": False,
        }
    )


def _da_agrees_json() -> str:
    return json.dumps(
        {
            "counter_points": [],
            "blind_spots": [],
            "risks": [],
            "agrees_with_coach": True,
        }
    )


def _make_input(user_facts: dict | None = None) -> DevilsAdvocateInput:
    return DevilsAdvocateInput(
        coach_output=_coach_output(),
        research_brief=None,
        user_facts=user_facts or {"age": 21, "degree": "Mathematics, University of Warwick"},
        active_hypotheses=[],
        intent_packet=_intent(),
    )


# ---------------------------------------------------------------------------
# 1. Honest agreement path
# ---------------------------------------------------------------------------


class TestDAHonestAgreement:
    """agrees_with_coach=True with empty counter_points is a valid, non-error output."""

    @pytest.mark.asyncio
    async def test_honest_agreement_returns_valid_output(self) -> None:
        mock = MockLLMClient()
        mock.queue(_da_agrees_json())
        da = _make_da(mock)

        output = await da.run(_make_input())

        assert output.agrees_with_coach is True
        assert output.counter_points == []
        # blind_spots and risks allowed to be empty
        assert isinstance(output, DevilsAdvocateOutput)

    @pytest.mark.asyncio
    async def test_honest_agreement_does_not_raise(self) -> None:
        """Honest agreement (empty counter_points + agrees_with_coach=True) must not error."""
        mock = MockLLMClient()
        mock.queue(_da_agrees_json())
        da = _make_da(mock)
        try:
            output = await da.run(_make_input())
            assert output.agrees_with_coach is True
        except Exception as exc:
            pytest.fail(f"run() raised unexpectedly on honest agreement: {exc}")

    @pytest.mark.asyncio
    async def test_disagreement_with_counter_points(self) -> None:
        """Normal disagreement path produces ≥2 counter_points."""
        mock = MockLLMClient()
        mock.queue(_da_disagrees_json())
        da = _make_da(mock)

        output = await da.run(_make_input())

        assert output.agrees_with_coach is False
        assert len(output.counter_points) >= 2


# ---------------------------------------------------------------------------
# 2. User profile grounding
# ---------------------------------------------------------------------------


class TestDAGroundsInUserProfile:
    """The rendered prompt includes user_facts so the model can ground counter-points."""

    @pytest.mark.asyncio
    async def test_prompt_includes_user_facts(self) -> None:
        """Verify the rendered prompt contains the user's degree."""
        mock = MockLLMClient()
        mock.queue(_da_agrees_json())
        da = _make_da(mock)

        user_facts = {"degree": "Mathematics, University of Warwick", "age": 21}
        await da.run(_make_input(user_facts=user_facts))

        # Inspect the recorded call — the prompt is the first message's content
        assert mock.calls, "No LLM call was recorded"
        first_message_content = mock.calls[0].messages[0].content
        assert "Warwick" in first_message_content, (
            "user_facts not rendered into the prompt"
        )

    @pytest.mark.asyncio
    async def test_user_profile_source_type_accepted(self) -> None:
        """source_type='user_profile' must not raise a validation error."""
        output_with_profile = json.dumps(
            {
                "counter_points": [
                    {
                        "point": "You have no internship experience.",
                        "reasoning": "Your profile shows no internship history.",
                        "severity": "high",
                        "source_type": "user_profile",
                    },
                    {
                        "point": "Quant hours may conflict with your stated preference.",
                        "reasoning": "You mentioned wanting work-life balance.",
                        "severity": "med",
                        "source_type": "user_profile",
                    },
                ],
                "blind_spots": [],
                "risks": [],
                "agrees_with_coach": False,
            }
        )
        mock = MockLLMClient()
        mock.queue(output_with_profile)
        da = _make_da(mock)

        output = await da.run(_make_input())
        assert all(
            cp.source_type == "user_profile" for cp in output.counter_points
        )


# ---------------------------------------------------------------------------
# 3. Severity is required and validated
# ---------------------------------------------------------------------------


class TestDASeverityRequired:
    """Every counter_point must have a valid severity — missing severity triggers retry."""

    @pytest.mark.asyncio
    async def test_missing_severity_triggers_retry_and_succeeds(self) -> None:
        """First response omits severity → parse fails → retry produces valid output."""
        bad_response = json.dumps(
            {
                "counter_points": [
                    {
                        "point": "University mismatch",
                        "reasoning": "Not a target university.",
                        # severity missing → Pydantic validation should fail
                        "source_type": "user_profile",
                    },
                    {
                        "point": "No competition background",
                        "reasoning": "Firms prefer competition track records.",
                        # severity missing
                        "source_type": "general",
                    },
                ],
                "blind_spots": [],
                "risks": [],
                "agrees_with_coach": False,
            }
        )
        good_response = _da_disagrees_json()

        mock = MockLLMClient()
        mock.queue(bad_response, good_response)
        da = _make_da(mock)

        output = await da.run(_make_input())

        # Should have retried and used the good response
        assert len(mock.calls) == 2, "Expected exactly one retry"
        assert output.agrees_with_coach is False
        for cp in output.counter_points:
            assert cp.severity in ("low", "med", "high")

    @pytest.mark.asyncio
    async def test_invalid_severity_value_triggers_retry(self) -> None:
        """A severity value outside the Literal set triggers a parse error and retry."""
        bad_severity = json.dumps(
            {
                "counter_points": [
                    {
                        "point": "Point one",
                        "reasoning": "Some reasoning here.",
                        "severity": "critical",  # not in Literal
                        "source_type": "general",
                    },
                    {
                        "point": "Point two",
                        "reasoning": "More reasoning.",
                        "severity": "extreme",  # not in Literal
                        "source_type": "general",
                    },
                ],
                "blind_spots": [],
                "risks": [],
                "agrees_with_coach": False,
            }
        )
        mock = MockLLMClient()
        mock.queue(bad_severity, _da_agrees_json())
        da = _make_da(mock)

        # Should not raise — fallback on retry
        output = await da.run(_make_input())
        assert output is not None

    @pytest.mark.asyncio
    async def test_agrees_false_requires_two_counter_points(self) -> None:
        """agrees_with_coach=False with only one counter_point must fail validation."""
        only_one_cp = json.dumps(
            {
                "counter_points": [
                    {
                        "point": "Only one point",
                        "reasoning": "This reasoning explains the point.",
                        "severity": "high",
                        "source_type": "general",
                    }
                ],
                "blind_spots": [],
                "risks": [],
                "agrees_with_coach": False,
            }
        )
        mock = MockLLMClient()
        mock.queue(only_one_cp, _da_agrees_json())
        da = _make_da(mock)

        output = await da.run(_make_input())
        # Should have retried; fallback is honest agreement
        assert len(mock.calls) == 2
        # Fallback is honest agreement
        assert output.agrees_with_coach is True


# ---------------------------------------------------------------------------
# 4. Research findings in prompt
# ---------------------------------------------------------------------------


class TestDAResearchIntegration:
    """Research findings are passed into the DA prompt when available."""

    @pytest.mark.asyncio
    async def test_research_findings_appear_in_prompt(self) -> None:
        """When a non-empty ResearchBrief is provided, its findings appear in the prompt."""
        from datetime import UTC, datetime

        from career_coach.models.agent_io import Citation, Finding, ResearchBrief

        finding = Finding(
            claim="Quant finance roles at top firms require competition-level maths skills.",
            confidence=0.85,
            citations=[
                Citation(
                    url="https://example.com/quant-hiring",
                    title="Quant hiring guide",
                    source_type="web",
                    accessed_at=datetime.now(UTC),
                )
            ],
            is_numeric=False,
        )
        brief = ResearchBrief(
            question="Should I go into quant finance?",
            findings=[finding],
            caveats=[],
            next_questions=[],
            used_kb_files=[],
            web_searches_run=["quant finance hiring UK 2025"],
            web_sources_consulted=["https://example.com/quant-hiring"],
        )

        mock = MockLLMClient()
        mock.queue(_da_agrees_json())
        da = _make_da(mock)

        inp = DevilsAdvocateInput(
            coach_output=_coach_output(),
            research_brief=brief,
            user_facts={"age": 21},
            active_hypotheses=[],
            intent_packet=_intent(),
        )
        await da.run(inp)

        first_message_content = mock.calls[0].messages[0].content
        assert "competition-level maths" in first_message_content, (
            "Research findings not rendered into DA prompt"
        )


# ---------------------------------------------------------------------------
# 5. Fallback on persistent parse failure
# ---------------------------------------------------------------------------


class TestDAFallback:
    """Persistent parse failures fall back to honest agreement, never raise."""

    @pytest.mark.asyncio
    async def test_persistent_failure_falls_back_silently(self) -> None:
        """If both attempts fail, DA returns the honest-agreement fallback."""
        mock = MockLLMClient(default_response="NOT JSON AT ALL {{{")
        da = _make_da(mock)

        output = await da.run(_make_input())

        assert output.agrees_with_coach is True
        assert output.counter_points == []

    @pytest.mark.asyncio
    async def test_fallback_does_not_raise(self) -> None:
        mock = MockLLMClient(default_response="{}")
        da = _make_da(mock)

        try:
            output = await da.run(_make_input())
            assert isinstance(output, DevilsAdvocateOutput)
        except Exception as exc:
            pytest.fail(f"run() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 6. Integration test (skipped without real API key)
# ---------------------------------------------------------------------------

_SKIP_LIVE = not os.getenv("HUGGINGFACE_API_TOKEN")


@pytest.mark.live
@pytest.mark.skipif(_SKIP_LIVE, reason="HUGGINGFACE_API_TOKEN not set")
@pytest.mark.asyncio
async def test_da_live_integration() -> None:
    """Live test: real LLM → valid DevilsAdvocateOutput with ≥2 counter_points."""
    factory = LLMFactory(config_path=_CONFIG)
    da = DevilsAdvocate(factory=factory)

    inp = DevilsAdvocateInput(
        coach_output=CoachOutput(
            response_text=(
                "Quant finance is an excellent choice for you. With your mathematics "
                "degree, strong programming skills, and interest in markets, you have "
                "the right profile. The salaries are very high — expect £80k–£120k "
                "in your first year at a top firm. Competition is fierce but your "
                "background gives you a real shot."
            ),
            referenced_facts=["degree", "skills", "interests"],
            referenced_hypotheses=[],
            proposed_challenge=None,
            uncertainty_flags=[],
        ),
        research_brief=None,
        user_facts={
            "age": 21,
            "degree": "Mathematics, University of Warwick (2:1)",
            "skills": ["Python", "numpy"],
            "internships": [],
            "competition_history": "none",
        },
        active_hypotheses=[],
        intent_packet=_intent(),
    )
    output = await da.run(inp)

    assert isinstance(output, DevilsAdvocateOutput)
    if not output.agrees_with_coach:
        assert len(output.counter_points) >= 2, (
            f"Expected ≥2 counter_points, got {len(output.counter_points)}"
        )
        for cp in output.counter_points:
            assert cp.severity in ("low", "med", "high")
            assert cp.source_type in ("user_profile", "research", "general")
