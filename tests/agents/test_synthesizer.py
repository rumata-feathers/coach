"""Unit tests for :class:`Synthesizer` using :class:`MockLLMClient`.

All tests are purely in-memory — no DB, no real LLM.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from career_coach.agents.synthesizer import Synthesizer
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import (
    Citation,
    CoachOutput,
    CounterPoint,
    DevilsAdvocateOutput,
    Finding,
    ResearchBrief,
    SynthesizedResponse,
    SynthesizerInput,
)
from career_coach.models.intent import IntentPacket

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _make_synth(mock: MockLLMClient) -> Synthesizer:
    return Synthesizer(factory=_make_factory(mock))


def _intent(ask: str = "Should I go into quant finance or software engineering?") -> IntentPacket:
    return IntentPacket(
        session_theory="User deciding between career paths.",
        turn_intent="decide",
        specific_ask=ask,
        emotional_tenor="curious",
        clarity_score=0.85,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=["UK-based", "maths degree"],
        budget_hint="deep",
    )


def _coach_out() -> CoachOutput:
    return CoachOutput(
        response_text=(
            "Given your mathematics degree and interest in markets, quant finance is "
            "a strong option. SWE gives more optionality. Both paths are viable."
        ),
        referenced_facts=["degree", "interests"],
        referenced_hypotheses=[],
        proposed_challenge=None,
        uncertainty_flags=[],
    )


def _da_disagrees() -> DevilsAdvocateOutput:
    return DevilsAdvocateOutput(
        counter_points=[
            CounterPoint(
                point="Non-target university limits quant access.",
                reasoning="Top quant firms recruit almost exclusively from Oxbridge/Imperial.",
                severity="high",
                source_type="user_profile",
            ),
            CounterPoint(
                point="No competition track record is a disadvantage.",
                reasoning="Quant firms use competition history as a filter.",
                severity="med",
                source_type="user_profile",
            ),
        ],
        blind_spots=["Role types within quant vary enormously."],
        risks=[],
        agrees_with_coach=False,
    )


def _da_agrees() -> DevilsAdvocateOutput:
    return DevilsAdvocateOutput(
        counter_points=[],
        blind_spots=[],
        risks=[],
        agrees_with_coach=True,
    )


def _web_citation() -> dict:
    return {
        "url": "https://www.efinancialcareers.com/quant-salary",
        "kb_path": None,
        "title": "eFinancialCareers quant salary guide",
        "accessed_at": datetime.now(UTC).isoformat(),
        "source_type": "web",
    }


def _kb_citation() -> dict:
    return {
        "url": None,
        "kb_path": "kb/careers/quant_finance.yaml",
        "title": "Quantitative Finance — World KB",
        "accessed_at": datetime.now(UTC).isoformat(),
        "source_type": "kb",
    }


def _brief_with_findings() -> ResearchBrief:
    return ResearchBrief(
        question="Should I go into quant finance or software engineering?",
        findings=[
            Finding(
                claim="Quant analysts at top London firms earn £80k-£120k in their first year.",
                confidence=0.75,
                citations=[
                    Citation(
                        url="https://www.efinancialcareers.com/quant-salary",
                        title="eFinancialCareers quant salary guide",
                        source_type="web",
                        accessed_at=datetime.now(UTC),
                    )
                ],
                is_numeric=True,
            ),
            Finding(
                claim="Software engineers at FAANG-equivalent London offices earn £70k-£110k.",
                confidence=0.70,
                citations=[
                    Citation(
                        kb_path="kb/careers/software_engineering.yaml",
                        title="Software Engineering — World KB",
                        source_type="kb",
                        accessed_at=datetime.now(UTC),
                    )
                ],
                is_numeric=True,
            ),
        ],
        caveats=[],
        next_questions=[],
        used_kb_files=["kb/careers/software_engineering.yaml"],
        web_searches_run=["quant finance salary London 2025"],
        web_sources_consulted=["https://www.efinancialcareers.com/quant-salary"],
    )


def _good_synth_json(
    integrated_from=None, tradeoffs=None, chart_specs=None, findings=None
) -> str:
    return json.dumps(
        {
            "response_text": (
                "Given your mathematics degree from Warwick and your stated interest "
                "in markets [1], quant finance offers higher starting compensation "
                "than SWE but demands a more specialised skill set [2]. The Devil's "
                "Advocate raises a valid concern: your university may put you at a "
                "disadvantage in quant's highly competitive screening process."
            ),
            "referenced_facts": ["degree", "interests"],
            "referenced_hypotheses": [],
            "referenced_findings": (
                findings
                if findings is not None
                else ["Quant analysts at top London firms earn £80k-£120k in their first year."]
            ),
            "citations": [_web_citation(), _kb_citation()],
            "surfaced_tradeoffs": (
                tradeoffs
                if tradeoffs is not None
                else ["Quant offers higher ceiling but narrower access vs SWE's broader optionality."]
            ),
            "integrated_from": integrated_from or ["coach", "devils_advocate", "researcher"],
            "chart_specs": chart_specs if chart_specs is not None else [],
            "proposed_challenge": None,
            "uncertainty_flags": [],
        }
    )


def _make_input(
    da: DevilsAdvocateOutput | None = None,
    brief: ResearchBrief | None = None,
    critic_feedback: str | None = None,
) -> SynthesizerInput:
    return SynthesizerInput(
        coach_output=_coach_out(),
        da_output=da or _da_disagrees(),
        research_brief=brief or _brief_with_findings(),
        user_facts={"degree": "Mathematics, Warwick", "interests": ["markets", "coding"]},
        active_hypotheses=[],
        intent_packet=_intent(),
        critic_feedback=critic_feedback,
    )


# ---------------------------------------------------------------------------
# 1. Integrates all sources
# ---------------------------------------------------------------------------


class TestSynthesizerIntegratesAll:
    """When all three inputs are non-empty, integrated_from contains all three."""

    @pytest.mark.asyncio
    async def test_integrates_all_sources(self) -> None:
        mock = MockLLMClient()
        mock.queue(_good_synth_json())
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        assert "coach" in output.integrated_from
        assert "devils_advocate" in output.integrated_from
        assert "researcher" in output.integrated_from

    @pytest.mark.asyncio
    async def test_referenced_findings_present(self) -> None:
        mock = MockLLMClient()
        mock.queue(_good_synth_json())
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        assert output.referenced_findings, "expected ≥1 referenced finding"

    @pytest.mark.asyncio
    async def test_citations_present_when_research_available(self) -> None:
        mock = MockLLMClient()
        mock.queue(_good_synth_json())
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        assert output.citations, "expected ≥1 citation"

    @pytest.mark.asyncio
    async def test_tradeoffs_present_when_da_disagreed(self) -> None:
        mock = MockLLMClient()
        mock.queue(_good_synth_json())
        synth = _make_synth(mock)

        output = await synth.run(_make_input(da=_da_disagrees()))

        assert output.surfaced_tradeoffs, "DA disagreed — expected ≥1 tradeoff"


# ---------------------------------------------------------------------------
# 2. Empty research brief
# ---------------------------------------------------------------------------


class TestSynthesizerEmptyResearch:
    """With an empty research brief, integrated_from excludes 'researcher'."""

    @pytest.mark.asyncio
    async def test_empty_research_excludes_researcher(self) -> None:
        no_research_json = _good_synth_json(
            integrated_from=["coach", "devils_advocate"],
            findings=[],
        )
        mock = MockLLMClient()
        mock.queue(no_research_json)
        synth = _make_synth(mock)

        empty_brief = ResearchBrief(
            question="Should I go into quant finance?",
            findings=[],
            caveats=["no web results available — KB only"],
            next_questions=[],
            used_kb_files=[],
            web_searches_run=[],
            web_sources_consulted=[],
        )
        output = await synth.run(_make_input(brief=empty_brief))

        assert "researcher" not in output.integrated_from
        assert "coach" in output.integrated_from
        assert "devils_advocate" in output.integrated_from

    @pytest.mark.asyncio
    async def test_none_research_excludes_researcher(self) -> None:
        no_research_json = _good_synth_json(integrated_from=["coach", "devils_advocate"])
        mock = MockLLMClient()
        mock.queue(no_research_json)
        synth = _make_synth(mock)

        output = await synth.run(_make_input(brief=None))

        assert "researcher" not in output.integrated_from


# ---------------------------------------------------------------------------
# 3. DA agrees → empty tradeoffs allowed
# ---------------------------------------------------------------------------


class TestSynthesizerDAAgrees:
    """When DA agrees, surfaced_tradeoffs may be empty."""

    @pytest.mark.asyncio
    async def test_da_agrees_allows_empty_tradeoffs(self) -> None:
        agreed_json = _good_synth_json(tradeoffs=[])
        mock = MockLLMClient()
        mock.queue(agreed_json)
        synth = _make_synth(mock)

        output = await synth.run(_make_input(da=_da_agrees()))

        # Should parse cleanly — DA agreement with empty tradeoffs is valid
        assert isinstance(output, SynthesizedResponse)
        assert output.surfaced_tradeoffs == []

    @pytest.mark.asyncio
    async def test_da_agrees_does_not_raise(self) -> None:
        agreed_json = _good_synth_json(tradeoffs=[], integrated_from=["coach", "researcher"])
        mock = MockLLMClient()
        mock.queue(agreed_json)
        synth = _make_synth(mock)

        try:
            output = await synth.run(_make_input(da=_da_agrees()))
            assert output is not None
        except Exception as exc:
            pytest.fail(f"run() raised unexpectedly with DA agreement: {exc}")


# ---------------------------------------------------------------------------
# 4. Chart emission gate — uncited chart triggers retry
# ---------------------------------------------------------------------------


class TestSynthesizerChartEmissionGated:
    """A chart with empty source_citation_indices fails Pydantic validation → retry."""

    @pytest.mark.asyncio
    async def test_uncited_chart_triggers_retry(self) -> None:
        """First response includes a chart with no source_citation_indices → parse error.
        Second response has no chart → passes."""
        uncited_chart_json = json.dumps(
            {
                "response_text": "Quant finance pays more but is harder to break into [1].",
                "referenced_facts": ["degree"],
                "referenced_hypotheses": [],
                "referenced_findings": [],
                "citations": [_web_citation()],
                "surfaced_tradeoffs": ["Higher comp vs harder access."],
                "integrated_from": ["coach", "devils_advocate"],
                "chart_specs": [
                    {
                        "chart_type": "bar",
                        "title": "Salary comparison",
                        "description": "Early career salaries quant vs SWE.",
                        "x_axis": {"label": "Career", "unit": None, "scale": "linear"},
                        "y_axis": {"label": "Salary", "unit": "GBP", "scale": "linear"},
                        "data": [
                            {"Career": "Quant", "Salary": 95000},
                            {"Career": "SWE", "Salary": 65000},
                            {"Career": "Consulting", "Salary": 58000},
                        ],
                        "source_citation_indices": [],  # INVALID — triggers parse failure
                    }
                ],
                "proposed_challenge": None,
                "uncertainty_flags": [],
            }
        )
        no_chart_json = _good_synth_json(chart_specs=[])

        mock = MockLLMClient()
        mock.queue(uncited_chart_json, no_chart_json)
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        assert len(mock.calls) == 2, "Expected exactly one retry triggered by uncited chart"
        assert output.chart_specs == [], "No chart should appear after fallback"

    @pytest.mark.asyncio
    async def test_two_charts_triggers_retry(self) -> None:
        """Two charts violates v1 max-1 constraint → parse failure → retry."""
        cited_chart = {
            "chart_type": "bar",
            "title": "Salary",
            "description": "Salary comparison.",
            "x_axis": {"label": "Career", "unit": None, "scale": "linear"},
            "y_axis": {"label": "Salary", "unit": "GBP", "scale": "linear"},
            "data": [{"Career": "Quant", "Salary": 95000}],
            "source_citation_indices": [0],
        }
        two_charts_json = json.dumps(
            {
                "response_text": "Response with two charts.",
                "referenced_facts": [],
                "referenced_hypotheses": [],
                "referenced_findings": [],
                "citations": [_web_citation()],
                "surfaced_tradeoffs": ["A tradeoff."],
                "integrated_from": ["coach"],
                "chart_specs": [cited_chart, cited_chart],  # two charts — INVALID
                "proposed_challenge": None,
                "uncertainty_flags": [],
            }
        )
        mock = MockLLMClient()
        mock.queue(two_charts_json, _good_synth_json())
        synth = _make_synth(mock)

        await synth.run(_make_input())
        assert len(mock.calls) == 2


# ---------------------------------------------------------------------------
# 5. Valid chart with source_citation_indices
# ---------------------------------------------------------------------------


class TestSynthesizerChartWithValidData:
    """A chart with populated source_citation_indices parses cleanly."""

    @pytest.mark.asyncio
    async def test_valid_chart_parses_cleanly(self) -> None:
        chart_json = json.dumps(
            {
                "response_text": (
                    "Based on your maths background and the research data [1][2], "
                    "the salary comparison below shows quant's advantage clearly."
                ),
                "referenced_facts": ["degree", "interests"],
                "referenced_hypotheses": [],
                "referenced_findings": [
                    "Quant analysts at top London firms earn £80k-£120k in their first year."
                ],
                "citations": [_web_citation(), _kb_citation()],
                "surfaced_tradeoffs": ["Higher quant ceiling vs broader SWE optionality."],
                "integrated_from": ["coach", "devils_advocate", "researcher"],
                "chart_specs": [
                    {
                        "chart_type": "bar",
                        "title": "Early-career salary comparison (London, 2025)",
                        "description": "Median gross annual salary at 0-3 years for three careers.",
                        "x_axis": {"label": "Career", "unit": None, "scale": "linear"},
                        "y_axis": {"label": "Median salary", "unit": "GBP", "scale": "linear"},
                        "data": [
                            {"Career": "Quant finance", "Median salary": 90000},
                            {"Career": "Software engineering", "Median salary": 55000},
                            {"Career": "Management consulting", "Median salary": 58000},
                        ],
                        "source_citation_indices": [0, 1],
                    }
                ],
                "proposed_challenge": None,
                "uncertainty_flags": [],
            }
        )
        mock = MockLLMClient()
        mock.queue(chart_json)
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        assert len(output.chart_specs) == 1
        chart = output.chart_specs[0]
        assert chart.chart_type == "bar"
        assert chart.source_citation_indices == [0, 1]
        assert len(chart.data) == 3

    @pytest.mark.asyncio
    async def test_chart_source_indices_reference_valid_citations(self) -> None:
        """source_citation_indices values should be valid indices into citations list."""
        chart_json = json.dumps(
            {
                "response_text": "Salary data [1] shows quant leads.",
                "referenced_facts": ["degree"],
                "referenced_hypotheses": [],
                "referenced_findings": [],
                "citations": [_web_citation()],
                "surfaced_tradeoffs": ["A tradeoff here."],
                "integrated_from": ["coach", "devils_advocate"],
                "chart_specs": [
                    {
                        "chart_type": "bar",
                        "title": "Salaries",
                        "description": "Career salary comparison.",
                        "x_axis": {"label": "Career", "unit": None},
                        "y_axis": {"label": "Salary", "unit": "GBP"},
                        "data": [
                            {"Career": "Quant", "Salary": 90000},
                            {"Career": "SWE", "Salary": 55000},
                            {"Career": "Law", "Salary": 55000},
                        ],
                        "source_citation_indices": [0],  # cites citations[0]
                    }
                ],
                "proposed_challenge": None,
                "uncertainty_flags": [],
            }
        )
        mock = MockLLMClient()
        mock.queue(chart_json)
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        assert output.chart_specs[0].source_citation_indices == [0]
        # All source_citation_indices should be valid indices
        for idx in output.chart_specs[0].source_citation_indices:
            assert 0 <= idx < len(output.citations)


# ---------------------------------------------------------------------------
# 6. Fallback on persistent parse failure
# ---------------------------------------------------------------------------


class TestSynthesizerFallback:
    """Persistent parse failures fall back to Coach text, never raise."""

    @pytest.mark.asyncio
    async def test_persistent_failure_returns_coach_fallback(self) -> None:
        mock = MockLLMClient(default_response="NOT JSON {{{")
        synth = _make_synth(mock)

        output = await synth.run(_make_input())

        # Fallback returns Coach text
        assert output.response_text == _coach_out().response_text
        assert "coach" in output.integrated_from
        assert "synthesis_failed_fallback_to_coach" in output.uncertainty_flags

    @pytest.mark.asyncio
    async def test_missing_coach_in_integrated_from_triggers_retry(self) -> None:
        """integrated_from without 'coach' fails the Pydantic validator → retry."""
        bad_json = json.dumps(
            {
                "response_text": "Some response.",
                "referenced_facts": [],
                "referenced_hypotheses": [],
                "referenced_findings": [],
                "citations": [],
                "surfaced_tradeoffs": [],
                "integrated_from": ["researcher"],  # coach missing → invalid
                "chart_specs": [],
                "proposed_challenge": None,
                "uncertainty_flags": [],
            }
        )
        mock = MockLLMClient()
        mock.queue(bad_json, _good_synth_json())
        synth = _make_synth(mock)

        output = await synth.run(_make_input())
        assert len(mock.calls) == 2
        assert "coach" in output.integrated_from


# ---------------------------------------------------------------------------
# 7. Model validators
# ---------------------------------------------------------------------------


class TestSynthesizerModelValidators:
    """AxisSpec, ChartSpec, and SynthesizedResponse validators work correctly."""

    def test_chart_spec_requires_source_citation_indices(self) -> None:
        from pydantic import ValidationError

        from career_coach.models.agent_io import ChartSpec

        with pytest.raises(ValidationError, match="source_citation_indices"):
            ChartSpec(
                chart_type="bar",
                title="Test",
                description="Test chart.",
                data=[{"x": 1}],
                source_citation_indices=[],  # empty → invalid
            )

    def test_synthesized_response_requires_coach_in_integrated_from(self) -> None:
        from pydantic import ValidationError

        from career_coach.models.agent_io import SynthesizedResponse

        with pytest.raises(ValidationError, match="integrated_from"):
            SynthesizedResponse(
                response_text="Hello.",
                integrated_from=["researcher"],  # no 'coach'
            )

    def test_max_one_chart_in_coach_output(self) -> None:
        from pydantic import ValidationError

        from career_coach.models.agent_io import ChartSpec, CoachOutput

        chart = ChartSpec(
            chart_type="bar",
            title="T",
            description="D.",
            data=[{"x": 1}, {"x": 2}, {"x": 3}],
            source_citation_indices=[0],
        )
        with pytest.raises(ValidationError, match="at most 1 chart"):
            CoachOutput(
                response_text="Hello.",
                chart_specs=[chart, chart],
            )

    def test_valid_chart_spec_parses(self) -> None:
        from career_coach.models.agent_io import AxisSpec, ChartSpec

        chart = ChartSpec(
            chart_type="line",
            title="Career progression",
            description="Salary over years of experience.",
            x_axis=AxisSpec(label="Years", unit="years"),
            y_axis=AxisSpec(label="Salary", unit="GBP"),
            data=[
                {"Years": 1, "Salary": 60000},
                {"Years": 3, "Salary": 80000},
                {"Years": 5, "Salary": 100000},
            ],
            source_citation_indices=[0, 1],
        )
        assert chart.source_citation_indices == [0, 1]
        assert chart.x_axis is not None


# ---------------------------------------------------------------------------
# 8. Integration test (skipped without real API key)
# ---------------------------------------------------------------------------

_SKIP_LIVE = not os.getenv("HUGGINGFACE_API_TOKEN")


@pytest.mark.skipif(_SKIP_LIVE, reason="HUGGINGFACE_API_TOKEN not set")
@pytest.mark.asyncio
async def test_synthesizer_live_integration() -> None:
    """Live test: real LLM → valid SynthesizedResponse with inline citations."""
    factory = LLMFactory(config_path=_CONFIG)
    synth = Synthesizer(factory=factory)

    output = await synth.run(
        SynthesizerInput(
            coach_output=_coach_out(),
            da_output=_da_disagrees(),
            research_brief=_brief_with_findings(),
            user_facts={"degree": "Mathematics, Warwick", "interests": ["markets"]},
            active_hypotheses=[],
            intent_packet=_intent(),
        )
    )

    assert output.response_text, "response_text should be non-empty"
    assert "coach" in output.integrated_from
    assert output.surfaced_tradeoffs, "DA disagreed — expect ≥1 tradeoff"
    # Should contain an inline citation
    assert "[" in output.response_text, "expected inline citation [N] in response_text"
