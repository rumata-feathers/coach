"""Unit tests for :class:`Researcher` using :class:`MockLLMClient` and
:class:`MockWebSearchClient`.

All tests are purely in-memory — no DB, no real web search, no real LLM.
Persistence is skipped because :meth:`Researcher._persist` swallows errors
from an unavailable DB pool (unit test environment has no pool).

Integration test at the bottom is skipped unless both TAVILY_API_KEY and
HUGGINGFACE_API_TOKEN are present in the environment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from career_coach.agents.researcher import Researcher
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import ResearcherInput
from career_coach.web.client import SearchResult
from tests.fixtures.mock_web_search import MockWebSearchClient

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
_KB_ROOT = Path(__file__).resolve().parents[2] / "kb"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _make_researcher(mock_llm: MockLLMClient, mock_web: MockWebSearchClient) -> Researcher:
    return Researcher(
        factory=_make_factory(mock_llm),
        web_client=mock_web,
        kb_root=_KB_ROOT,
    )


def _make_input(
    question: str = "Should I go into quant finance?",
    depth: str = "shallow",
) -> ResearcherInput:
    return ResearcherInput(
        question=question,
        user_facts={"age": 20, "degree": "Mathematics"},
        depth=depth,
    )


def _plan_json(queries: list[str] | None = None, tags: list[str] | None = None) -> str:
    return json.dumps(
        {
            "search_queries": queries or ["quant finance graduate salary UK 2025"],
            "kb_tags": tags or ["math_heavy", "high_comp"],
            "kb_career_names": ["quant_finance"],
            "rationale": "Need current pay data and structural KB info.",
        }
    )


def _brief_json(
    findings: list[dict] | None = None,
    caveats: list[str] | None = None,
) -> str:
    if findings is None:
        findings = [
            {
                "claim": "Quant finance graduates at top London firms earn £80k–£120k base in their first year.",
                "confidence": 0.75,
                "citations": [
                    {
                        "url": "https://www.efinancialcareers.com/news/quant-salary-2025",
                        "kb_path": None,
                        "title": "eFinancialCareers quant salary guide 2025",
                        "accessed_at": datetime.now(timezone.utc).isoformat(),
                        "source_type": "web",
                    }
                ],
                "is_numeric": True,
            },
            {
                "claim": "Typical entry paths into quant finance require a strong mathematics or physics undergraduate degree.",
                "confidence": 0.90,
                "citations": [
                    {
                        "url": None,
                        "kb_path": "kb/careers/quant_finance.yaml",
                        "title": "Quantitative Finance — World KB",
                        "accessed_at": datetime.now(timezone.utc).isoformat(),
                        "source_type": "kb",
                    }
                ],
                "is_numeric": False,
            },
        ]
    return json.dumps(
        {
            "question": "Should I go into quant finance?",
            "findings": findings,
            "caveats": caveats or [],
            "next_questions": ["What programming languages do quant firms expect?"],
            "used_kb_files": ["kb/careers/quant_finance.yaml"],
            "web_searches_run": ["quant finance graduate salary UK 2025"],
            "web_sources_consulted": [
                "https://www.efinancialcareers.com/news/quant-salary-2025"
            ],
        }
    )


# ---------------------------------------------------------------------------
# 1. Planner produces a valid plan
# ---------------------------------------------------------------------------


class TestResearcherPlannerOutputsValidPlan:
    """The planner LLM call produces a valid RetrievalPlan."""

    @pytest.mark.asyncio
    async def test_planner_outputs_valid_plan(self) -> None:
        mock_llm = MockLLMClient()
        # First response = plan, second = brief
        mock_llm.queue(_plan_json(), _brief_json())
        mock_web = MockWebSearchClient(
            default_results=[
                SearchResult(
                    url="https://example.com",
                    title="Example",
                    snippet="Quant salaries are high.",
                )
            ],
            default_fetch_content="Full article: Quant analyst salaries range from £80k to £150k.",
        )
        researcher = _make_researcher(mock_llm, mock_web)

        plan = await researcher._plan(_make_input(), max_queries=3)

        assert len(plan.search_queries) <= 3
        assert "quant finance graduate salary UK 2025" in plan.search_queries
        assert "math_heavy" in plan.kb_tags
        assert "quant_finance" in plan.kb_career_names
        assert plan.rationale  # non-empty

    @pytest.mark.asyncio
    async def test_planner_caps_queries_to_depth_limit(self) -> None:
        # Plan returns 10 queries but shallow caps at 3
        big_plan = json.dumps(
            {
                "search_queries": [f"query {i}" for i in range(10)],
                "kb_tags": [],
                "kb_career_names": [],
                "rationale": "many queries",
            }
        )
        mock_llm = MockLLMClient()
        mock_llm.queue(big_plan, _brief_json())
        researcher = _make_researcher(mock_llm, MockWebSearchClient())

        plan = await researcher._plan(_make_input(depth="shallow"), max_queries=3)
        assert len(plan.search_queries) == 3

    @pytest.mark.asyncio
    async def test_planner_failure_returns_empty_plan(self) -> None:
        """If the planner LLM returns invalid JSON, fall back to an empty plan."""
        mock_llm = MockLLMClient()
        mock_llm.queue("this is not json at all", _brief_json())
        researcher = _make_researcher(mock_llm, MockWebSearchClient())

        plan = await researcher._plan(_make_input(), max_queries=3)
        assert plan.search_queries == []
        assert plan.kb_tags == []


# ---------------------------------------------------------------------------
# 2. Empty web results → KB-only with caveat
# ---------------------------------------------------------------------------


class TestResearcherHandlesEmptyWebResults:
    """When web search returns no results, the brief degrades to KB-only."""

    @pytest.mark.asyncio
    async def test_empty_web_results_degrades_to_kb_only(self) -> None:
        kb_only_brief = json.dumps(
            {
                "question": "Should I go into quant finance?",
                "findings": [
                    {
                        "claim": "Quant finance typically requires a maths or physics degree.",
                        "confidence": 0.90,
                        "citations": [
                            {
                                "url": None,
                                "kb_path": "kb/careers/quant_finance.yaml",
                                "title": "Quantitative Finance — World KB",
                                "accessed_at": datetime.now(timezone.utc).isoformat(),
                                "source_type": "kb",
                            }
                        ],
                        "is_numeric": False,
                    }
                ],
                "caveats": ["no web results available — KB only"],
                "next_questions": [],
                "used_kb_files": ["kb/careers/quant_finance.yaml"],
                "web_searches_run": ["quant finance UK 2025"],
                "web_sources_consulted": [],
            }
        )
        mock_llm = MockLLMClient()
        mock_llm.queue(_plan_json(), kb_only_brief)
        mock_web = MockWebSearchClient(default_results=[])  # always empty

        researcher = _make_researcher(mock_llm, mock_web)
        brief = await researcher.run(_make_input())

        assert any("kb only" in c.lower() for c in brief.caveats)
        # All citations should be KB-sourced
        for finding in brief.findings:
            for citation in finding.citations:
                assert citation.source_type == "kb"

    @pytest.mark.asyncio
    async def test_no_web_searches_run_when_empty_results(self) -> None:
        """web_sources_consulted must be empty when web returned nothing."""
        kb_only_brief = json.dumps(
            {
                "question": "Should I go into quant finance?",
                "findings": [],
                "caveats": ["no web results available — KB only"],
                "next_questions": [],
                "used_kb_files": [],
                "web_searches_run": [],
                "web_sources_consulted": [],
            }
        )
        mock_llm = MockLLMClient()
        mock_llm.queue(_plan_json(queries=[]), kb_only_brief)  # empty plan
        mock_web = MockWebSearchClient(default_results=[])

        researcher = _make_researcher(mock_llm, mock_web)
        brief = await researcher.run(_make_input())

        assert brief.web_sources_consulted == []


# ---------------------------------------------------------------------------
# 3. Empty KB match → web-only, no error
# ---------------------------------------------------------------------------


class TestResearcherHandlesEmptyKBMatch:
    """When no KB entries match the plan, the brief uses web sources only."""

    @pytest.mark.asyncio
    async def test_empty_kb_match_uses_web_alone(self) -> None:
        # Plan with tags that match nothing
        plan_no_kb = json.dumps(
            {
                "search_queries": ["quant finance salary 2025"],
                "kb_tags": [],
                "kb_career_names": [],
                "rationale": "web only",
            }
        )
        web_brief = json.dumps(
            {
                "question": "Should I go into quant finance?",
                "findings": [
                    {
                        "claim": "Quant salaries exceed £100k at senior levels.",
                        "confidence": 0.70,
                        "citations": [
                            {
                                "url": "https://www.example.com/quant",
                                "kb_path": None,
                                "title": "Quant salary article",
                                "accessed_at": datetime.now(timezone.utc).isoformat(),
                                "source_type": "web",
                            }
                        ],
                        "is_numeric": True,
                    }
                ],
                "caveats": ["no KB entries matched — web only"],
                "next_questions": [],
                "used_kb_files": [],
                "web_searches_run": ["quant finance salary 2025"],
                "web_sources_consulted": ["https://www.example.com/quant"],
            }
        )
        mock_llm = MockLLMClient()
        mock_llm.queue(plan_no_kb, web_brief)
        mock_web = MockWebSearchClient(
            default_results=[
                SearchResult(
                    url="https://www.example.com/quant",
                    title="Quant salary article",
                    snippet="Quant salaries are high.",
                )
            ],
            default_fetch_content="Quant salaries exceed £100k at senior levels.",
        )

        researcher = _make_researcher(mock_llm, mock_web)
        brief = await researcher.run(_make_input())

        assert len(brief.findings) >= 1
        assert brief.used_kb_files == []
        # No error raised
        assert brief.question == "Should I go into quant finance?"


# ---------------------------------------------------------------------------
# 4. Both sources empty → empty findings + caveats, no error
# ---------------------------------------------------------------------------


class TestResearcherHandlesBothEmpty:
    """When both web and KB return nothing, the brief is empty but valid."""

    @pytest.mark.asyncio
    async def test_both_sources_empty_returns_valid_brief(self) -> None:
        plan_empty = json.dumps(
            {
                "search_queries": [],
                "kb_tags": [],
                "kb_career_names": [],
                "rationale": "nothing to search",
            }
        )
        empty_brief = json.dumps(
            {
                "question": "Should I go into quant finance?",
                "findings": [],
                "caveats": [
                    "no web results available — KB only",
                    "no KB entries matched — web only",
                ],
                "next_questions": [],
                "used_kb_files": [],
                "web_searches_run": [],
                "web_sources_consulted": [],
            }
        )
        mock_llm = MockLLMClient()
        mock_llm.queue(plan_empty, empty_brief)
        mock_web = MockWebSearchClient(default_results=[])

        researcher = _make_researcher(mock_llm, mock_web)
        brief = await researcher.run(_make_input())

        assert brief.findings == []
        assert len(brief.caveats) >= 1
        # No exception
        assert brief.question == "Should I go into quant finance?"

    @pytest.mark.asyncio
    async def test_both_empty_does_not_raise(self) -> None:
        """run() must never raise even with empty sources and bad LLM response."""
        mock_llm = MockLLMClient(default_response="{}")
        mock_web = MockWebSearchClient(default_results=[])

        researcher = _make_researcher(mock_llm, mock_web)
        # Should not raise even with a degenerate LLM response
        brief = await researcher.run(_make_input())
        assert isinstance(brief, type(brief))


# ---------------------------------------------------------------------------
# 5. Truncation on timeout
# ---------------------------------------------------------------------------


class TestResearcherTruncatesOnTimeout:
    """Budget exhaustion returns a truncated brief, not an exception."""

    @pytest.mark.asyncio
    async def test_zero_budget_returns_truncated_brief(self) -> None:
        """With _budget_override_s=0.0 the budget is gone after planning."""
        mock_llm = MockLLMClient()
        mock_llm.queue(_plan_json())  # Only planning response needed
        mock_web = MockWebSearchClient()

        researcher = _make_researcher(mock_llm, mock_web)
        brief = await researcher.run(_make_input(), _budget_override_s=0.0)

        assert any("truncated" in c.lower() for c in brief.caveats)
        assert brief.findings == []

    @pytest.mark.asyncio
    async def test_truncated_brief_includes_planned_queries(self) -> None:
        """Truncated brief should record the queries that were planned."""
        mock_llm = MockLLMClient()
        mock_llm.queue(_plan_json(queries=["quant finance UK salary"]))
        mock_web = MockWebSearchClient()

        researcher = _make_researcher(mock_llm, mock_web)
        brief = await researcher.run(_make_input(), _budget_override_s=0.0)

        # The planned queries appear in web_searches_run even if we didn't run them
        assert "quant finance UK salary" in brief.web_searches_run

    @pytest.mark.asyncio
    async def test_truncation_does_not_raise(self) -> None:
        """A timeout must not propagate as an exception."""
        mock_llm = MockLLMClient(delay_ms=200)
        mock_llm.queue(_plan_json())  # Planning response (fast enough at 200ms)
        mock_web = MockWebSearchClient()

        researcher = _make_researcher(mock_llm, mock_web)
        # Give a very small budget so execution times out
        try:
            brief = await researcher.run(_make_input(), _budget_override_s=0.0)
            assert brief is not None  # No exception
        except Exception as exc:
            pytest.fail(f"run() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 6. Brief model validation
# ---------------------------------------------------------------------------


class TestResearchBriefValidation:
    """Citation and Finding validators work correctly."""

    def test_citation_must_have_url_or_kb_path(self) -> None:
        from career_coach.models.agent_io import Citation
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Citation(url=None, kb_path=None, title="no source", source_type="web")

    def test_citation_cannot_have_both(self) -> None:
        from career_coach.models.agent_io import Citation
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Citation(
                url="https://x.com",
                kb_path="kb/careers/law_uk.yaml",
                title="ambiguous",
                source_type="web",
            )

    def test_finding_requires_at_least_one_citation(self) -> None:
        from career_coach.models.agent_io import Finding
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Finding(claim="This claim has no support.", confidence=0.5, citations=[])

    def test_finding_confidence_range(self) -> None:
        from career_coach.models.agent_io import Citation, Finding
        from pydantic import ValidationError

        good_citation = Citation(
            url="https://example.com",
            title="Example",
            source_type="web",
        )
        with pytest.raises(ValidationError):
            Finding(claim="Claim.", confidence=1.5, citations=[good_citation])


# ---------------------------------------------------------------------------
# 7. Live integration test (skipped without real API keys)
# ---------------------------------------------------------------------------

_SKIP_LIVE = not (
    os.getenv("TAVILY_API_KEY") and os.getenv("HUGGINGFACE_API_TOKEN")
)


@pytest.mark.skipif(_SKIP_LIVE, reason="TAVILY_API_KEY and HUGGINGFACE_API_TOKEN not set")
@pytest.mark.asyncio
async def test_researcher_live_integration() -> None:
    """Live test: real Tavily + real LLM → valid ResearchBrief."""
    from career_coach.web.factory import get_client

    web_client = get_client("researcher")
    factory = LLMFactory(config_path=_CONFIG)
    researcher = Researcher(factory=factory, web_client=web_client, kb_root=_KB_ROOT)

    inp = ResearcherInput(
        question="Should I go into quant finance?",
        user_facts={"age": 20, "major": "Mathematics"},
        depth="shallow",
    )
    brief = await researcher.run(inp)

    assert brief.question == inp.question
    assert len(brief.findings) >= 2, f"Expected ≥2 findings, got {len(brief.findings)}"
    assert len(brief.caveats) >= 1 or len(brief.findings) >= 2
    for finding in brief.findings:
        assert finding.citations, f"Finding has no citations: {finding.claim}"
    assert brief.web_sources_consulted or brief.used_kb_files, (
        "Brief has no consulted sources"
    )
