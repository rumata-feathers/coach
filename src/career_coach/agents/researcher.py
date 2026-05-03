"""Researcher agent.

Produces a :class:`ResearchBrief` by combining web search (via Tavily) with
static KB lookups. Uses a two-phase approach:

1. **Plan** — a cheap Haiku-tier LLM call decides which queries to run and
   which KB tags/careers to load.
2. **Execute and synthesise** — searches and fetches run in parallel; the
   results feed a Coach-tier LLM call that produces the final brief with
   citation hygiene enforced in the prompt.

Latency budgets: 8 s shallow, 15 s deep. On timeout, returns a minimal brief
with a truncation caveat rather than raising.

See SPEC_v1.md §5.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import ValidationError

from career_coach.agents.base import Agent
from career_coach.db import get_pool
from career_coach.kb.models import CareerEntry
from career_coach.kb.repo import WorldKBRepo
from career_coach.llm.client import Message
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import (
    ResearchBrief,
    ResearcherInput,
    RetrievalPlan,
)
from career_coach.web.client import SearchResult, WebSearchClient

logger = logging.getLogger("career_coach.agents.researcher")

_KB_ROOT = Path(__file__).resolve().parents[3] / "kb"
_RETRY_PROMPT = (
    "Your previous response was not valid JSON matching the ResearchBrief schema. "
    "Return ONLY a JSON object — no prose, no code fences, no think-blocks. "
    "Ensure every finding has at least one citation with source_type 'web' or 'kb'."
)

# Depth → (max_queries, max_fetches_total, max_kb_careers, budget_seconds)
# deep budget set to 28s: planning ~4s + parallel searches ~5s + synthesis ~15s
# for Qwen3-235B, leaving a small margin.
_DEPTH_CONFIG: dict[str, tuple[int, int, int, float]] = {
    "shallow": (3, 2, 1, 8.0),
    "deep": (8, 5, 999, 28.0),
}


class Researcher(Agent):
    """Produces a :class:`ResearchBrief` from a question + user model.

    Args:
        factory: Shared LLM factory.
        web_client: Web search client (``TavilyClient`` in production,
            ``MockWebSearchClient`` in tests).
        kb_root: Path to the ``kb/`` directory. Defaults to the project root
            ``kb/`` directory.
    """

    def __init__(
        self,
        factory: LLMFactory,
        web_client: WebSearchClient,
        kb_root: Path = _KB_ROOT,
    ) -> None:
        super().__init__("researcher", factory)
        self._planner_config = factory.config_for("researcher_planner")
        self._web = web_client
        self._kb = WorldKBRepo(kb_root=kb_root)

    async def run(
        self,
        input_data: ResearcherInput,
        *,
        turn_id: UUID | None = None,
        _budget_override_s: float | None = None,
    ) -> ResearchBrief:
        """Produce a :class:`ResearchBrief`.

        Args:
            input_data: Typed researcher input.
            turn_id: For agent_calls logging and research_briefs persistence.
            _budget_override_s: Internal — overrides the depth-based budget.
                Used in unit tests to force truncation without real delays.

        Returns:
            A validated :class:`ResearchBrief`. Never raises — returns an empty
            brief with caveats on all failure paths.
        """
        max_queries, max_fetches, max_kb, default_budget = _DEPTH_CONFIG[input_data.depth]
        budget_s = _budget_override_s if _budget_override_s is not None else default_budget
        deadline = time.monotonic() + budget_s

        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None

        # ── Phase 1: Plan ──────────────────────────────────────────────────
        plan = await self._plan(input_data, max_queries=max_queries)

        # ── Phase 2: Execute + synthesise (within budget) ──────────────────
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            logger.warning(
                "Researcher budget exhausted after planning phase; returning truncated brief."
            )
            brief = _truncated_brief(input_data, plan)
            fallback_reason = "budget_exhausted_after_planning"
        else:
            try:
                brief = await asyncio.wait_for(
                    self._execute_and_synthesise(
                        input_data,
                        plan,
                        max_fetches=max_fetches,
                        max_kb_careers=max_kb,
                        turn_id=turn_id,
                    ),
                    timeout=remaining,
                )
            except TimeoutError:
                logger.warning(
                    "Researcher timed out (budget=%.1fs, depth=%s); returning truncated brief.",
                    budget_s,
                    input_data.depth,
                )
                brief = _truncated_brief(input_data, plan)
                fallback_reason = "flow_timeout"

        latency = self.now_ms() - t0

        # Persist brief to DB (swallows errors so a DB outage never breaks the caller)
        brief = await self._persist(brief, turn_id=turn_id)

        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=_serialize_brief(brief),
            latency_ms=latency,
            tokens_in=None,
            tokens_out=None,
            error=error_str,
            retry_count=retry_count,
            fallback_reason=fallback_reason,
        )
        return brief

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _plan(
        self,
        input_data: ResearcherInput,
        *,
        max_queries: int,
    ) -> RetrievalPlan:
        """Phase 1: ask the planner model what to retrieve."""
        all_tags = sorted({tag for c in self._kb.list_all_careers() for tag in c.tags})
        prompt = self.render_prompt(
            "researcher_planner.j2",
            question=input_data.question,
            user_facts=input_data.user_facts,
            depth=input_data.depth,
            available_tags=all_tags,
            current_date=str(date.today()),
        )
        messages = [Message(role="user", content=prompt)]

        try:
            response = await self._client.complete(
                messages,
                model=self._planner_config.model,
                temperature=self._planner_config.temperature,
                max_tokens=self._planner_config.max_tokens,
                response_format="json",
                extra_body=self._planner_config.extra_body,
            )
            plan = _parse_plan(response.text, max_queries=max_queries)
        except Exception as exc:
            logger.warning("Researcher planner failed (%s); using empty plan.", exc)
            plan = RetrievalPlan()

        return plan

    async def _execute_and_synthesise(
        self,
        input_data: ResearcherInput,
        plan: RetrievalPlan,
        *,
        max_fetches: int,
        max_kb_careers: int,
        turn_id: UUID | None,
    ) -> ResearchBrief:
        """Phase 2: run searches + KB lookups in parallel, then synthesise."""
        # Run all searches and KB lookup concurrently
        search_tasks = [
            self._web.search(q, max_results=5, turn_id=turn_id)
            for q in plan.search_queries
        ]
        kb_task = asyncio.create_task(
            asyncio.to_thread(self._load_kb_careers, plan, max_kb_careers)
        )

        search_results_per_query: list[list[SearchResult]]
        kb_careers: list[dict[str, Any]]
        if search_tasks:
            gather_results = await asyncio.gather(*search_tasks, kb_task)
            search_results_per_query = list(gather_results[:-1])  # type: ignore[arg-type]
            kb_careers = gather_results[-1]  # type: ignore[assignment]
        else:
            kb_careers = await kb_task
            search_results_per_query = []

        # Fetch top-N content for promising results (parallel)
        fetch_pairs = _select_urls_to_fetch(search_results_per_query, max_fetches=max_fetches)
        if fetch_pairs:
            fetch_contents = await asyncio.gather(
                *[self._web.fetch(url) for _, url in fetch_pairs],
                return_exceptions=True,
            )
        else:
            fetch_contents = []

        # Attach fetched content back to results
        url_to_content: dict[str, str] = {}
        for (_, url), content in zip(fetch_pairs, fetch_contents, strict=False):
            if isinstance(content, str) and content:
                url_to_content[url] = content

        # Build structured web_results for the prompt
        web_results = _build_web_results(
            plan.search_queries, search_results_per_query, url_to_content
        )

        # Synthesise via Coach-tier LLM (with retry on parse failure)
        max_findings = 8 if input_data.depth == "deep" else 4
        prompt = self.render_prompt(
            "researcher.j2",
            question=input_data.question,
            user_facts=input_data.user_facts,
            depth=input_data.depth,
            web_results=web_results,
            kb_careers=kb_careers,
            current_date=str(date.today()),
            max_findings=max_findings,
        )
        messages = [Message(role="user", content=prompt)]
        response = await self.complete(messages, response_format="json")

        try:
            brief = _parse_brief(response.text, input_data.question)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "Researcher parse error on first attempt (%s); retrying.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            try:
                brief = _parse_brief(response.text, input_data.question)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                logger.warning("Researcher parse error after retry (%s); empty brief.", exc)
                brief = _empty_brief(input_data, plan, caveat=f"parse_error: {exc!s:.200}")

        return brief

    def _load_kb_careers(
        self,
        plan: RetrievalPlan,
        max_kb_careers: int,
    ) -> list[dict[str, Any]]:
        """Load and serialise KB career entries matched by the plan."""
        matched: dict[str, CareerEntry] = {}

        # Tag-based lookup (AND semantics per tag set)
        if plan.kb_tags:
            for entry in self._kb.search_by_tags(plan.kb_tags):
                matched[entry.name] = entry

        # Direct name lookup
        for name in plan.kb_career_names:
            entry = self._kb.get_career(name)
            if entry and entry.name not in matched:
                matched[entry.name] = entry

        careers = list(matched.values())[:max_kb_careers]

        result = []
        for entry in careers:
            kb_path = f"kb/careers/{entry.name}.yaml"
            yaml_text = yaml.dump(
                entry.model_dump(mode="json"), default_flow_style=False, allow_unicode=True
            )
            result.append(
                {
                    "name": entry.name,
                    "display_name": entry.display_name,
                    "kb_path": kb_path,
                    "yaml_text": yaml_text,
                }
            )
        return result

    async def _persist(
        self,
        brief: ResearchBrief,
        *,
        turn_id: UUID | None,
    ) -> ResearchBrief:
        """Insert brief into research_briefs and attach the generated brief_id."""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO research_briefs
                        (turn_id, question, findings, caveats, next_questions,
                         used_kb_files, web_searches_run, web_sources_consulted)
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb,
                            $6::jsonb, $7::jsonb, $8::jsonb)
                    RETURNING brief_id
                    """,
                    turn_id,
                    brief.question,
                    [f.model_dump(mode="json") for f in brief.findings],
                    brief.caveats,
                    brief.next_questions,
                    brief.used_kb_files,
                    brief.web_searches_run,
                    brief.web_sources_consulted,
                )
            brief = brief.model_copy(update={"brief_id": row["brief_id"]})
        except Exception as exc:
            logger.warning("Failed to persist research brief: %s", exc)
        return brief


# ------------------------------------------------------------------
# Pure helpers (no I/O)
# ------------------------------------------------------------------


def _parse_plan(raw: str, *, max_queries: int) -> RetrievalPlan:
    """Parse the planner LLM response into a RetrievalPlan."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    # Cap to depth-appropriate limits
    queries = data.get("search_queries", [])[:max_queries]
    return RetrievalPlan(
        search_queries=queries,
        kb_tags=data.get("kb_tags", []),
        kb_career_names=data.get("kb_career_names", []),
        rationale=data.get("rationale", ""),
    )


def _parse_brief(raw: str, question: str) -> ResearchBrief:
    """Parse the synthesis LLM response into a ResearchBrief.

    Strips findings that are missing ``citations`` (LLM sometimes omits them)
    rather than rejecting the entire response.  A partial result with N good
    findings is far better than an empty brief.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    # Ensure question is preserved
    data["question"] = question

    # Drop findings that lack citations — they fail the Finding schema and
    # would invalidate the entire response.  Log the loss so it's visible.
    findings = data.get("findings")
    if isinstance(findings, list):
        good = [f for f in findings if isinstance(f, dict) and f.get("citations")]
        dropped = len(findings) - len(good)
        if dropped:
            logger.warning(
                "Researcher: dropped %d/%d finding(s) missing citations before parse.",
                dropped,
                len(findings),
            )
            data["findings"] = good

    # Coerce caveats: LLM sometimes returns a plain string instead of a list.
    caveats = data.get("caveats")
    if isinstance(caveats, str):
        data["caveats"] = [caveats] if caveats.strip() else []

    # Pydantic validates the remaining Citation/Finding structure
    return ResearchBrief(**data)


def _select_urls_to_fetch(
    results_per_query: list[list[SearchResult]],
    *,
    max_fetches: int,
) -> list[tuple[str, str]]:
    """Select top-N URLs to fetch across all query results.

    Picks the highest-scoring result per query first, then fills remaining
    slots. Deduplicates by URL. Returns list of (query, url) pairs.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []

    # Round-robin: one top result per query first
    for results in results_per_query:
        if results and results[0].url not in seen:
            pairs.append(("", results[0].url))
            seen.add(results[0].url)
            if len(pairs) >= max_fetches:
                break

    # Fill remaining slots with second results
    for results in results_per_query:
        if len(pairs) >= max_fetches:
            break
        if len(results) > 1 and results[1].url not in seen:
            pairs.append(("", results[1].url))
            seen.add(results[1].url)

    return pairs


def _build_web_results(
    queries: list[str],
    results_per_query: list[list[SearchResult]],
    url_to_content: dict[str, str],
) -> list[dict[str, Any]]:
    """Build the web_results structure passed to the synthesis prompt."""
    out = []
    for query, results in zip(queries, results_per_query, strict=False):
        entries = []
        for r in results:
            entries.append(
                {
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "content": url_to_content.get(r.url),
                }
            )
        out.append({"query": query, "results": entries})
    return out


def _truncated_brief(input_data: ResearcherInput, plan: RetrievalPlan) -> ResearchBrief:
    """Minimal brief returned when the latency budget is exceeded."""
    return ResearchBrief(
        question=input_data.question,
        findings=[],
        caveats=["research truncated due to time budget"],
        next_questions=[],
        used_kb_files=[],
        web_searches_run=list(plan.search_queries),
        web_sources_consulted=[],
    )


def _empty_brief(
    input_data: ResearcherInput,
    plan: RetrievalPlan,
    *,
    caveat: str,
) -> ResearchBrief:
    """Minimal brief returned on graceful failure (both sources empty or parse error)."""
    return ResearchBrief(
        question=input_data.question,
        findings=[],
        caveats=[caveat],
        next_questions=[],
        used_kb_files=[],
        web_searches_run=list(plan.search_queries),
        web_sources_consulted=[],
    )


def _serialize_input(inp: ResearcherInput) -> dict[str, Any]:
    return {
        "question": inp.question[:200],
        "depth": inp.depth,
        "num_facts": len(inp.user_facts),
        "num_hypotheses": len(inp.active_hypotheses),
    }


def _serialize_brief(brief: ResearchBrief) -> dict[str, Any]:
    return {
        "num_findings": len(brief.findings),
        "num_caveats": len(brief.caveats),
        "web_searches_run": brief.web_searches_run,
        "web_sources_consulted": brief.web_sources_consulted,
        "used_kb_files": brief.used_kb_files,
        "brief_id": str(brief.brief_id) if brief.brief_id else None,
    }
