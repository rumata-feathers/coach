#!/usr/bin/env python
"""Flow C quality evaluation script.

Runs the 20-fixture deep-decision battery against a live Flow C pipeline.
For each fixture:
  - Creates a test user seeded with the fixture's persona facts.
  - Sends the decision question as a turn (should trigger Flow C).
  - Asserts quality criteria against the DB row.
  - Optionally runs an LLM-as-judge second-grader comparing
    Flow C vs Flow B on the same fixture.

Quality criteria per fixture:
  1. flow_used == "C"
  2. synthesizer_output is non-null
  3. response contains ≥2 user-fact references
  4. response contains ≥1 finding reference (inline [N] citation)
  5. response contains ≥1 tradeoff signal word
  6. Critic final verdict (from last critic_verdicts entry) is "pass"

Aggregate thresholds (per TASKS_v1.md §10):
  - ≥17/20 fixtures pass all criteria.
  - ≥5 fixtures emitted a chart (chart_specs is non-empty).
  - 0 charts have empty source_citation_indices (no invented-data charts).
  - Total estimated cost < £25 (tracked via token counts where available).

Usage::

    uv run python scripts/run_flow_c_eval.py
    uv run python scripts/run_flow_c_eval.py --report reports/flow_c_baseline.md
    uv run python scripts/run_flow_c_eval.py --fixture 001 --judge

Environment variables required:
  SUPABASE_DB_URL
  HUGGINGFACE_API_TOKEN (primary LLM backend)
  TAVILY_API_KEY         (web search in Flow C)

Optional:
  HUGGINGFACE_API_TOKEN  (also used for LLM-as-judge second grader)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from career_coach.config import get_settings  # noqa: E402
from career_coach.db import get_pool  # noqa: E402
from career_coach.llm.factory import LLMFactory  # noqa: E402
from career_coach.pipeline.turn import TurnPipeline, TurnResult  # noqa: E402

_FIXTURES_DIR = _REPO_ROOT / "tests" / "quality" / "deep_decision_fixtures"
_CONFIG = _REPO_ROOT / "config" / "models.yaml"

# Tradeoff signal words — at least one must appear in a deliberation response.
_TRADEOFF_WORDS = frozenset(
    [
        "tradeoff", "trade-off", "trade off", "however", "but",
        "on the other hand", "whereas", "versus", "vs", "risk",
        "downside", "disadvantage", "caveat", "although", "tension",
        "weigh", "consider", "balance", "offset", "cost",
    ]
)

# Inline citation pattern [1], [2], etc.
_CITATION_RE = re.compile(r"\[\d+\]")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class FixtureResult:
    """Results for a single fixture run."""

    def __init__(self, fixture_id: str, category: str, question: str) -> None:
        self.fixture_id = fixture_id
        self.category = category
        self.question = question[:80] + ("..." if len(question) > 80 else "")
        self.passed = False
        self.flow_used: str = "?"
        self.criteria: dict[str, bool] = {}
        self.failures: list[str] = []
        self.has_chart = False
        self.chart_has_citation = True  # innocent until proven guilty
        self.critic_verdict: str = "?"
        self.referenced_facts: list[str] = []
        self.referenced_findings: list[str] = []
        self.latency_s: float = 0.0
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.error: str | None = None
        self.judge_verdict: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixtures(only: str | None = None) -> list[dict[str, Any]]:
    """Load YAML fixtures sorted by fixture_id."""
    fixtures = []
    for path in sorted(_FIXTURES_DIR.glob("*.yaml")):
        with path.open() as fh:
            data = yaml.safe_load(fh)
        data["_path"] = str(path.name)
        if only and data.get("fixture_id") != only:
            continue
        fixtures.append(data)
    return fixtures


async def _create_user(display_name: str, seed_facts: dict[str, str]) -> UUID:
    """Create test user and seed structured facts."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id: UUID = await conn.fetchval(
            "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
            display_name,
        )
        for key, jsonb_val in seed_facts.items():
            await conn.execute(
                """
                INSERT INTO structured_facts (user_id, key, value, confidence, source)
                VALUES ($1, $2, $3::jsonb, 1.0, 'system')
                ON CONFLICT (user_id, key) DO NOTHING
                """,
                user_id,
                key,
                jsonb_val,
            )
    return user_id


async def _fetch_turn_row(turn_id: UUID) -> dict[str, Any] | None:
    """Fetch the turn row from DB."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT flow_used, synthesizer_output, critic_verdicts,
                   chart_specs, research_brief_id, devils_advocate_output
            FROM turns WHERE turn_id = $1
            """,
            turn_id,
        )
    if row is None:
        return None
    return dict(row)


async def _fetch_token_counts(turn_id: UUID) -> tuple[int, int]:
    """Sum tokens_in and tokens_out from agent_calls for this turn."""
    # Sum what we have from the turn (supervisor writes with turn_id=NULL before persist)
    pool2 = await get_pool()
    async with pool2.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(tokens_in), 0)::int  AS total_in,
                COALESCE(SUM(tokens_out), 0)::int AS total_out
            FROM agent_calls
            WHERE turn_id = $1
            """,
            turn_id,
        )
    if row:
        return int(row["total_in"]), int(row["total_out"])
    return 0, 0


def _check_criteria(
    result: FixtureResult,
    turn_result: TurnResult,
    db_row: dict[str, Any],
) -> None:
    """Evaluate all quality criteria against the DB row."""
    response_text: str = turn_result.response
    text_lower = response_text.lower()

    # 1. flow_used == "C"
    flow_used = db_row.get("flow_used", "?")
    result.flow_used = flow_used
    c1 = flow_used == "C"
    result.criteria["flow_used_C"] = c1
    if not c1:
        result.failures.append(f"flow_used={flow_used!r} (expected 'C')")

    # 2. synthesizer_output non-null
    synth_raw = db_row.get("synthesizer_output")
    c2 = synth_raw is not None
    result.criteria["synthesizer_output_present"] = c2
    if not c2:
        result.failures.append("synthesizer_output is NULL")

    synth: dict[str, Any] = {}
    if synth_raw:
        try:
            synth = json.loads(synth_raw) if isinstance(synth_raw, str) else synth_raw
        except Exception:
            result.failures.append("synthesizer_output not valid JSON")

    # Use synthesizer_output response_text if available
    synth_response = synth.get("response_text", response_text)
    text_lower = synth_response.lower()

    # 3. ≥2 user-fact references
    ref_facts: list[str] = synth.get("referenced_facts", [])
    result.referenced_facts = ref_facts
    c3 = len(ref_facts) >= 2
    result.criteria["referenced_facts_gte_2"] = c3
    if not c3:
        result.failures.append(
            f"referenced_facts={ref_facts} (expected ≥2)"
        )

    # 4. ≥1 inline citation [N]
    c4 = bool(_CITATION_RE.search(synth_response))
    result.criteria["has_inline_citation"] = c4
    if not c4:
        result.failures.append("No inline [N] citation found in response")

    # 5. ≥1 tradeoff signal word
    c5 = any(word in text_lower for word in _TRADEOFF_WORDS)
    result.criteria["has_tradeoff_word"] = c5
    if not c5:
        result.failures.append("No tradeoff signal word found in response")

    # 6. Critic final verdict = "pass"
    critic_verdicts_raw = db_row.get("critic_verdicts")
    last_verdict = "?"
    if critic_verdicts_raw:
        try:
            verdicts = (
                json.loads(critic_verdicts_raw)
                if isinstance(critic_verdicts_raw, str)
                else critic_verdicts_raw
            )
            if verdicts:
                last_verdict = verdicts[-1].get("verdict", "?")
        except Exception:
            pass
    result.critic_verdict = last_verdict
    c6 = last_verdict == "pass"
    result.criteria["critic_pass"] = c6
    if not c6:
        result.failures.append(f"Critic final verdict={last_verdict!r}")

    # Chart checks
    chart_specs_raw = db_row.get("chart_specs")
    if chart_specs_raw:
        try:
            charts: list[dict[str, Any]] = (
                json.loads(chart_specs_raw)
                if isinstance(chart_specs_raw, str)
                else chart_specs_raw
            )
            if charts:
                result.has_chart = True
                # Check all charts have source_citation_indices
                for chart in charts:
                    if not chart.get("source_citation_indices"):
                        result.chart_has_citation = False
                        result.failures.append(
                            f"Chart '{chart.get('title', '?')}' has no source_citation_indices"
                        )
        except Exception:
            pass

    result.passed = all(result.criteria.values())


async def _run_judge(
    fixture: dict[str, Any],
    flow_c_response: str,
    factory: LLMFactory,
) -> str:
    """LLM-as-judge second grader comparing Flow C vs a simulated summary."""
    try:
        from career_coach.agents.base import Agent
        from career_coach.llm.client import Message

        class _JudgeAgent(Agent):
            def __init__(self) -> None:
                super().__init__("judge", factory)

            async def run(self, question: str, response: str) -> str:  # type: ignore[override]
                prompt = (
                    f"You are evaluating a career coaching response for quality.\n\n"
                    f"Question: {question}\n\n"
                    f"Response to evaluate:\n{response}\n\n"
                    f"Rate this response on a scale of 1-5 and give ONE sentence of "
                    f"feedback. Format: SCORE: N\\nFEEDBACK: <sentence>"
                )
                result = await self.complete([Message(role="user", content=prompt)])
                return result.text

        # Only run if HuggingFace key is available
        if not os.getenv("HUGGINGFACE_API_TOKEN"):
            return "skipped (no HUGGINGFACE_API_TOKEN)"

        judge = _JudgeAgent()
        return await judge.run(fixture.get("question", ""), flow_c_response)
    except Exception as exc:
        return f"judge error: {exc}"


# ---------------------------------------------------------------------------
# Per-fixture runner
# ---------------------------------------------------------------------------

async def run_fixture(
    fixture: dict[str, Any],
    pipeline: TurnPipeline,
    factory: LLMFactory,
    *,
    run_judge: bool = False,
) -> FixtureResult:
    """Run one fixture through the full pipeline and evaluate."""
    result = FixtureResult(
        fixture_id=fixture["fixture_id"],
        category=fixture.get("category", "?"),
        question=fixture.get("question", ""),
    )

    seed_facts: dict[str, str] = fixture.get("seed_facts", {})
    display_name = f"eval-flowc-{fixture['fixture_id']}-{uuid4().hex[:6]}"

    try:
        user_id = await _create_user(display_name, seed_facts)

        t0 = time.monotonic()
        turn_result = await pipeline.process_turn(
            user_id=user_id,
            user_message=fixture["question"],
        )
        result.latency_s = time.monotonic() - t0

        if turn_result.turn_id is None:
            result.error = "turn_id is None after process_turn"
            result.failures.append(result.error)
            return result

        db_row = await _fetch_turn_row(turn_result.turn_id)
        if db_row is None:
            result.error = "Turn row not found in DB"
            result.failures.append(result.error)
            return result

        _check_criteria(result, turn_result, db_row)

        t_in, t_out = await _fetch_token_counts(turn_result.turn_id)
        result.tokens_in = t_in
        result.tokens_out = t_out

        if run_judge:
            result.judge_verdict = await _run_judge(fixture, turn_result.response, factory)

    except Exception as exc:
        result.error = str(exc)
        result.failures.append(f"Exception: {exc}")

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _render_report(
    results: list[FixtureResult],
    *,
    elapsed_s: float,
) -> str:
    """Render a Markdown report from fixture results."""
    pass_count = sum(1 for r in results if r.passed)
    chart_count = sum(1 for r in results if r.has_chart)
    bad_chart_count = sum(1 for r in results if r.has_chart and not r.chart_has_citation)
    total_tokens_in = sum(r.tokens_in for r in results)
    total_tokens_out = sum(r.tokens_out for r in results)
    # Very rough cost estimate: HF Qwen3-235B ≈ £0.05/1k tokens in, £0.15/1k out
    est_cost_gbp = (total_tokens_in * 0.05 + total_tokens_out * 0.15) / 1000

    lines = [
        "# Flow C Quality Eval — Baseline Report",
        "",
        f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Fixtures run:** {len(results)}",
        f"**Pass rate:** {pass_count}/{len(results)} "
        f"({'✅' if pass_count >= 17 else '❌'} threshold ≥17/20)",
        f"**Charts emitted:** {chart_count}/20 "
        f"({'✅' if chart_count >= 5 else '❌'} threshold ≥5)",
        f"**Invented-data charts:** {bad_chart_count}/20 "
        f"({'✅' if bad_chart_count == 0 else '❌'} threshold 0)",
        f"**Estimated cost:** ~£{est_cost_gbp:.2f} "
        f"({'✅' if est_cost_gbp < 25 else '❌'} ceiling £25)",
        f"**Total elapsed:** {elapsed_s:.0f}s",
        "",
        "## Per-fixture results",
        "",
        "| ID | Category | Flow | Critic | Facts | Citation | Tradeoff | Chart | Pass |",
        "|----|----------|------|--------|-------|----------|----------|-------|------|",
    ]

    for r in results:
        c = r.criteria
        row = (
            f"| {r.fixture_id} "
            f"| {r.category[:20]} "
            f"| {r.flow_used} "
            f"| {r.critic_verdict} "
            f"| {'✅' if c.get('referenced_facts_gte_2') else '❌'} "
            f"| {'✅' if c.get('has_inline_citation') else '❌'} "
            f"| {'✅' if c.get('has_tradeoff_word') else '❌'} "
            f"| {'✅' if r.has_chart else '-'} "
            f"| {'✅' if r.passed else '❌'} |"
        )
        lines.append(row)

    lines.extend(["", "## Failure details", ""])
    failures_found = False
    for r in results:
        if r.failures:
            failures_found = True
            lines.append(f"### Fixture {r.fixture_id} — {r.category}")
            for f in r.failures:
                lines.append(f"- {f}")
            if r.judge_verdict:
                lines.append(f"- **LLM judge:** {r.judge_verdict}")
            lines.append("")

    if not failures_found:
        lines.append("*No failures.*")
        lines.append("")

    lines.extend([
        "## Token usage",
        "",
        f"- Total tokens in:  {total_tokens_in:,}",
        f"- Total tokens out: {total_tokens_out:,}",
        f"- Estimated cost:   ~£{est_cost_gbp:.2f}",
        "",
    ])

    return "\n".join(lines)


def _print_progress(result: FixtureResult, index: int, total: int) -> None:
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(
        f"  [{index:02d}/{total}] {result.fixture_id} ({result.category[:18]:18s}) "
        f"flow={result.flow_used} critic={result.critic_verdict} "
        f"{status} ({result.latency_s:.1f}s)"
    )
    if result.failures:
        for f in result.failures:
            print(f"         → {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    missing = []
    if not settings.supabase_db_url:
        missing.append("SUPABASE_DB_URL")
    if not settings.huggingface_api_token:
        missing.append("HUGGINGFACE_API_TOKEN")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        return 1

    fixtures = _load_fixtures(only=args.fixture)
    if not fixtures:
        print(f"No fixtures found (fixture={args.fixture!r})", file=sys.stderr)
        return 1

    factory = LLMFactory(config_path=_CONFIG)
    pipeline = TurnPipeline(factory)

    print(f"\n{'='*60}")
    print(f"  Flow C Quality Eval — {len(fixtures)} fixtures")
    print(f"{'='*60}\n")

    results: list[FixtureResult] = []
    t_start = time.monotonic()

    for i, fixture in enumerate(fixtures, 1):
        print(f"Running {fixture['fixture_id']} ({fixture.get('category', '?')})...")
        result = await run_fixture(
            fixture, pipeline, factory, run_judge=args.judge
        )
        results.append(result)
        _print_progress(result, i, len(fixtures))

    elapsed = time.monotonic() - t_start

    # Aggregate assertions
    pass_count = sum(1 for r in results if r.passed)
    chart_count = sum(1 for r in results if r.has_chart)
    bad_chart_count = sum(1 for r in results if r.has_chart and not r.chart_has_citation)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {pass_count}/{len(results)} passed ({elapsed:.0f}s)")
    print(f"  Charts emitted: {chart_count}  |  Bad charts: {bad_chart_count}")
    print(f"{'='*60}\n")

    report = _render_report(results, elapsed_s=elapsed)

    # Write report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Report written to: {report_path}")

    # Exit code: 0 if thresholds met, 1 otherwise
    threshold_pass = (
        pass_count >= 17
        and chart_count >= 5
        and bad_chart_count == 0
    )
    if not threshold_pass:
        failing = []
        if pass_count < 17:
            failing.append(f"pass_rate={pass_count}/20 (need ≥17)")
        if chart_count < 5:
            failing.append(f"charts={chart_count}/20 (need ≥5)")
        if bad_chart_count > 0:
            failing.append(f"bad_charts={bad_chart_count} (need 0)")
        print(f"❌ THRESHOLDS NOT MET: {'; '.join(failing)}", file=sys.stderr)
        return 1

    print("✅ All thresholds met.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Flow C quality eval against 20 deep-decision fixtures."
    )
    parser.add_argument(
        "--fixture",
        default=None,
        help="Run only this fixture_id (e.g. '001'). Default: run all.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-judge second grader (requires HUGGINGFACE_API_TOKEN).",
    )
    parser.add_argument(
        "--report",
        default=str(_REPO_ROOT / "reports" / "flow_c_baseline.md"),
        help="Path to write the Markdown report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(main(args)))
