#!/usr/bin/env python
"""Distillation quality evaluation harness.

For each fixture in tests/quality/distillation_fixtures/:
  1. Creates a fresh user in the DB.
  2. Seeds the fixture's facts into structured_facts.
  3. Replays the fixture's turns through process_turn (real LLM).
  4. Runs distillation.
  5. Reads back the produced hypotheses.
  6. For each expected/forbidden theme, asks a judge-tier LLM:
       "Does this hypothesis set include one expressing theme X?"
  7. Scores: matched_expected - matched_forbidden per fixture.

Aggregates average score across fixtures.
Prints a Markdown report and writes it to reports/distillation_baseline.md.

Usage:
    uv run python scripts/run_distillation_eval.py
    uv run python scripts/run_distillation_eval.py --output reports/distillation_eval_20260101.md

Requires: HUGGINGFACE_API_TOKEN in .env, running Postgres.
Skips automatically if token is absent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_coach.config import get_settings
from career_coach.db import close_pool, get_pool
from career_coach.jobs.distillation import run_distillation
from career_coach.llm.client import Message
from career_coach.llm.factory import LLMFactory
from career_coach.memory.semantic import SemanticRepo
from career_coach.pipeline.turn import TurnPipeline

_FIXTURES_DIR = (
    Path(__file__).resolve().parents[1] / "tests" / "quality" / "distillation_fixtures"
)
_JUDGE_MODEL_ALIAS = "coach"  # Qwen3-235B — highest-quality judge


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _create_user(display_name: str) -> UUID:
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id: UUID = await conn.fetchval(
            "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
            display_name,
        )
    return user_id


async def _seed_facts(user_id: UUID, facts: dict[str, object]) -> None:
    """Write seed facts directly into structured_facts table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        for key, value in facts.items():
            await conn.execute(
                """
                INSERT INTO structured_facts (user_id, key, value, confidence, source)
                VALUES ($1, $2, $3::jsonb, 0.9, 'seeded')
                ON CONFLICT (user_id, key) DO UPDATE
                    SET value = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        source = EXCLUDED.source
                """,
                user_id,
                key,
                json.dumps(value),
            )


# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------


async def _judge_theme_present(
    theme: str,
    hypotheses_text: str,
    factory: LLMFactory,
) -> bool:
    """Ask the judge LLM whether the hypothesis set expresses the given theme.

    Returns True if the judge says yes, False otherwise.
    """
    prompt = (
        "You are evaluating whether a set of hypotheses about a person expresses a given theme.\n\n"
        f"Theme to check: \"{theme}\"\n\n"
        "Hypotheses:\n"
        f"{hypotheses_text}\n\n"
        "Does this set of hypotheses include at least one that substantively expresses "
        "the theme? Answer with JSON only:\n"
        "{\"present\": true, \"reason\": \"one sentence\"} or {\"present\": false, \"reason\": \"one sentence\"}"
    )
    client = factory.client_for(_JUDGE_MODEL_ALIAS)
    model = factory.config_for(_JUDGE_MODEL_ALIAS).model
    try:
        response = await client.complete(
            [Message(role="user", content=prompt)],
            model=model,
            temperature=0.0,
            max_tokens=200,
            response_format="json",
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("```"))
        data = json.loads(text)
        return bool(data.get("present", False))
    except Exception as exc:
        print(f"    [WARN] Judge call failed for theme '{theme}': {exc}")
        return False


# ---------------------------------------------------------------------------
# Per-fixture runner
# ---------------------------------------------------------------------------


async def _run_fixture(
    fixture: dict[str, object],
    factory: LLMFactory,
    pipeline: TurnPipeline,
) -> dict[str, object]:
    """Run one fixture end-to-end and return a result dict."""
    name = str(fixture["name"])
    print(f"\n── Fixture: {name} ──────────────────────────────────────")

    # Create user + seed facts
    user_id = await _create_user(f"distil_eval_{name}_{uuid4().hex[:6]}")
    seed_facts = fixture.get("seed_facts", {})
    if isinstance(seed_facts, dict):
        await _seed_facts(user_id, seed_facts)

    # Replay turns
    turns = fixture.get("turns", [])
    assert isinstance(turns, list)
    session_id: UUID | None = None

    for turn in turns:
        role = turn.get("role", "user")
        text = str(turn.get("text", ""))
        if role != "user":
            continue  # assistant turns are just context in the fixture; skip
        result = await pipeline.process_turn(
            user_id=user_id,
            user_message=text,
            session_id=session_id,
        )
        session_id = result.session_id
        await asyncio.sleep(0.05)  # allow background Profiler to complete
        print(f"  turn → {result.response[:80]}…")

    # Run distillation
    distil = await run_distillation(user_id, factory=factory)
    print(f"  distillation: {distil}")

    # Read hypotheses
    hypotheses = await SemanticRepo().get_active(user_id)
    if not hypotheses:
        print("  WARNING: No hypotheses produced.")
        hyp_text = "(none)"
    else:
        hyp_lines = [f"- [{h.confidence:.2f}] {h.statement}" for h in hypotheses]
        hyp_text = "\n".join(hyp_lines)
        print("  hypotheses:\n" + "\n".join(f"    {ln}" for ln in hyp_lines))

    # Judge expected themes
    expected_results: list[dict[str, object]] = []
    expected_hypotheses = fixture.get("expected_hypotheses", [])
    assert isinstance(expected_hypotheses, list)
    for exp in expected_hypotheses:
        theme = str(exp.get("theme", ""))
        present = await _judge_theme_present(theme, hyp_text, factory)
        expected_results.append({"theme": theme, "present": present})
        mark = "✓" if present else "✗"
        print(f"  [{mark}] expected: {theme}")

    # Judge forbidden themes
    forbidden_results: list[dict[str, object]] = []
    forbidden_hypotheses = fixture.get("forbidden_hypotheses", [])
    assert isinstance(forbidden_hypotheses, list)
    for theme in forbidden_hypotheses:
        theme_str = str(theme)
        present = await _judge_theme_present(theme_str, hyp_text, factory)
        forbidden_results.append({"theme": theme_str, "present": present})
        mark = "✗ (BAD)" if present else "✓"
        print(f"  [{mark}] forbidden: {theme_str}")

    expected_matched = sum(1 for r in expected_results if r["present"])
    forbidden_matched = sum(1 for r in forbidden_results if r["present"])
    score = expected_matched - forbidden_matched
    max_score = len(expected_results)
    normalised = score / max_score if max_score > 0 else 0.0

    print(f"  score: {score}/{max_score} → normalised {normalised:.2f}")

    return {
        "name": name,
        "persona": str(fixture.get("persona", "")),
        "expected_results": expected_results,
        "forbidden_results": forbidden_results,
        "score": score,
        "max_score": max_score,
        "normalised": normalised,
        "hypotheses": [{"statement": h.statement, "confidence": h.confidence} for h in hypotheses],
        "distillation": distil,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _render_report(
    results: list[dict[str, object]],
    elapsed_s: float,
) -> str:
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    total_score = sum(float(r["score"]) for r in results)
    total_max = sum(float(r["max_score"]) for r in results)
    avg = total_score / len(results) if results else 0.0
    pass_rate = sum(1 for r in results if float(r["normalised"]) >= 0.5) / len(results) if results else 0.0

    lines = [
        "# Distillation Quality Baseline",
        f"\nGenerated: {now}  |  Elapsed: {elapsed_s:.1f}s",
        f"\n**Fixtures:** {len(results)}  |  **Avg normalised score:** {avg:.2f}  "
        f"|  **Pass rate (≥0.5):** {pass_rate:.0%}  |  **Total:** {total_score:.0f}/{total_max:.0f}",
        "\n---\n",
        "## Per-fixture results\n",
    ]

    for r in results:
        norm = float(r["normalised"])
        icon = "✓" if norm >= 0.5 else "✗"
        lines.append(f"### [{icon}] `{r['name']}` — {norm:.2f}")
        lines.append(f"\n*Persona:* {r['persona']}\n")

        hyps = r.get("hypotheses", [])
        assert isinstance(hyps, list)
        if hyps:
            lines.append("**Hypotheses produced:**\n")
            for h in hyps:
                assert isinstance(h, dict)
                lines.append(f"- [{h.get('confidence', 0):.2f}] {h.get('statement', '')}")
        else:
            lines.append("*No hypotheses produced.*")

        lines.append("\n**Expected themes:**\n")
        for exp in r.get("expected_results", []):
            assert isinstance(exp, dict)
            mark = "✓" if exp.get("present") else "✗"
            lines.append(f"- [{mark}] {exp.get('theme', '')}")

        lines.append("\n**Forbidden themes:**\n")
        for fbd in r.get("forbidden_results", []):
            assert isinstance(fbd, dict)
            bad = fbd.get("present", False)
            mark = "✗ FAIL" if bad else "✓ OK"
            lines.append(f"- [{mark}] {fbd.get('theme', '')}")

        lines.append("")

    lines += [
        "---\n",
        "## Interpretation\n",
        "- Score per fixture = matched_expected - matched_forbidden",
        "- Normalised = score / len(expected_hypotheses)",
        "- Pass threshold: normalised ≥ 0.5 per fixture, avg ≥ 0.7 overall",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main(output_path: Path) -> int:
    settings = get_settings()
    if not settings.huggingface_api_token:
        print(
            "ERROR: HUGGINGFACE_API_TOKEN not set. Add it to .env and retry.",
            file=sys.stderr,
        )
        return 1

    fixture_files = sorted(_FIXTURES_DIR.glob("*.yaml"))
    if not fixture_files:
        print(f"ERROR: No fixture files found in {_FIXTURES_DIR}", file=sys.stderr)
        return 1

    fixtures = []
    for path in fixture_files:
        with path.open() as f:
            fixtures.append(yaml.safe_load(f))

    print(f"Running distillation eval on {len(fixtures)} fixtures…\n")

    factory = LLMFactory()
    pipeline = TurnPipeline(factory)
    # Patch onboarding so eval fixtures exercise the standard Coach path
    pipeline._onboarding_policy = AsyncMock()  # type: ignore[method-assign]
    pipeline._onboarding_policy.is_new_user.return_value = False  # type: ignore[union-attr]

    t0 = asyncio.get_event_loop().time()
    results = []
    for fixture in fixtures:
        result = await _run_fixture(fixture, factory, pipeline)
        results.append(result)

    elapsed = asyncio.get_event_loop().time() - t0

    # Summary
    avg = sum(float(r["normalised"]) for r in results) / len(results) if results else 0.0
    print(f"\n{'='*60}")
    print(f"AVERAGE NORMALISED SCORE: {avg:.2f}  (pass threshold: 0.70)")
    passed = avg >= 0.70
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")

    # Write report
    report = _render_report(results, elapsed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")

    await close_pool()
    return 0 if passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distillation quality evaluation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "distillation_baseline.md",
        help="Path to write Markdown report (default: reports/distillation_baseline.md).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(_main(args.output)))
