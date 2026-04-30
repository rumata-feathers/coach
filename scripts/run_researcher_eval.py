"""Researcher quality evaluation script.

Runs the 20-fixture battery in ``tests/quality/researcher_fixtures/`` against
a real Researcher agent (real Tavily + real LLM). Asserts §14.2 thresholds:

  - ≥18/20 produce ≥2 findings.
  - ≥18/20 have every finding cited to a fetched URL or KB entry.
  - 0/20 contain numeric claims without a web or sourced-KB citation.

Outputs a per-fixture table to stdout and optionally writes
``reports/researcher_baseline.md``.

Usage::

    uv run python scripts/run_researcher_eval.py
    uv run python scripts/run_researcher_eval.py --depth deep --report

Environment variables required:
  TAVILY_API_KEY
  HUGGINGFACE_API_TOKEN (or ANTHROPIC_API_KEY depending on provider config)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add project src to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from career_coach.agents.researcher import Researcher  # noqa: E402
from career_coach.llm.factory import LLMFactory  # noqa: E402
from career_coach.models.agent_io import ResearchBrief, ResearcherInput  # noqa: E402
from career_coach.web.factory import get_client  # noqa: E402

_FIXTURES_DIR = _REPO_ROOT / "tests" / "quality" / "researcher_fixtures"
_REPORTS_DIR = _REPO_ROOT / "reports"
_CONFIG = _REPO_ROOT / "config" / "models.yaml"
_KB_ROOT = _REPO_ROOT / "kb"


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def _check_every_finding_cited(brief: ResearchBrief) -> tuple[bool, str]:
    """Assert every finding has ≥1 citation. Returns (pass, explanation)."""
    for finding in brief.findings:
        if not finding.citations:
            return False, f"Finding uncited: {finding.claim[:80]!r}"
    return True, "all findings cited"


def _check_no_uncited_numerics(brief: ResearchBrief) -> tuple[bool, str]:
    """Assert no numeric finding lacks a valid citation.

    A numeric finding is valid if it has ≥1 web citation (URL present) OR
    ≥1 KB citation. Snippet-only claims are already excluded by the prompt,
    so we trust the model's source_type here.
    """
    for finding in brief.findings:
        if not finding.is_numeric:
            continue
        has_valid_citation = any(
            c.url is not None or c.kb_path is not None for c in finding.citations
        )
        if not has_valid_citation:
            return False, f"Numeric finding has no valid citation: {finding.claim[:80]!r}"
    return True, "no uncited numerics"


def _evaluate_brief(fixture: dict[str, Any], brief: ResearchBrief) -> dict[str, Any]:
    """Score one fixture against §14.2 thresholds."""
    min_findings = fixture.get("expected_min_findings", 2)
    is_contrarian = fixture.get("is_contrarian", False)

    findings_count = len(brief.findings)
    has_enough_findings = findings_count >= min_findings or is_contrarian

    all_cited, cited_reason = _check_every_finding_cited(brief)
    no_bad_numerics, numeric_reason = _check_no_uncited_numerics(brief)

    # Contrarian fixtures are allowed to have 0 findings with honest caveats
    if is_contrarian:
        has_honest_caveat = len(brief.caveats) >= 1
        overall_pass = has_honest_caveat and no_bad_numerics
    else:
        overall_pass = has_enough_findings and all_cited and no_bad_numerics

    return {
        "fixture_id": fixture["id"],
        "question": fixture["question"][:80],
        "is_contrarian": is_contrarian,
        "findings_count": findings_count,
        "has_enough_findings": has_enough_findings,
        "all_cited": all_cited,
        "cited_reason": cited_reason,
        "no_bad_numerics": no_bad_numerics,
        "numeric_reason": numeric_reason,
        "caveats": brief.caveats,
        "web_sources": brief.web_sources_consulted,
        "kb_files": brief.used_kb_files,
        "overall_pass": overall_pass,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_fixture(
    fixture: dict[str, Any],
    researcher: Researcher,
    depth: str,
) -> dict[str, Any]:
    """Run one fixture and return an evaluation result dict."""
    inp = ResearcherInput(
        question=fixture["question"],
        user_facts=fixture.get("persona", {}),
        depth=depth,
    )
    t0 = time.monotonic()
    try:
        brief = await researcher.run(inp)
        elapsed = time.monotonic() - t0
        result = _evaluate_brief(fixture, brief)
        result["latency_s"] = round(elapsed, 2)
    except Exception as exc:
        result = {
            "fixture_id": fixture["id"],
            "question": fixture["question"][:80],
            "is_contrarian": fixture.get("is_contrarian", False),
            "overall_pass": False,
            "error": str(exc),
            "latency_s": round(time.monotonic() - t0, 2),
        }
    return result


async def run_eval(depth: str = "shallow") -> list[dict[str, Any]]:
    """Run the full 20-fixture battery."""
    fixtures = sorted(_FIXTURES_DIR.glob("*.yaml"))
    if not fixtures:
        print(f"ERROR: No fixture files found in {_FIXTURES_DIR}", file=sys.stderr)
        sys.exit(1)

    factory = LLMFactory(config_path=_CONFIG)
    web_client = get_client("researcher")
    researcher = Researcher(factory=factory, web_client=web_client, kb_root=_KB_ROOT)

    results = []
    for i, path in enumerate(fixtures, 1):
        fixture = yaml.safe_load(path.read_text(encoding="utf-8"))
        print(f"[{i:02d}/{len(fixtures)}] {fixture['id']} ... ", end="", flush=True)
        result = await run_fixture(fixture, researcher, depth=depth)
        status = "PASS" if result["overall_pass"] else "FAIL"
        print(f"{status} ({result.get('latency_s', '?')}s)")
        if result.get("error"):
            print(f"         ERROR: {result['error']}")
        results.append(result)

    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])
    errors = sum(1 for r in results if r.get("error"))
    non_contrarian = [r for r in results if not r.get("is_contrarian")]
    enough_findings = sum(1 for r in non_contrarian if r.get("has_enough_findings"))
    all_cited = sum(1 for r in non_contrarian if r.get("all_cited"))
    no_numerics = sum(1 for r in results if r.get("no_bad_numerics", True))

    print()
    print("=" * 60)
    print(f"RESEARCHER QUALITY BATTERY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"Total fixtures:      {total}")
    print(f"Overall PASS:        {passed}/{total}")
    print(f"Errors:              {errors}")
    print()
    print("§14.2 Thresholds:")
    nc = len(non_contrarian)
    print(f"  ≥2 findings:       {enough_findings}/{nc} (need ≥{nc - 2})")
    print(f"  All findings cited: {all_cited}/{nc} (need ≥{nc - 2})")
    print(f"  No uncited numerics: {no_numerics}/{total} (need {total})")
    print()

    # Failures
    failures = [r for r in results if not r["overall_pass"]]
    if failures:
        print("FAILURES:")
        for r in failures:
            reason = r.get("error") or (
                f"findings={r.get('findings_count', '?')}, "
                f"all_cited={r.get('all_cited', '?')}, "
                f"no_bad_numerics={r.get('no_bad_numerics', '?')}"
            )
            print(f"  {r['fixture_id']}: {reason}")


def _check_thresholds(results: list[dict[str, Any]]) -> bool:
    """Return True if §14.2 thresholds are met."""
    non_contrarian = [r for r in results if not r.get("is_contrarian")]
    nc = len(non_contrarian)
    enough_findings = sum(1 for r in non_contrarian if r.get("has_enough_findings"))
    all_cited = sum(1 for r in non_contrarian if r.get("all_cited"))
    no_numerics = all(r.get("no_bad_numerics", True) for r in results)
    return enough_findings >= nc - 2 and all_cited >= nc - 2 and no_numerics


def _write_report(results: list[dict[str, Any]], depth: str) -> Path:
    """Write reports/researcher_baseline.md."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    report_path = _REPORTS_DIR / "researcher_baseline.md"

    passed = sum(1 for r in results if r["overall_pass"])
    total = len(results)
    thresholds_met = _check_thresholds(results)

    lines = [
        "# Researcher Quality Baseline",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Depth:** {depth}  ",
        f"**Overall:** {passed}/{total} PASS  ",
        f"**§14.2 thresholds met:** {'YES' if thresholds_met else 'NO'}  ",
        "",
        "## Per-fixture results",
        "",
        "| ID | Q (truncated) | Findings | All cited | No bad numerics | PASS | Latency |",
        "|----|--------------|----------|-----------|-----------------|------|---------|",
    ]
    for r in results:
        icon = "✅" if r["overall_pass"] else "❌"
        cited = "✅" if r.get("all_cited", True) else "❌"
        numerics = "✅" if r.get("no_bad_numerics", True) else "❌"
        findings = r.get("findings_count", "ERR")
        latency = f"{r.get('latency_s', '?')}s"
        q = r["question"][:45].replace("|", "/")
        lines.append(
            f"| {r['fixture_id']} | {q}... | {findings} | {cited} | {numerics} | {icon} | {latency} |"
        )

    lines += [
        "",
        "## Threshold check (§14.2)",
        "",
    ]
    non_contrarian = [r for r in results if not r.get("is_contrarian")]
    nc = len(non_contrarian)
    enough = sum(1 for r in non_contrarian if r.get("has_enough_findings"))
    cited = sum(1 for r in non_contrarian if r.get("all_cited"))
    numerics = sum(1 for r in results if r.get("no_bad_numerics", True))
    lines += [
        f"- ≥2 findings: {enough}/{nc} (threshold: ≥{nc - 2}) {'✅' if enough >= nc - 2 else '❌'}",
        f"- All findings cited: {cited}/{nc} (threshold: ≥{nc - 2}) {'✅' if cited >= nc - 2 else '❌'}",
        f"- No uncited numerics: {numerics}/{len(results)} (threshold: {len(results)}) {'✅' if numerics == len(results) else '❌'}",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Researcher quality battery.")
    parser.add_argument(
        "--depth",
        choices=["shallow", "deep"],
        default="shallow",
        help="Retrieval depth for all fixtures (default: shallow).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write reports/researcher_baseline.md.",
    )
    args = parser.parse_args()

    # Validate required env vars
    missing = [k for k in ("TAVILY_API_KEY", "HUGGINGFACE_API_TOKEN") if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    results = asyncio.run(run_eval(depth=args.depth))
    _print_summary(results)

    if args.report:
        path = _write_report(results, args.depth)
        print(f"\nReport written: {path}")

    if not _check_thresholds(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
