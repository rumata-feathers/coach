#!/usr/bin/env python
"""Run a scripted demo conversation through the full TurnPipeline.

Usage examples::

    # 5-turn demo with auto-generated display name
    uv run python scripts/demo_conversation.py

    # Custom user name and message script
    uv run python scripts/demo_conversation.py \\
        --display-name "Alex" \\
        --script scripts/demo_messages.txt

    # Two-session mode: run the script twice, preserving user_id but starting a
    # fresh session on the second pass (tests cross-session memory)
    uv run python scripts/demo_conversation.py \\
        --display-name "Alex" \\
        --two-session-mode

Output:
    Prints a live transcript to stdout and writes a Markdown report to
    ``reports/demo_<timestamp>.md``.

Requires:
    HUGGINGFACE_API_TOKEN set in ``.env`` (or the environment).
    A running Postgres matching DATABASE_URL in ``.env``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

# Make sure src/ is on the path whether the script is run via uv or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_coach.config import get_settings
from career_coach.db import close_pool, get_pool
from career_coach.jobs.distillation import run_distillation
from career_coach.llm.factory import LLMFactory
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo
from career_coach.pipeline.turn import TurnPipeline

# Default message script used when --script is not supplied.
_DEFAULT_MESSAGES = [
    "Should I study economics?",
    "I'm 19, studying in London, and I find maths pretty easy.",
    "What kinds of careers would actually use an economics degree in the UK?",
    "I'm also considering computer science — how do I choose?",
    "What do you think is the one question I should be asking myself right now?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_user(display_name: str) -> UUID:
    """Insert a fresh user row and return the new user_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id: UUID = await conn.fetchval(
            "INSERT INTO users (display_name, is_test) VALUES ($1, TRUE) RETURNING user_id",
            display_name,
        )
    return user_id


def _load_messages(script_path: Path | None) -> list[str]:
    if script_path is None:
        return list(_DEFAULT_MESSAGES)
    lines = [ln.strip() for ln in script_path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Script file {script_path} is empty.")
    return lines


def _now_str() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


async def _print_final_state(user_id: UUID) -> dict[str, object]:
    """Return and pretty-print the user's final memory state."""
    facts = await StructuredFactsRepo().get_all(user_id)
    hypotheses = await SemanticRepo().get_active(user_id)

    print("\n─── Final memory state ───────────────────────────────────────")
    if facts:
        print("Facts:")
        for k, v in facts.items():
            print(f"  {k}: {v!r}")
    else:
        print("Facts: (none)")

    if hypotheses:
        print("Hypotheses:")
        for h in hypotheses:
            print(f"  [{h.confidence:.2f}] {h.statement}")
    else:
        print("Hypotheses: (none)")
    print("──────────────────────────────────────────────────────────────\n")

    return {"facts": facts, "hypotheses": [h.statement for h in hypotheses]}


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


async def _run_session(
    pipeline: TurnPipeline,
    user_id: UUID,
    messages: list[str],
    *,
    session_id: UUID | None,
    session_label: str = "Session",
) -> tuple[UUID, list[dict[str, object]]]:
    """Run all messages through the pipeline and return (session_id, transcript)."""
    transcript: list[dict[str, object]] = []

    for i, msg in enumerate(messages):
        print(f"\n[{session_label} | turn {i}]")
        print(f"User : {msg}")
        result = await pipeline.process_turn(
            user_id=user_id,
            user_message=msg,
            session_id=session_id,
        )
        session_id = result.session_id
        print(f"Coach: {result.response}")

        transcript.append(
            {
                "session_id": str(session_id),
                "turn": i,
                "user": msg,
                "coach": result.response,
                "clarification_only": result.clarification_only,
            }
        )

        # Give the background Profiler task a moment to write to DB.
        await asyncio.sleep(0.1)

    # session_id is set after the first turn; messages list is non-empty (validated by caller)
    assert session_id is not None, "No turns were processed"
    return session_id, transcript


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _render_report(
    display_name: str,
    user_id: UUID,
    sessions: list[tuple[str, list[dict[str, object]]]],
    distillation_summary: dict[str, object],
    final_state: dict[str, object],
    elapsed_s: float,
) -> str:
    lines = [
        f"# Demo Conversation — {display_name}",
        f"\nGenerated: {_now_str()}  |  user_id: `{user_id}`  |  elapsed: {elapsed_s:.1f}s",
        "\n---\n",
    ]

    for label, transcript in sessions:
        lines.append(f"## {label}\n")
        for turn in transcript:
            sid = turn["session_id"]
            lines.append(f"**Turn {turn['turn']}** (session `{sid[:8]}…`)")
            lines.append(f"> **User:** {turn['user']}")
            marker = " _(clarification)_" if turn["clarification_only"] else ""
            lines.append(f"> **Coach:** {turn['coach']}{marker}")
            lines.append("")

    lines += [
        "---\n",
        "## Distillation summary\n",
        f"- Processed: {distillation_summary.get('processed', '?')}",
        f"- Created:   {distillation_summary.get('created', '?')}",
        f"- Matched:   {distillation_summary.get('matched', '?')}",
        "\n## Final memory state\n",
        "### Facts\n",
    ]
    facts = final_state.get("facts", {})
    if facts:
        for k, v in facts.items():
            lines.append(f"- `{k}`: {v!r}")
    else:
        lines.append("_(none)_")

    lines += ["\n### Hypotheses\n"]
    hypotheses = final_state.get("hypotheses", [])
    if hypotheses:
        for h in hypotheses:
            lines.append(f"- {h}")
    else:
        lines.append("_(none)_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.huggingface_api_token:
        print(
            "ERROR: HUGGINGFACE_API_TOKEN not set. Add it to .env and retry.",
            file=sys.stderr,
        )
        return 1

    messages = _load_messages(args.script)
    display_name: str = args.display_name or f"demo_{datetime.now(tz=UTC).strftime('%H%M%S')}"

    print(f"Creating user '{display_name}'…")
    user_id = await _create_user(display_name)
    print(f"user_id = {user_id}\n")

    factory = LLMFactory()
    pipeline = TurnPipeline(factory)

    t0 = asyncio.get_event_loop().time()
    sessions_transcripts: list[tuple[str, list[dict[str, object]]]] = []

    # First (or only) session
    _session_id, transcript = await _run_session(
        pipeline,
        user_id,
        messages,
        session_id=None,
        session_label="Session 1",
    )
    sessions_transcripts.append(("Session 1", transcript))

    # Optional second session (fresh session, same user — tests cross-session memory)
    if args.two_session_mode:
        print("\n\n══ Starting Session 2 (new session, same user) ══\n")
        _, transcript2 = await _run_session(
            pipeline,
            user_id,
            messages[:3],  # use first 3 messages as a shorter follow-up
            session_id=None,
            session_label="Session 2",
        )
        sessions_transcripts.append(("Session 2", transcript2))

    # Run distillation
    print("\nRunning distillation…")
    distil_summary = await run_distillation(user_id, factory=factory)
    print(f"Distillation: {distil_summary}")

    # Print and capture final memory state
    final_state = await _print_final_state(user_id)

    elapsed = asyncio.get_event_loop().time() - t0

    # Write report
    report = _render_report(
        display_name, user_id, sessions_transcripts, distil_summary, final_state, elapsed
    )
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).resolve().parents[1] / "reports" / f"demo_{ts}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Transcript written to {report_path}")

    await close_pool()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a scripted demo conversation through the career coach pipeline."
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="User display name (default: demo_<timestamp>).",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Path to a plain-text file with one message per line.",
    )
    parser.add_argument(
        "--two-session-mode",
        action="store_true",
        help="Run the script twice with a fresh session on the second pass.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(_parse_args())))
