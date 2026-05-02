"""Print formatted conversation transcripts from the DB.

Usage
-----
List recent sessions (last 20):
    uv run python scripts/transcript.py

Show all turns in a session:
    uv run python scripts/transcript.py <session_id>

Show with full assistant text (default truncates to 300 chars):
    uv run python scripts/transcript.py <session_id> --full

Show which flow each turn used (default: shown in header):
    uv run python scripts/transcript.py <session_id>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from career_coach.db import close_pool, get_pool  # noqa: E402


async def list_sessions() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.session_id, u.display_name, s.created_at,
                   COUNT(t.turn_id)::int AS turns
            FROM sessions s
            JOIN users u ON u.user_id = s.user_id
            LEFT JOIN turns t ON t.session_id = s.session_id
            GROUP BY s.session_id, u.display_name, s.created_at
            ORDER BY s.created_at DESC
            LIMIT 20
            """,
        )
    if not rows:
        print("No sessions found.")
        return
    print(f"{'session_id':<38}  {'user':<20}  {'created':<20}  turns")
    print("-" * 85)
    for r in rows:
        created = str(r["created_at"])[:19]
        print(f"{str(r['session_id']):<38}  {r['display_name']:<20}  {created:<20}  {r['turns']}")


async def show_session(session_id: str, full: bool = False) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        header = await conn.fetchrow(
            """
            SELECT u.display_name, s.created_at, s.session_theory
            FROM sessions s JOIN users u ON u.user_id = s.user_id
            WHERE s.session_id = $1::uuid
            """,
            session_id,
        )
        if header is None:
            print(f"Session {session_id!r} not found.")
            return

        turns = await conn.fetch(
            """
            SELECT turn_index, flow_used, user_message, assistant_message,
                   deployment_version, created_at
            FROM turns
            WHERE session_id = $1::uuid
            ORDER BY turn_index
            """,
            session_id,
        )

    print(f"\nSession : {session_id}")
    print(f"User    : {header['display_name']}")
    print(f"Created : {str(header['created_at'])[:19]}")
    if header["session_theory"]:
        print(f"Theory  : {header['session_theory']}")
    print(f"Turns   : {len(turns)}")
    print("=" * 80)

    for t in turns:
        flow = t["flow_used"] or "?"
        ts = str(t["created_at"])[:19]
        print(f"\n[Turn {t['turn_index'] + 1}  flow={flow}  {ts}]")
        print(f"  USER : {t['user_message']}")
        coach_text: str = t["assistant_message"] or ""
        if not full and len(coach_text) > 300:
            coach_text = coach_text[:300] + "…"
        print(f"  COACH: {coach_text}")

    print("\n" + "=" * 80)


async def _main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv

    try:
        if not args:
            await list_sessions()
        else:
            await show_session(args[0], full=full)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
