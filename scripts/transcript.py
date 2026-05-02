"""Print formatted conversation transcripts from the DB.

Usage
-----
List recent sessions (last 20):
    uv run python scripts/transcript.py --dsn "postgresql://..."

Show all turns in a session:
    uv run python scripts/transcript.py <session_id> --dsn "postgresql://..."

Show with full assistant text (default truncates to 300 chars):
    uv run python scripts/transcript.py <session_id> --dsn "..." --full

If --dsn is omitted the script falls back to the SUPABASE_DB_URL env var
(or whatever is in your .env file).

Tip — paste your Supabase session-pooler URL:
    postgresql://postgres.<project>:[password]@aws-X.pooler.supabase.com:5432/postgres
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _parse_args() -> tuple[str | None, str | None, bool]:
    """Return (session_id, dsn_override, full)."""
    argv = sys.argv[1:]
    session_id: str | None = None
    dsn: str | None = None
    full = False

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--full":
            full = True
        elif a == "--dsn":
            i += 1
            dsn = argv[i] if i < len(argv) else None
        elif a.startswith("--dsn="):
            dsn = a[6:]
        elif not a.startswith("--"):
            session_id = a
        i += 1

    return session_id, dsn, full


async def list_sessions(pool: object) -> None:
    async with pool.acquire() as conn:  # type: ignore[union-attr]
        rows = await conn.fetch(
            """
            SELECT s.session_id, u.display_name, s.started_at,
                   COUNT(t.turn_id)::int AS turns
            FROM sessions s
            JOIN users u ON u.user_id = s.user_id
            LEFT JOIN turns t ON t.session_id = s.session_id
            GROUP BY s.session_id, u.display_name, s.started_at
            ORDER BY s.started_at DESC
            LIMIT 20
            """,
        )
    if not rows:
        print("No sessions found.")
        return
    print(f"{'session_id':<38}  {'user':<20}  {'created':<20}  turns")
    print("-" * 85)
    for r in rows:
        created = str(r["started_at"])[:19]
        print(f"{str(r['session_id']):<38}  {r['display_name']:<20}  {created:<20}  {r['turns']}")


async def show_session(pool: object, session_id: str, full: bool = False) -> None:
    async with pool.acquire() as conn:  # type: ignore[union-attr]
        header = await conn.fetchrow(
            """
            SELECT u.display_name, s.started_at, s.session_theory
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
    print(f"Created : {str(header['started_at'])[:19]}")
    if header["session_theory"]:
        print(f"Theory  : {header['session_theory']}")
    print(f"Turns   : {len(turns)}")
    print("=" * 80)

    for t in turns:
        flow = t["flow_used"] or "?"
        ts = str(t["created_at"])[:19]
        ver = t["deployment_version"] or ""
        print(f"\n[Turn {t['turn_index'] + 1}  flow={flow}  {ts}  {ver}]")
        print(f"  USER : {t['user_message']}")
        coach_text: str = t["assistant_message"] or ""
        if not full and len(coach_text) > 300:
            coach_text = coach_text[:300] + "…"
        print(f"  COACH: {coach_text}")

    print("\n" + "=" * 80)


async def _main() -> None:
    import asyncpg

    session_id, dsn_override, full = _parse_args()

    # DSN resolution: CLI flag > env var > settings
    if dsn_override:
        dsn = dsn_override
    else:
        # Try to load from settings (reads .env)
        try:
            from career_coach.config import get_settings
            dsn = get_settings().supabase_db_url
        except Exception:
            dsn = os.environ.get("SUPABASE_DB_URL", "")

    if not dsn:
        print("ERROR: no DSN. Pass --dsn 'postgresql://...' or set SUPABASE_DB_URL.", file=sys.stderr)
        sys.exit(1)

    # Detect remote vs local to set SSL
    local_hints = ("localhost", "127.0.0.1", "host.docker.internal")
    ssl = None if any(h in dsn for h in local_hints) else "require"

    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1, max_size=3,
        ssl=ssl,
        statement_cache_size=0 if ssl else 100,
        timeout=20.0,
        command_timeout=30.0,
    )
    try:
        if not session_id:
            await list_sessions(pool)
        else:
            await show_session(pool, session_id, full=full)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
