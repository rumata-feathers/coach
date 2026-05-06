#!/usr/bin/env python3
"""Back-fill is_test = TRUE for legacy test users created before migration 007.

Dry-run by default — prints the matched rows without changing anything.
Pass --apply to write the UPDATE.

Usage
-----
    uv run python scripts/backfill_test_users.py
    uv run python scripts/backfill_test_users.py --apply
    uv run python scripts/backfill_test_users.py --dsn "postgresql://..."
    uv run python scripts/backfill_test_users.py --apply --dsn "postgresql://..."
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Known test-user naming patterns
# ---------------------------------------------------------------------------
#
# Prefixes: any display_name starting with one of these is a test user.
# Kept in sync with every INSERT in tests/ and scripts/*.

_TEST_PREFIXES: list[str] = [
    # integration tests
    "chat_test_",
    "session_test_",
    "distill_",
    "distil_",       # alternate spelling in distillation eval scripts
    "dedup_",
    "memloop_",
    "sup-block-",
    "sup-onboard-",
    "integration_",
    "obs_test_",
    "continuity_",
    "test_",         # test_memory.py: "test_<hex8>"
    "eval-flowc-",
    "demo_",
    # flow-c end-to-end
    "test-flow-c",
]

# Exact literal names that don't match the prefix patterns above.
_TEST_EXACT: list[str] = [
    "test_student",
    "Test Student",
    "migration-004-test-user",
    "migration-004-supervisor-test",
    "migration-004-chart-test",
]


def _build_where() -> str:
    """Return the WHERE clause identifying test users."""
    prefix_checks = "\n        OR ".join(
        f"display_name LIKE '{p}%'" for p in _TEST_PREFIXES
    )
    exact_list = ", ".join(f"'{e}'" for e in _TEST_EXACT)
    return (
        f"(\n        {prefix_checks}\n"
        f"        OR display_name IN ({exact_list})\n"
        "    )"
    )


async def _run(dsn: str, apply: bool) -> None:
    import asyncpg  # type: ignore[import]

    needs_ssl = not any(h in dsn for h in ("localhost", "127.0.0.1", "host.docker.internal"))
    conn = await asyncpg.connect(dsn=dsn, ssl="require" if needs_ssl else None)

    where = _build_where()

    try:
        rows = await conn.fetch(
            f"""
            SELECT user_id, display_name, created_at
            FROM users
            WHERE NOT is_test
              AND {where}
            ORDER BY created_at
            """
        )

        if not rows:
            print("No legacy test users found — nothing to do.")
            return

        print(f"Found {len(rows)} legacy test user(s) to mark is_test = TRUE:\n")
        for r in rows:
            print(f"  {str(r['user_id'])[:8]}…  {r['display_name']!r:40s}  created {str(r['created_at'])[:16]}")

        if not apply:
            print(
                f"\nDry-run — no changes made.  "
                f"Re-run with --apply to update {len(rows)} row(s)."
            )
            return

        result = await conn.execute(
            f"""
            UPDATE users
            SET is_test = TRUE
            WHERE NOT is_test
              AND {where}
            """
        )
        # result is e.g. "UPDATE 12"
        print(f"\nDone: {result}")
    finally:
        await conn.close()


def _parse_args() -> tuple[str | None, bool]:
    argv = sys.argv[1:]
    dsn: str | None = None
    apply = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--apply":
            apply = True
        elif a in ("--dsn", "-d"):
            i += 1
            dsn = argv[i] if i < len(argv) else None
        elif a.startswith("--dsn="):
            dsn = a[len("--dsn="):]
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        i += 1
    return dsn, apply


def _resolve_dsn(override: str | None) -> str:
    if override:
        return override
    try:
        from career_coach.config import get_settings
        return get_settings().supabase_db_url
    except Exception:
        import os
        return os.environ.get("SUPABASE_DB_URL", "")


async def _main() -> None:
    dsn_override, apply = _parse_args()
    dsn = _resolve_dsn(dsn_override)
    if not dsn:
        print("ERROR: no DSN. Pass --dsn or set SUPABASE_DB_URL.", file=sys.stderr)
        sys.exit(1)
    await _run(dsn, apply)


if __name__ == "__main__":
    asyncio.run(_main())
