"""Plain-SQL migration runner.

Applies every ``NNN_*.sql`` file under ``migrations/`` in lexical order inside a
single transaction. Tracks applied versions in a ``schema_migrations`` table so
reruns are idempotent.

Usage::

    uv run python scripts/run_migrations.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import asyncpg

from career_coach.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def apply_migrations(dsn: str) -> list[str]:
    """Apply pending migrations. Returns the versions newly applied."""
    conn = await asyncpg.connect(dsn=dsn)
    applied: list[str] = []
    try:
        await conn.execute(SCHEMA_MIGRATIONS_DDL)
        already: set[str] = {
            r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")
        }

        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not files:
            print(f"No migrations found under {MIGRATIONS_DIR}", file=sys.stderr)
            return applied

        for path in files:
            version = path.stem
            if version in already:
                print(f"  skip  {version} (already applied)")
                continue
            sql = path.read_text()
            print(f"  apply {version} ({len(sql)} bytes)")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)",
                    version,
                )
            applied.append(version)
    finally:
        await conn.close()
    return applied


async def _main() -> int:
    settings = get_settings()
    print(f"Applying migrations to {settings.supabase_db_url}")
    applied = await apply_migrations(settings.supabase_db_url)
    if applied:
        print(f"Applied: {', '.join(applied)}")
    else:
        print("Nothing to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
