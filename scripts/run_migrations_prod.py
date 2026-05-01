"""Production migration runner — Railway pre-deploy hook.

Applies any pending SQL migrations from ``migrations/`` to the database
pointed to by ``SUPABASE_DB_URL``. Designed to run as a Railway
``releaseCommand`` (inside the deployed image, before traffic cuts over).

Reuses :func:`~scripts.run_migrations.apply_migrations` for the core logic
so there is one canonical migration path for both dev and prod.

Exit codes:
  0  — all migrations applied (or nothing to apply)
  1  — configuration error (SUPABASE_DB_URL missing / looks like the
       local docker-compose default)
  2  — migration failure (DB unreachable, SQL error, etc.)

Usage inside Railway release command::

    uv run python scripts/run_migrations_prod.py

Or locally to dry-run against a remote DB::

    SUPABASE_DB_URL=postgresql://... uv run python scripts/run_migrations_prod.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make sure the src package tree is importable whether running via uv or plain python.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from career_coach.config import get_settings  # noqa: E402

# Reuse the apply_migrations function from the dev runner.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from run_migrations import apply_migrations  # noqa: E402

_LOCAL_DEFAULT = "postgresql://coach:coach@localhost:5432/coach"


async def _main() -> int:
    settings = get_settings()
    dsn = settings.supabase_db_url

    # ── Guard: reject the local docker-compose default in production ──────────
    # If someone forgets to set SUPABASE_DB_URL on Railway, it would fall back
    # to the default which points at localhost (unreachable in production).
    if dsn == _LOCAL_DEFAULT and not os.getenv("ALLOW_LOCAL_DB"):
        print(
            "ERROR: SUPABASE_DB_URL is not set (or is still the docker-compose "
            "default). Set it to your Supabase pooler URL before deploying.\n"
            "To run against the local DB intentionally, set ALLOW_LOCAL_DB=1.",
            file=sys.stderr,
        )
        return 1

    # Mask credentials in logs — show host/db but not user:password.
    try:
        from urllib.parse import urlparse
        parsed = urlparse(dsn)
        safe_url = f"{parsed.scheme}://***@{parsed.hostname}{parsed.path}"
    except Exception:
        safe_url = "<unparseable DSN>"

    print(f"[migrations] target: {safe_url}")

    try:
        applied = await apply_migrations(dsn)
    except Exception as exc:
        print(f"[migrations] FAILED: {exc}", file=sys.stderr)
        return 2

    if applied:
        print(f"[migrations] applied: {', '.join(applied)}")
    else:
        print("[migrations] nothing to apply — schema is up to date")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
