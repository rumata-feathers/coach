"""Shared pytest fixtures for the test suite.

Integration tests need a live Postgres with the schema applied. We connect
using the DSN in ``SUPABASE_DB_URL`` (defaults to the local docker-compose
instance) and skip the integration layer cleanly when no database is reachable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from scripts.run_migrations import apply_migrations

from career_coach.config import get_settings
from career_coach.db import close_pool, get_pool

# Load .env into os.environ so that os.getenv() skip-guards in tests work the
# same whether the token was exported in the shell or only set in .env.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


async def _postgres_reachable(dsn: str) -> bool:
    try:
        conn = await asyncio.wait_for(asyncpg.connect(dsn=dsn), timeout=2.0)
    except (TimeoutError, OSError, asyncpg.PostgresError):
        return False
    await conn.close()
    return True


@pytest_asyncio.fixture(scope="session")
async def migrated_db() -> AsyncIterator[str]:
    """Ensure the test DB exists, is reachable, and has migrations applied.

    Skips the entire test if Postgres is unreachable — CI without a DB still
    runs the non-integration tests.
    """
    settings = get_settings()
    dsn = settings.supabase_db_url
    if not await _postgres_reachable(dsn):
        pytest.skip(f"Postgres at {dsn} not reachable; skipping integration test")
    await apply_migrations(dsn)
    yield dsn
    await close_pool()


@pytest_asyncio.fixture()
async def db_conn(migrated_db: str) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pooled connection for a single test."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn
