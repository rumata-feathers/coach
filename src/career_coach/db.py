"""Async Postgres connection pool.

Every repo and script must acquire connections from :func:`get_pool`; no module
should create its own pool. A single process-wide pool keeps pgvector codec
registration and connection setup in one place.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from career_coach.config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup.

    Registers JSON/JSONB codecs so ``dict``/``list`` round-trip without the
    caller having to ``json.dumps`` at every call site. pgvector's codec is
    installed lazily via :func:`pgvector.asyncpg.register_vector` so the import
    cost is only paid once pgvector is actually used by a repo.
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    # Register the pgvector codec so list[float] round-trips to/from `vector`.
    # The extension must already be installed (migration 001 does so).
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide asyncpg pool, creating it on first call."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            dsn=settings.supabase_db_url,
            min_size=1,
            max_size=10,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    """Close the pool. Call from application shutdown hooks and test teardown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def fetchval(sql: str, *args: Any) -> Any:
    """Convenience: run ``fetchval`` against a pooled connection."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(sql, *args)


async def fetch(sql: str, *args: Any) -> list[asyncpg.Record]:
    """Convenience: run ``fetch`` against a pooled connection."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows: list[asyncpg.Record] = await conn.fetch(sql, *args)
        return rows


async def execute(sql: str, *args: Any) -> str:
    """Convenience: run ``execute`` against a pooled connection."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        status: str = await conn.execute(sql, *args)
        return status
