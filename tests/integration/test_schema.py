"""Integration test for Task 2.

Spins up the migrations + seed script against the configured Postgres and
verifies the round-trip: user row exists, all four structured facts are stored
with the expected shapes, and pgvector is installed.
"""

from __future__ import annotations

import asyncpg
import pytest
from scripts.seed_test_user import TEST_FACTS, TEST_USER_NAME, seed

pytestmark = pytest.mark.asyncio


async def test_pgvector_extension_installed(db_conn: asyncpg.Connection) -> None:
    installed = await db_conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    assert installed == 1, "pgvector extension should be installed by the migration"


async def test_all_tables_exist(db_conn: asyncpg.Connection) -> None:
    rows = await db_conn.fetch(
        """
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
        """
    )
    tables = {r["tablename"] for r in rows}
    expected = {
        "users",
        "structured_facts",
        "sessions",
        "turns",
        "turn_embeddings",
        "hypotheses",
        "hypothesis_evidence",
        "agent_calls",
        "schema_migrations",
    }
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}"


async def test_seed_round_trip(migrated_db: str, db_conn: asyncpg.Connection) -> None:
    user_id = await seed(migrated_db)
    assert user_id is not None

    row = await db_conn.fetchrow("SELECT display_name FROM users WHERE user_id = $1", user_id)
    assert row is not None
    assert row["display_name"] == TEST_USER_NAME

    facts = await db_conn.fetch(
        "SELECT key, value, source FROM structured_facts WHERE user_id = $1 ORDER BY key",
        user_id,
    )
    stored = {r["key"]: r["value"] for r in facts}
    # JSONB decoded via the codec registered in db.py.
    assert stored == TEST_FACTS
    assert {r["source"] for r in facts} == {"user_stated"}
