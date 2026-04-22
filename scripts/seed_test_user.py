"""Seed a single test user with structured facts for v0 development.

Creates (idempotently by display name) one user and populates a handful of
structured facts used by agent prompts.

Usage::

    uv run python scripts/seed_test_user.py
"""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

import asyncpg

from career_coach.config import get_settings

TEST_USER_NAME = "test_student"

TEST_FACTS: dict[str, object] = {
    "name": "Alex",
    "age": 19,
    "location": "London",
    "education_stage": "undergrad",
}


async def seed(dsn: str) -> UUID:
    """Insert or reuse the test user and upsert the structured facts."""
    conn = await asyncpg.connect(dsn=dsn)
    try:
        async with conn.transaction():
            existing = await conn.fetchval(
                "SELECT user_id FROM users WHERE display_name = $1",
                TEST_USER_NAME,
            )
            if existing is None:
                user_id = await conn.fetchval(
                    "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
                    TEST_USER_NAME,
                )
            else:
                user_id = existing

            for key, value in TEST_FACTS.items():
                await conn.execute(
                    """
                    INSERT INTO structured_facts (user_id, key, value, source)
                    VALUES ($1, $2, $3::jsonb, 'user_stated')
                    ON CONFLICT (user_id, key) DO UPDATE
                        SET value = EXCLUDED.value,
                            updated_at = now()
                    """,
                    user_id,
                    key,
                    json.dumps(value),
                )
        return user_id
    finally:
        await conn.close()


async def _main() -> int:
    settings = get_settings()
    user_id = await seed(settings.supabase_db_url)
    print(f"Seeded test user {user_id} with facts: {sorted(TEST_FACTS.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
