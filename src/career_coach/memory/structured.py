"""Structured facts repository.

Tier 1 of the three-tier memory model. Stores keyed attributes about the
user (age, location, education stage, …) that are stable across sessions.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from career_coach.db import get_pool
from career_coach.models.user_model import FactUpdate


class StructuredFactsRepo:
    """CRUD for the ``structured_facts`` table."""

    async def get_all(self, user_id: UUID) -> dict[str, Any]:
        """Return all facts for ``user_id`` as a ``{key: value}`` dict.

        Agents pass this dict straight into prompts, so the representation is
        deliberately ergonomic rather than mirroring full rows.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM structured_facts WHERE user_id = $1 ORDER BY key",
                user_id,
            )
        return {row["key"]: row["value"] for row in rows}

    async def upsert(self, user_id: UUID, fact: FactUpdate) -> None:
        """Insert or update a single fact.

        Uses the ``(user_id, key)`` unique index to decide; on conflict the
        value/confidence/source are replaced and ``updated_at`` bumped.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO structured_facts (user_id, key, value, confidence, source)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                ON CONFLICT (user_id, key) DO UPDATE
                    SET value      = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        source     = EXCLUDED.source,
                        updated_at = now()
                """,
                user_id,
                fact.key,
                fact.value,
                fact.confidence,
                fact.source,
            )

    async def bulk_upsert(self, user_id: UUID, facts: list[FactUpdate]) -> None:
        """Upsert many facts atomically in a single transaction."""
        if not facts:
            return
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            for fact in facts:
                await conn.execute(
                    """
                    INSERT INTO structured_facts (user_id, key, value, confidence, source)
                    VALUES ($1, $2, $3::jsonb, $4, $5)
                    ON CONFLICT (user_id, key) DO UPDATE
                        SET value      = EXCLUDED.value,
                            confidence = EXCLUDED.confidence,
                            source     = EXCLUDED.source,
                            updated_at = now()
                    """,
                    user_id,
                    fact.key,
                    fact.value,
                    fact.confidence,
                    fact.source,
                )
