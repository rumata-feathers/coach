"""Episodic memory repository.

Tier 2 of the three-tier memory model. Owns sessions, turns, and pgvector
embeddings for similarity search over prior turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from career_coach.db import get_pool
from career_coach.models.user_model import TurnSummary


@dataclass(slots=True)
class Session:
    """Lightweight session row."""

    session_id: UUID
    user_id: UUID
    session_theory: str | None


@dataclass(slots=True)
class SimilarTurn:
    """A search hit for :meth:`EpisodicRepo.search_similar`."""

    turn: TurnSummary
    distance: float


class EpisodicRepo:
    """CRUD for ``sessions``, ``turns``, and ``turn_embeddings``."""

    async def create_session(self, user_id: UUID, session_theory: str | None = None) -> Session:
        """Start a new session and return it."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO sessions (user_id, session_theory)
                VALUES ($1, $2)
                RETURNING session_id, user_id, session_theory
                """,
                user_id,
                session_theory,
            )
        assert row is not None
        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            session_theory=row["session_theory"],
        )

    async def get_session(self, session_id: UUID) -> Session | None:
        """Fetch a session by id, or ``None`` if not found."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT session_id, user_id, session_theory FROM sessions WHERE session_id = $1",
                session_id,
            )
        if row is None:
            return None
        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            session_theory=row["session_theory"],
        )

    async def update_session_theory(self, session_id: UUID, theory: str) -> None:
        """Overwrite the Understander's working theory for a session."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET session_theory = $1 WHERE session_id = $2",
                theory,
                session_id,
            )

    async def save_turn(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        user_message: str | None,
        assistant_message: str | None,
        intent_packet: dict[str, Any] | None,
        flow_used: str | None,
        critic_verdicts: list[dict[str, Any]] | None,
        tokens_used: dict[str, int] | None,
    ) -> UUID:
        """Insert a new turn. Auto-assigns the next ``turn_index`` per session.

        Returns the new ``turn_id``. Embeddings are stored separately via
        :meth:`save_turn_embedding` so the write path stays synchronous and
        cheap; the embedding API call happens off the critical path.
        """
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            next_index = await conn.fetchval(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE session_id = $1",
                session_id,
            )
            turn_id = await conn.fetchval(
                """
                INSERT INTO turns (
                    session_id, user_id, turn_index,
                    user_message, assistant_message,
                    intent_packet, flow_used, critic_verdicts, tokens_used
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::jsonb, $9::jsonb)
                RETURNING turn_id
                """,
                session_id,
                user_id,
                next_index,
                user_message,
                assistant_message,
                intent_packet,
                flow_used,
                critic_verdicts,
                tokens_used,
            )
        assert turn_id is not None
        return turn_id  # type: ignore[no-any-return]

    async def save_turn_embedding(self, turn_id: UUID, embedding: list[float]) -> None:
        """Attach a dense embedding vector to an existing turn."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO turn_embeddings (turn_id, embedding)
                VALUES ($1, $2)
                ON CONFLICT (turn_id) DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                turn_id,
                np.array(embedding, dtype=np.float32),
            )

    async def count_user_turns(self, user_id: UUID) -> int:
        """Return the total number of turns recorded for ``user_id`` across all sessions."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            count: int = await conn.fetchval(
                "SELECT COUNT(*) FROM turns WHERE user_id = $1",
                user_id,
            )
        return count

    async def get_recent(self, session_id: UUID, limit: int = 5) -> list[TurnSummary]:
        """Return the most recent turns in chronological (ascending) order."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT turn_id, turn_index, user_message, assistant_message,
                       intent_packet, created_at
                FROM turns
                WHERE session_id = $1
                ORDER BY turn_index DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )
        summaries = [
            TurnSummary(
                turn_id=row["turn_id"],
                turn_index=row["turn_index"],
                user_message=row["user_message"],
                assistant_message=row["assistant_message"],
                intent=(row["intent_packet"] or {}).get("turn_intent")
                if row["intent_packet"]
                else None,
                created_at=row["created_at"],
            )
            for row in rows
        ]
        summaries.reverse()
        return summaries

    async def search_similar(
        self,
        user_id: UUID,
        embedding: list[float],
        limit: int = 5,
    ) -> list[SimilarTurn]:
        """Return the ``limit`` most similar prior turns for ``user_id``.

        Uses pgvector's cosine distance (``<=>``).
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT t.turn_id, t.turn_index, t.user_message, t.assistant_message,
                       t.intent_packet, t.created_at,
                       (e.embedding <=> $2) AS distance
                FROM turn_embeddings e
                JOIN turns t ON t.turn_id = e.turn_id
                WHERE t.user_id = $1
                ORDER BY e.embedding <=> $2
                LIMIT $3
                """,
                user_id,
                np.array(embedding, dtype=np.float32),
                limit,
            )
        return [
            SimilarTurn(
                turn=TurnSummary(
                    turn_id=row["turn_id"],
                    turn_index=row["turn_index"],
                    user_message=row["user_message"],
                    assistant_message=row["assistant_message"],
                    intent=(row["intent_packet"] or {}).get("turn_intent")
                    if row["intent_packet"]
                    else None,
                    created_at=row["created_at"],
                ),
                distance=float(row["distance"]),
            )
            for row in rows
        ]
