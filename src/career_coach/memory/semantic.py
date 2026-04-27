"""Semantic / hypothesis repository.

Tier 3 of the three-tier memory model. Hypotheses are append-only with
weighted evidence — we never overwrite a hypothesis's confidence based on a
single signal; we accumulate evidence rows and recompute.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from career_coach.db import get_pool
from career_coach.models.user_model import (
    EvidenceDraft,
    Hypothesis,
    HypothesisStatus,
)


class SemanticRepo:
    """CRUD for ``hypotheses`` and ``hypothesis_evidence``."""

    async def get_active(
        self,
        user_id: UUID,
        limit: int = 10,
    ) -> list[Hypothesis]:
        """Return the most-recently-updated active hypotheses for ``user_id``.

        Args:
            user_id: Target user.
            limit: Maximum hypotheses to return, ordered by confidence DESC
                then last_updated DESC. Capped to prevent prompt bloat;
                distillation dedup keeps the count low in practice.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT hypothesis_id, statement, confidence, status, open_questions,
                       created_at, last_updated, last_reviewed
                FROM hypotheses
                WHERE user_id = $1 AND status = 'active'
                ORDER BY confidence DESC, last_updated DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
        return [
            Hypothesis(
                hypothesis_id=row["hypothesis_id"],
                statement=row["statement"],
                confidence=row["confidence"],
                status=row["status"],
                open_questions=row["open_questions"] or [],
                created_at=row["created_at"],
                last_updated=row["last_updated"],
                last_reviewed=row["last_reviewed"],
            )
            for row in rows
        ]

    async def create_hypothesis(
        self,
        user_id: UUID,
        statement: str,
        confidence: float,
        open_questions: list[str] | None = None,
    ) -> UUID:
        """Insert a new hypothesis and return its id."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            hypothesis_id = await conn.fetchval(
                """
                INSERT INTO hypotheses (user_id, statement, confidence, open_questions)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING hypothesis_id
                """,
                user_id,
                statement,
                confidence,
                open_questions or [],
            )
        assert hypothesis_id is not None
        return hypothesis_id  # type: ignore[no-any-return]

    async def append_evidence(
        self,
        hypothesis_id: UUID,
        evidence: EvidenceDraft,
        turn_id: UUID | None = None,
    ) -> UUID:
        """Append one evidence row to a hypothesis (append-only principle)."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            evidence_id = await conn.fetchval(
                """
                INSERT INTO hypothesis_evidence
                    (hypothesis_id, turn_id, source_type, excerpt, weight)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING evidence_id
                """,
                hypothesis_id,
                turn_id,
                evidence.source_type,
                evidence.excerpt,
                evidence.weight,
            )
        assert evidence_id is not None
        return evidence_id  # type: ignore[no-any-return]

    async def queue_evidence(
        self,
        evidence: EvidenceDraft,
        turn_id: UUID | None = None,
    ) -> UUID:
        """Insert a raw evidence row with ``hypothesis_id = NULL`` for later distillation.

        Called by the Profiler to queue signals extracted from each turn.
        The distillation job later matches these to hypotheses.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            evidence_id = await conn.fetchval(
                """
                INSERT INTO hypothesis_evidence
                    (hypothesis_id, turn_id, source_type, excerpt, weight, hypothesis_hint)
                VALUES (NULL, $1, $2, $3, $4, $5)
                RETURNING evidence_id
                """,
                turn_id,
                evidence.source_type,
                evidence.excerpt,
                evidence.weight,
                evidence.hypothesis_hint,
            )
        assert evidence_id is not None
        return evidence_id  # type: ignore[no-any-return]

    async def get_unmatched_evidence(self, user_id: UUID) -> list[dict[str, Any]]:
        """Return all evidence rows with ``hypothesis_id IS NULL`` for a user.

        Joins through ``turns`` to filter by ``user_id``. Rows without a
        ``turn_id`` (e.g., manually inserted) are included via a union so
        no evidence is silently dropped.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.evidence_id, e.turn_id, e.source_type, e.excerpt,
                       e.weight, e.hypothesis_hint
                FROM hypothesis_evidence e
                JOIN turns t ON t.turn_id = e.turn_id
                WHERE e.hypothesis_id IS NULL AND t.user_id = $1
                ORDER BY e.created_at
                """,
                user_id,
            )
        return [dict(r) for r in rows]

    async def assign_evidence(
        self,
        evidence_id: UUID,
        hypothesis_id: UUID,
    ) -> None:
        """Set ``hypothesis_id`` on an evidence row after distillation matching."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE hypothesis_evidence SET hypothesis_id = $1 WHERE evidence_id = $2",
                hypothesis_id,
                evidence_id,
            )

    async def update_confidence(
        self,
        hypothesis_id: UUID,
        confidence: float,
        status: HypothesisStatus | None = None,
    ) -> None:
        """Update confidence (and optionally status) and bump ``last_updated``."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        pool = await get_pool()
        async with pool.acquire() as conn:
            if status is None:
                await conn.execute(
                    """
                    UPDATE hypotheses
                       SET confidence = $1, last_updated = now()
                     WHERE hypothesis_id = $2
                    """,
                    confidence,
                    hypothesis_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE hypotheses
                       SET confidence   = $1,
                           status       = $2,
                           last_updated = now()
                     WHERE hypothesis_id = $3
                    """,
                    confidence,
                    status,
                    hypothesis_id,
                )
