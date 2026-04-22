"""Value objects that describe *the user* across the three memory tiers.

These are the primitives that flow between agents and repositories:
``FactUpdate`` for structured facts, ``TurnSummary`` for episodic context,
``Hypothesis`` + ``EvidenceDraft`` for semantic reasoning, ``Challenge`` for
the Coach's proposed next step.

The contracts match the DB schema in ``migrations/001_initial.sql`` but are
deliberately *thinner* — optional DB-only columns (timestamps,
``last_updated``) are allowed to be ``None`` when agents are constructing
these objects before persistence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

FactSource = Literal["user_stated", "inferred", "system"]
HypothesisStatus = Literal["active", "dormant", "retired", "superseded"]
EvidenceSource = Literal[
    "user_statement",
    "reflection",
    "challenge_outcome",
    "forced_choice",
]


class FactUpdate(BaseModel):
    """A single structured-fact mutation produced by the Profiler.

    ``value`` is any JSON-serialisable payload to match the JSONB column.
    """

    key: str = Field(..., min_length=1, description="Fact key, e.g. 'age'.")
    value: Any = Field(..., description="JSON-serialisable fact value.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: FactSource = "inferred"


class TurnSummary(BaseModel):
    """Compact representation of a prior turn for context windows.

    Agents usually only need role/content snippets plus ``turn_index`` for
    ordering; full DB rows are too large to fit in prompts.
    """

    turn_id: UUID
    turn_index: int = Field(..., ge=0)
    user_message: str | None = None
    assistant_message: str | None = None
    intent: str | None = None
    created_at: datetime | None = None


class Hypothesis(BaseModel):
    """A semantic hypothesis about the user."""

    hypothesis_id: UUID
    statement: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: HypothesisStatus = "active"
    open_questions: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    last_updated: datetime | None = None
    last_reviewed: datetime | None = None


class EvidenceDraft(BaseModel):
    """An evidence row queued by the Profiler for the distillation job.

    The distillation job decides which hypothesis this belongs to (or creates
    a new one); at Profiler-time we only know the signal, not its target.
    """

    source_type: EvidenceSource
    excerpt: str
    weight: float = Field(..., description="Positive supports, negative contradicts.")
    hypothesis_hint: str | None = Field(
        default=None,
        description="Free-form hint the distiller may use to match or seed a hypothesis.",
    )

    @field_validator("weight")
    @classmethod
    def _nonzero_weight(cls, value: float) -> float:
        if value == 0.0:
            raise ValueError("Evidence weight must be non-zero.")
        return value


class Challenge(BaseModel):
    """Coach-proposed next step the user could take between sessions."""

    title: str
    description: str
    estimated_effort: Literal["quick", "medium", "deep"] = "medium"
    ties_to: list[str] = Field(
        default_factory=list,
        description="Free-form keys or hypothesis ids this challenge probes.",
    )
