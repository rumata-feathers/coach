"""The :class:`IntentPacket` — Understander's output, Orchestrator's input.

Keeping it in its own module makes it easy to import without circular
dependencies; every other agent I/O schema references it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TurnIntent = Literal["explore", "decide", "reflect", "research", "vent"]
BudgetHint = Literal["quick", "standard", "deep"]


class IntentPacket(BaseModel):
    """Structured distillation of a user message.

    See SPEC §6.1. Two invariants encoded here:

    * ``clarity_score`` is in ``[0, 1]``.
    * Asking a clarifying question is only allowed when the intent is not
      cheap — cheap turns should short-circuit to the Coach regardless.
    """

    session_theory: str = Field(
        ..., description="Updated working theory after considering this message."
    )
    turn_intent: TurnIntent
    specific_ask: str = Field(..., description="Distilled concrete ask in one sentence.")
    emotional_tenor: str = Field(
        ..., description="One short phrase describing affect (e.g. 'anxious', 'curious')."
    )
    clarity_score: float = Field(..., ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: str | None = None
    inferred_constraints: list[str] = Field(default_factory=list)
    budget_hint: BudgetHint = "standard"

    def _validate_clarification_rule(self) -> IntentPacket:
        """Enforce SPEC §6.1: don't block cheap turns with clarifying questions."""
        if self.needs_clarification and self.budget_hint == "quick":
            raise ValueError(
                "needs_clarification must be False when budget_hint == 'quick'."
            )
        if self.needs_clarification and self.clarity_score >= 0.5:
            raise ValueError(
                "needs_clarification must be False when clarity_score >= 0.5."
            )
        if self.needs_clarification and not self.clarification_question:
            raise ValueError(
                "clarification_question is required when needs_clarification is True."
            )
        return self

    def model_post_init(self, __context: object) -> None:
        """Pydantic v2 hook — runs after default validation."""
        self._validate_clarification_rule()
