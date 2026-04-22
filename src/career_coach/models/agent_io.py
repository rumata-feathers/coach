"""Typed input / output schemas for every agent.

See SPEC §6. The schemas here are the *contract* — agent implementations in
:mod:`career_coach.agents` must not deviate. Keep this module free of logic
so the contract and implementation stay separately reviewable.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import (
    Challenge,
    EvidenceDraft,
    FactUpdate,
    Hypothesis,
    TurnSummary,
)

CriticFailureMode = Literal["generic", "ungrounded", "false_confidence", "off_intent"]


# -------- Understander --------------------------------------------------


class UnderstanderInput(BaseModel):
    """Input to :class:`~career_coach.agents.understander.Understander`."""

    user_message: str = Field(..., min_length=1)
    session_id: UUID
    session_theory: str | None = None
    recent_turns: list[TurnSummary] = Field(default_factory=list)
    user_facts: dict[str, Any] = Field(default_factory=dict)


# -------- Coach ---------------------------------------------------------


class CoachInput(BaseModel):
    """Input to :class:`~career_coach.agents.coach.Coach`."""

    intent_packet: IntentPacket
    user_facts: dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    recent_turns: list[TurnSummary] = Field(default_factory=list)
    critic_feedback: str | None = None


class CoachOutput(BaseModel):
    """Structured Coach response.

    The Coach MUST list which facts and hypotheses it grounded the response
    on — the Critic uses these fields to verify the grounding rule.
    """

    response_text: str = Field(..., min_length=1)
    referenced_facts: list[str] = Field(default_factory=list)
    referenced_hypotheses: list[UUID] = Field(default_factory=list)
    proposed_challenge: Challenge | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)


# -------- Critic --------------------------------------------------------


class CriticInput(BaseModel):
    """Input to :class:`~career_coach.agents.critic.Critic`."""

    coach_output: CoachOutput
    user_facts: dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    intent_packet: IntentPacket


class CriticVerdict(BaseModel):
    """Structured Critic decision.

    Rejection requires at least one failure mode so the Coach has a useful
    prompt on the retry; this is enforced by a root validator below.
    """

    verdict: Literal["pass", "reject"]
    failure_modes: list[CriticFailureMode] = Field(default_factory=list)
    specific_complaints: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None

    @model_validator(mode="after")
    def _reject_must_explain(self) -> CriticVerdict:
        if self.verdict == "reject" and not self.failure_modes:
            raise ValueError("A 'reject' verdict must list at least one failure mode.")
        if self.verdict == "pass" and self.failure_modes:
            raise ValueError("A 'pass' verdict must not list failure modes.")
        return self


# -------- Profiler ------------------------------------------------------


class ProfilerInput(BaseModel):
    """Input to :class:`~career_coach.agents.profiler.Profiler`."""

    turn_id: UUID
    user_message: str
    assistant_message: str
    existing_facts: dict[str, Any] = Field(default_factory=dict)


class ProfilerOutput(BaseModel):
    """Profiler extraction result — feeds the structured store + distillation queue."""

    new_facts: list[FactUpdate] = Field(default_factory=list)
    fact_updates: list[FactUpdate] = Field(default_factory=list)
    hypothesis_evidence: list[EvidenceDraft] = Field(default_factory=list)
