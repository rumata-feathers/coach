"""Typed input / output schemas for every agent.

See SPEC §6. The schemas here are the *contract* — agent implementations in
:mod:`career_coach.agents` must not deviate. Keep this module free of logic
so the contract and implementation stay separately reviewable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

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


# -------- Researcher ----------------------------------------------------


class Citation(BaseModel):
    """One source cited in a :class:`Finding`.

    Exactly one of ``url`` (web) or ``kb_path`` (KB file) must be populated.

    Attributes:
        url: Web URL; ``None`` for KB sources.
        kb_path: KB file path relative to the project root (e.g.
            ``"kb/careers/quant_finance.yaml"``); ``None`` for web sources.
        title: Human-readable title of the source.
        accessed_at: When the source was retrieved.
        source_type: ``"web"`` or ``"kb"``.
    """

    url: str | None = None
    kb_path: str | None = None
    title: str
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_type: Literal["web", "kb"]

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Citation:
        """Exactly one of url or kb_path must be set (never both, never neither)."""
        has_url = self.url is not None
        has_kb = self.kb_path is not None
        if has_url == has_kb:  # both True or both False
            raise ValueError(
                "Citation must have exactly one of 'url' (web) or 'kb_path' (KB); "
                f"got url={self.url!r}, kb_path={self.kb_path!r}"
            )
        return self


class Finding(BaseModel):
    """One grounded claim produced by the Researcher.

    Attributes:
        claim: The substantive finding, written as a full sentence.
        confidence: Researcher's self-assessed confidence, 0-1.
        citations: Sources supporting this claim. At least one required.
        is_numeric: ``True`` when the claim contains a specific number (salary,
            years, percentages, etc.). Numeric findings get extra Critic scrutiny.
    """

    claim: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation] = Field(..., min_length=1)
    is_numeric: bool = False


class ResearchBrief(BaseModel):
    """Output of the Researcher agent.

    Attributes:
        question: The original research question (unchanged from input).
        findings: Grounded claims with citations. Empty on graceful failure.
        caveats: Warnings about data gaps, truncation, or low confidence.
        next_questions: Follow-up questions the Researcher recommends.
        used_kb_files: KB file paths that contributed to this brief.
        web_searches_run: Search queries actually issued (not just planned).
        web_sources_consulted: URLs whose fetched content contributed to findings.
        brief_id: UUID of the persisted ``research_briefs`` row. Populated
            after DB persistence; ``None`` in unit tests that skip persistence.
    """

    question: str = Field(..., min_length=1)
    findings: list[Finding] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)
    used_kb_files: list[str] = Field(default_factory=list)
    web_searches_run: list[str] = Field(default_factory=list)
    web_sources_consulted: list[str] = Field(default_factory=list)
    brief_id: UUID | None = None


class RetrievalPlan(BaseModel):
    """Planning output from the researcher_planner LLM call.

    Consumed immediately by the Researcher — not persisted directly.

    Attributes:
        search_queries: Web search queries to issue (shallow: ≤3, deep: ≤8).
        kb_tags: KB tags to search for matching careers.
        kb_career_names: Specific career snake_case names to load directly,
            supplementing tag-based lookup.
        rationale: Brief explanation of the plan (for observability only).
    """

    search_queries: list[str] = Field(default_factory=list)
    kb_tags: list[str] = Field(default_factory=list)
    kb_career_names: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearcherInput(BaseModel):
    """Input to :class:`~career_coach.agents.researcher.Researcher`.

    Attributes:
        question: The research question to answer (usually derived from the
            user's turn intent).
        user_facts: Structured user model facts for personalisation.
        active_hypotheses: Current active hypotheses about the user.
        depth: Controls retrieval budget.
            ``"shallow"``: ≤3 searches, ≤2 fetches, ≤1 KB career deep-read, 8 s budget.
            ``"deep"``: ≤8 searches, ≤5 fetches, full matched KB, 15 s budget.
    """

    question: str = Field(..., min_length=1)
    user_facts: dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    depth: Literal["shallow", "deep"] = "shallow"

    @field_validator("depth")
    @classmethod
    def _valid_depth(cls, v: str) -> str:
        if v not in ("shallow", "deep"):
            raise ValueError(f"depth must be 'shallow' or 'deep', got {v!r}")
        return v


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
