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

CriticFailureMode = Literal[
    "generic",
    "ungrounded",
    "false_confidence",
    "off_intent",
    # Flow C — added in v1 (§7.3)
    "unintegrated",
    "uncontested",
    "chart_uncited",
    "chart_data_invented",
]


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

    ``chart_specs`` is populated on Flow B when the Coach emits a comparison
    chart (§6.3). Maximum 1 chart in v1. Flow A never emits charts.
    """

    response_text: str = Field(..., min_length=1)
    referenced_facts: list[str] = Field(default_factory=list)
    referenced_hypotheses: list[UUID] = Field(default_factory=list)
    proposed_challenge: Challenge | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)
    chart_specs: list[ChartSpec] = Field(default_factory=list)

    @field_validator("chart_specs")
    @classmethod
    def _max_one_chart(cls, v: list[ChartSpec]) -> list[ChartSpec]:
        """v1 allows at most 1 chart per Coach response."""
        if len(v) > 1:
            raise ValueError(
                f"CoachOutput allows at most 1 chart in v1, got {len(v)}"
            )
        return v


# -------- Critic --------------------------------------------------------


class CriticInput(BaseModel):
    """Input to :class:`~career_coach.agents.critic.Critic`.

    On Flow B: ``is_flow_c=False``, ``synthesizer_output=None``, ``da_output=None``.
    On Flow C: ``is_flow_c=True``; ``synthesizer_output`` and ``da_output`` are
    populated so the four additional failure modes (§7.3) can be checked.
    """

    coach_output: CoachOutput
    user_facts: dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    intent_packet: IntentPacket
    # Flow C extensions (§7.3)
    is_flow_c: bool = False
    synthesizer_output: SynthesizedResponse | None = None
    da_output: DevilsAdvocateOutput | None = None


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


# -------- Charts --------------------------------------------------------


class AxisSpec(BaseModel):
    """Axis definition for a :class:`ChartSpec`.

    Attributes:
        label: Display label (e.g. ``"Career"`` or ``"Median salary (GBP)``).
        unit: Physical unit of the axis values, e.g. ``"GBP"``, ``"years"``.
            ``None`` for dimensionless quantities or categorical axes.
        scale: ``"linear"`` (default) or ``"log"``.
    """

    label: str
    unit: str | None = None
    scale: Literal["linear", "log"] = "linear"


class ChartSpec(BaseModel):
    """Structured chart specification.

    Produced by the Coach (Flow B) or Synthesizer (Flow C) when §6.4 conditions
    hold. v2 frontend renders these; v1 persists them as structured output.

    Attributes:
        chart_type: Visual form of the chart.
        title: Chart title.
        description: One-sentence accessibility caption.
        x_axis: X-axis definition. ``None`` for pie charts and tables.
        y_axis: Y-axis definition. ``None`` for pie charts and tables.
        data: List of row dicts. Column names should match axis labels for
            bar/line/scatter charts.
        source_citation_indices: 0-based indices into the parent response's
            ``citations`` list. Must have at least one entry — uncited charts
            are rejected by the Critic (``chart_uncited`` failure mode).
    """

    chart_type: Literal["bar", "line", "scatter", "pie", "table", "range"]
    title: str
    description: str
    x_axis: AxisSpec | None = None
    y_axis: AxisSpec | None = None
    data: list[dict[str, Any]] = Field(default_factory=list)
    source_citation_indices: list[int] = Field(default_factory=list)

    @field_validator("source_citation_indices")
    @classmethod
    def _must_have_citation(cls, v: list[int]) -> list[int]:
        """A chart without citations is structurally invalid (Critic: chart_uncited)."""
        if not v:
            raise ValueError(
                "ChartSpec.source_citation_indices must have ≥1 entry. "
                "Uncited charts are not permitted (see SPEC_v1.md §6.4)."
            )
        return v


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


# -------- Devil's Advocate ----------------------------------------------


class CounterPoint(BaseModel):
    """One counter-point raised by the Devil's Advocate.

    Attributes:
        point: A concise statement of the counter-argument.
        reasoning: One or two sentences explaining why this matters to this user.
        severity: How significant the counter-point is: ``"low"``, ``"med"``, or ``"high"``.
        source_type: Where the counter-point originates:
            ``"user_profile"`` (grounded in user facts/hypotheses),
            ``"research"`` (from research brief findings), or
            ``"general"`` (common knowledge about the career path).
    """

    point: str = Field(..., min_length=5)
    reasoning: str = Field(..., min_length=10)
    severity: Literal["low", "med", "high"]
    source_type: Literal["user_profile", "research", "general"]


class Risk(BaseModel):
    """One structured risk raised by the Devil's Advocate.

    Attributes:
        scenario: A concrete description of what could go wrong.
        likelihood: Estimated likelihood of this scenario: ``"low"``, ``"med"``, ``"high"``.
        impact: Estimated impact if this scenario occurs: ``"low"``, ``"med"``, ``"high"``.
    """

    scenario: str = Field(..., min_length=10)
    likelihood: Literal["low", "med", "high"]
    impact: Literal["low", "med", "high"]


class DevilsAdvocateOutput(BaseModel):
    """Output of the Devil's Advocate agent.

    Attributes:
        counter_points: Structured counter-arguments. Must have ≥2 unless
            ``agrees_with_coach`` is ``True``.
        blind_spots: Observations about what the Coach may have overlooked.
        risks: Concrete downside scenarios with likelihood/impact estimates.
        agrees_with_coach: ``True`` if the DA genuinely agrees with the Coach
            response after honest assessment. When ``True``, ``counter_points``
            may be empty — this is an intentional "honest agreement" path.
    """

    counter_points: list[CounterPoint] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    agrees_with_coach: bool = False

    @model_validator(mode="after")
    def _enough_counter_points(self) -> DevilsAdvocateOutput:
        """Require ≥2 counter_points unless agrees_with_coach is True."""
        if not self.agrees_with_coach and len(self.counter_points) < 2:
            raise ValueError(
                "When agrees_with_coach is False, counter_points must have ≥2 entries "
                f"(got {len(self.counter_points)})."
            )
        return self


class DevilsAdvocateInput(BaseModel):
    """Input to :class:`~career_coach.agents.devils_advocate.DevilsAdvocate`.

    Attributes:
        coach_output: The Coach's response to critique.
        research_brief: Optional ResearchBrief from the Researcher (may be absent on
            Flow B or if Researcher produced an empty brief).
        user_facts: Structured user model facts for personalised counter-points.
        active_hypotheses: Current active hypotheses about the user.
        intent_packet: The original intent packet for context.
    """

    coach_output: CoachOutput
    research_brief: ResearchBrief | None = None
    user_facts: dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    intent_packet: IntentPacket


# -------- Synthesizer ---------------------------------------------------


class SynthesizerInput(BaseModel):
    """Input to :class:`~career_coach.agents.synthesizer.Synthesizer`.

    Attributes:
        coach_output: The Coach's response (always present on Flow C).
        da_output: The Devil's Advocate output.
        research_brief: Research brief, or ``None`` if Researcher was not run
            or produced an empty brief.
        user_facts: Structured user model facts for grounding.
        active_hypotheses: Current active hypotheses about the user.
        intent_packet: The original intent packet.
        critic_feedback: Populated on retry attempts — the Critic's complaints
            from the previous Synthesizer pass.
    """

    coach_output: CoachOutput
    da_output: DevilsAdvocateOutput
    research_brief: ResearchBrief | None = None
    user_facts: dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    intent_packet: IntentPacket
    critic_feedback: str | None = None


class SynthesizedResponse(BaseModel):
    """Output of the Synthesizer agent (Flow C user-facing voice).

    Attributes:
        response_text: User-facing response with inline citation numerals
            ``[1]``, ``[2]`` indexing into ``citations`` (1-based in text).
        referenced_facts: Fact keys from user_facts used in the response.
        referenced_hypotheses: Hypothesis UUIDs used in the response.
        referenced_findings: ``finding.claim`` strings from the ResearchBrief
            that the response draws on.
        citations: All sources cited in this response (deduped). Indexed by
            the ``[N]`` numerals in ``response_text`` (1-based, so ``[1]``
            is ``citations[0]``).
        surfaced_tradeoffs: Tradeoff statements that integrate Coach + DA + Research.
            Must have ≥1 entry when DA disagreed (``agrees_with_coach=False``).
        integrated_from: Which agent outputs were actually used. Critic checks
            this for completeness (``unintegrated`` failure mode).
        chart_specs: Optional chart. At most 1 in v1. Only present when §6.4
            conditions hold.
        proposed_challenge: Populated by v1.5 Initiator; v1 leaves this ``None``.
        uncertainty_flags: Hedges and caveats to flag to the user.
    """

    response_text: str = Field(..., min_length=1)
    referenced_facts: list[str] = Field(default_factory=list)
    referenced_hypotheses: list[UUID] = Field(default_factory=list)
    referenced_findings: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    surfaced_tradeoffs: list[str] = Field(default_factory=list)
    integrated_from: list[Literal["coach", "devils_advocate", "researcher"]] = Field(
        default_factory=list
    )
    chart_specs: list[ChartSpec] = Field(default_factory=list)
    proposed_challenge: Challenge | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)

    @field_validator("chart_specs")
    @classmethod
    def _max_one_chart(cls, v: list[ChartSpec]) -> list[ChartSpec]:
        """v1 allows at most 1 chart per synthesized response."""
        if len(v) > 1:
            raise ValueError(
                f"SynthesizedResponse allows at most 1 chart in v1, got {len(v)}"
            )
        return v

    @field_validator("integrated_from")
    @classmethod
    def _coach_always_integrated(
        cls, v: list[Literal["coach", "devils_advocate", "researcher"]]
    ) -> list[Literal["coach", "devils_advocate", "researcher"]]:
        """Coach output is always present on Flow C — must be in integrated_from."""
        if "coach" not in v:
            raise ValueError(
                "SynthesizedResponse.integrated_from must always include 'coach'."
            )
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


# Resolve forward references: CriticInput references SynthesizedResponse and
# DevilsAdvocateOutput which are defined later in this module.
CriticInput.model_rebuild()
