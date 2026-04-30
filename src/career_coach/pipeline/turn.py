"""Core turn processing pipeline — LangGraph StateGraph implementation.

Implements the ``process_turn`` pseudocode from SPEC §10 as a LangGraph
StateGraph.  All agent logic is unchanged; this module re-expresses the
control flow (routing, retries, fan-out) as graph edges instead of
hand-rolled conditionals.

Nodes (per TASKS_v1.md §9):
  ``understander``, ``orchestrator``, ``coach``, ``researcher``,
  ``devils_advocate``, ``synthesizer``, ``critic``, ``supervisor``,
  ``onboarding``, ``profiler_dispatch``.

Plus structural nodes: ``load_context``, ``clarification_reply``,
``coach_escalation``, ``persist``, ``session_update``.

Flow routing:
  A  : coach → supervisor → persist
  B  : coach ⇆ critic (loop, max 3) → supervisor → persist
  C  : researcher ∥ coach (fan-out) → devils_advocate → synthesizer
       ⇆ critic (loop, max 2) → supervisor → persist
  onboarding: onboarding → supervisor → persist
  clarification: clarification_reply → END (skips full pipeline)
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict

from career_coach.agents.coach import Coach
from career_coach.agents.critic import Critic
from career_coach.agents.devils_advocate import DevilsAdvocate
from career_coach.agents.orchestrator import Orchestrator
from career_coach.agents.profiler import Profiler
from career_coach.agents.researcher import Researcher
from career_coach.agents.supervisor import Supervisor
from career_coach.agents.synthesizer import Synthesizer
from career_coach.agents.understander import Understander
from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.memory.episodic import EpisodicRepo, Session
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo
from career_coach.models.agent_io import (
    CoachInput,
    CoachOutput,
    CriticInput,
    DevilsAdvocateInput,
    DevilsAdvocateOutput,
    ResearchBrief,
    ResearcherInput,
    SupervisorEvent,
    SupervisorInput,
    SynthesizedResponse,
    SynthesizerInput,
    UnderstanderInput,
)
from career_coach.models.intent import IntentPacket
from career_coach.pipeline.onboarding import OnboardingPipeline, OnboardingPolicy

logger = logging.getLogger("career_coach.pipeline.turn")

_MAX_COACH_RETRIES = 3   # Flow B: 1 attempt + 2 retries
_MAX_SYNTH_RETRIES = 2   # Flow C Synthesizer loop: 1 attempt + 1 retry
_FLOW_C_BUDGET_S = 25.0  # §7.2 latency budget

# Appended to response when Supervisor action is "warn".
_SUPERVISOR_CAVEAT = (
    "\n\n*Note: some aspects of this response may benefit from additional "
    "verification. If you have concerns, consider consulting a relevant expert.*"
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class TurnState(TypedDict, total=False):
    """Full mutable state threaded through every node in the turn graph.

    Fields are ``NotRequired`` (``total=False``) so nodes only return the
    subset they modify; the graph merges partial updates.  The
    ``critic_verdicts`` field uses an ``Annotated`` reducer so each Critic
    call *appends* rather than overwrites.
    """

    # ---- inputs ----
    user_id: UUID
    user_message: str
    session_id_opt: UUID | None

    # ---- context (load_context) ----
    session: Session
    user_facts: dict[str, Any]
    active_hypotheses: list[Any]
    recent_turns: list[Any]
    fact_count: int

    # ---- understander ----
    intent_packet: IntentPacket

    # ---- orchestrator ----
    flow: str
    is_new_user: bool
    is_flow_c: bool

    # ---- coach ----
    coach_out: CoachOutput
    coach_attempt: int
    critic_feedback_ab: str | None

    # ---- flow C ----
    research_brief: ResearchBrief | None
    research_brief_id: UUID | None
    da_out: DevilsAdvocateOutput | None
    da_output_dict: dict[str, Any] | None
    synth_out: SynthesizedResponse | None
    synth_attempt: int
    critic_feedback_c: str | None
    synth_output_dict: dict[str, Any] | None
    chart_specs_list: list[dict[str, Any]] | None

    # ---- critic (accumulates across retries) ----
    critic_verdicts: Annotated[list[dict[str, Any]], operator.add]

    # ---- supervisor ----
    supervisor_event: SupervisorEvent | None

    # ---- final ----
    final_response: str
    turn_id: UUID | None
    clarification_only: bool


# ---------------------------------------------------------------------------
# TurnResult — public return type (unchanged from v0)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TurnResult:
    """Result returned by :func:`process_turn`."""

    response: str
    turn_id: UUID | None
    session_id: UUID
    clarification_only: bool = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TurnPipeline:
    """Stateless service that processes one user turn end-to-end.

    Instantiate once at app start-up and reuse across requests.
    Internally uses a compiled LangGraph :class:`StateGraph`.
    """

    def __init__(self, factory: LLMFactory) -> None:
        self._factory = factory
        self._structured = StructuredFactsRepo()
        self._episodic = EpisodicRepo()
        self._semantic = SemanticRepo()
        self._understander = Understander(factory)
        self._orchestrator = Orchestrator()
        self._onboarding_policy = OnboardingPolicy()
        self._onboarding_pipeline = OnboardingPipeline(factory)
        self._coach = Coach(factory)
        self._critic = Critic(factory)
        self._supervisor = Supervisor(factory)
        self._profiler = Profiler(factory)
        # Researcher is initialised lazily so Tavily key not required at startup.
        self._researcher: Researcher | None = None
        self._devils_advocate = DevilsAdvocate(factory)
        self._synthesizer = Synthesizer(factory)
        # Background task tracking (prevents GC before completion).
        self._background_tasks: set[asyncio.Task[Any]] = set()
        # Compile the graph once at construction time.
        self._graph: CompiledGraph = self._build_graph()

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    async def process_turn(
        self,
        user_id: UUID,
        user_message: str,
        session_id: UUID | None = None,
    ) -> TurnResult:
        """Run one user turn through the full agent graph.

        Args:
            user_id: The user whose profile is used and updated.
            user_message: Raw text from the user.
            session_id: Optional existing session. Creates a new one if ``None``.

        Returns:
            :class:`TurnResult` with the assistant response and turn metadata.
        """
        initial_state: TurnState = {
            "user_id": user_id,
            "user_message": user_message,
            "session_id_opt": session_id,
            "user_facts": {},
            "active_hypotheses": [],
            "recent_turns": [],
            "fact_count": 0,
            "flow": "B",
            "is_new_user": False,
            "is_flow_c": False,
            "coach_attempt": 0,
            "synth_attempt": 0,
            "critic_verdicts": [],
            "clarification_only": False,
            "research_brief": None,
            "research_brief_id": None,
            "da_out": None,
            "da_output_dict": None,
            "synth_out": None,
            "critic_feedback_ab": None,
            "critic_feedback_c": None,
            "synth_output_dict": None,
            "chart_specs_list": None,
            "supervisor_event": None,
            "final_response": "",
            "turn_id": None,
        }

        result_state: dict[str, Any] = await self._graph.ainvoke(initial_state)

        session: Session = result_state["session"]
        return TurnResult(
            response=result_state.get("final_response", ""),
            turn_id=result_state.get("turn_id"),
            session_id=session.session_id,
            clarification_only=result_state.get("clarification_only", False),
        )

    # -------------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------------

    def _build_graph(self) -> CompiledGraph:
        """Compile the turn StateGraph.  Called once in ``__init__``."""
        g: StateGraph = StateGraph(TurnState)

        # --- Register nodes ---
        g.add_node("load_context", self._node_load_context)
        g.add_node("understander", self._node_understander)
        g.add_node("clarification_reply", self._node_clarification_reply)
        g.add_node("orchestrator", self._node_orchestrator)
        g.add_node("onboarding", self._node_onboarding)
        g.add_node("researcher", self._node_researcher)
        g.add_node("coach", self._node_coach)
        g.add_node("coach_escalation", self._node_coach_escalation)
        g.add_node("devils_advocate", self._node_devils_advocate)
        g.add_node("synthesizer", self._node_synthesizer)
        g.add_node("critic", self._node_critic)
        g.add_node("supervisor", self._node_supervisor)
        g.add_node("persist", self._node_persist)
        g.add_node("session_update", self._node_session_update)
        g.add_node("profiler_dispatch", self._node_profiler_dispatch)

        # --- Entry point ---
        g.set_entry_point("load_context")
        g.add_edge("load_context", "understander")

        # --- After understander: clarification short-circuit or main path ---
        g.add_conditional_edges("understander", self._route_understander)
        g.add_edge("clarification_reply", END)

        # --- After orchestrator: route to onboarding, Flow A/B (coach only),
        #     or Flow C fan-out (researcher + coach in parallel) ---
        g.add_edge("orchestrator", "orchestrator_route")
        g.add_node("orchestrator_route", lambda s: {})  # no-op fan-out bridge
        g.add_conditional_edges(
            "orchestrator_route",
            self._route_orchestrator,  # type: ignore[arg-type]
        )

        # Onboarding converges to supervisor
        g.add_edge("onboarding", "supervisor")

        # Researcher (Flow C only) always feeds into devils_advocate fan-in
        g.add_edge("researcher", "devils_advocate")

        # Coach: route depends on flow
        g.add_conditional_edges("coach", self._route_coach)

        # Coach escalation (Flow B retry exhausted) → supervisor
        g.add_edge("coach_escalation", "supervisor")

        # Flow C: DA → synthesizer → critic (loop)
        g.add_edge("devils_advocate", "synthesizer")
        g.add_edge("synthesizer", "critic")
        g.add_conditional_edges("critic", self._route_critic)

        # Supervisor → persist → session_update → profiler_dispatch → END
        g.add_edge("supervisor", "persist")
        g.add_edge("persist", "session_update")
        g.add_edge("session_update", "profiler_dispatch")
        g.add_edge("profiler_dispatch", END)

        return g.compile()

    # -------------------------------------------------------------------------
    # Node implementations
    # -------------------------------------------------------------------------

    async def _node_load_context(self, state: TurnState) -> dict[str, Any]:
        """Steps 1-2: Get or create session, load context."""
        session = await self._get_or_create_session(
            state["user_id"], state.get("session_id_opt")
        )
        user_facts = await self._structured.get_all(state["user_id"])
        active_hypotheses = await self._semantic.get_active(state["user_id"])
        recent_turns = await self._episodic.get_recent(session.session_id, limit=5)
        return {
            "session": session,
            "user_facts": user_facts,
            "active_hypotheses": active_hypotheses,
            "recent_turns": recent_turns,
            "fact_count": len(user_facts),
        }

    async def _node_understander(self, state: TurnState) -> dict[str, Any]:
        """Step 3: Understander → IntentPacket."""
        session: Session = state["session"]
        intent_packet = await self._understander.run(
            UnderstanderInput(
                user_message=state["user_message"],
                session_id=session.session_id,
                session_theory=session.session_theory,
                recent_turns=state.get("recent_turns", []),
                user_facts=state.get("user_facts", {}),
            )
        )
        return {"intent_packet": intent_packet}

    async def _node_clarification_reply(self, state: TurnState) -> dict[str, Any]:
        """Step 4 (short-circuit): Save clarification question, skip full pipeline."""
        intent_packet: IntentPacket = state["intent_packet"]
        session: Session = state["session"]
        turn_id = await self._episodic.save_turn(
            session_id=session.session_id,
            user_id=state["user_id"],
            user_message=state["user_message"],
            assistant_message=intent_packet.clarification_question,
            intent_packet=intent_packet.model_dump(),
            flow_used="clarification",
            critic_verdicts=None,
            tokens_used=None,
        )
        return {
            "final_response": intent_packet.clarification_question,
            "turn_id": turn_id,
            "clarification_only": True,
        }

    async def _node_orchestrator(self, state: TurnState) -> dict[str, Any]:
        """Step 5: OnboardingPolicy + Orchestrator → flow."""
        is_new = await self._onboarding_policy.is_new_user(state["user_id"])
        intent_packet: IntentPacket = state["intent_packet"]
        flow = self._orchestrator.decide(
            intent_packet, is_new_user=is_new, fact_count=state.get("fact_count", 0)
        )
        logger.debug(
            "Orchestrator chose flow %s (is_new=%s, fact_count=%d).",
            flow, is_new, state.get("fact_count", 0),
        )
        return {"flow": flow, "is_new_user": is_new, "is_flow_c": flow == "C"}

    async def _node_onboarding(self, state: TurnState) -> dict[str, Any]:
        """Onboarding flow (step 6 — onboarding path)."""
        session: Session = state["session"]
        coach_out = await self._onboarding_pipeline.process_turn(
            user_id=state["user_id"],
            session_id=session.session_id,
            user_message=state["user_message"],
            intent_packet=state["intent_packet"],
            user_facts=state.get("user_facts", {}),
            recent_turns=state.get("recent_turns", []),
        )
        return {"coach_out": coach_out}

    async def _node_researcher(self, state: TurnState) -> dict[str, Any]:
        """Flow C Phase 1a: Researcher → ResearchBrief.

        Runs in parallel with ``coach`` node via LangGraph fan-out.
        Fan-in at ``devils_advocate``.
        """
        researcher_input = ResearcherInput(
            question=state["intent_packet"].specific_ask,
            user_facts=state.get("user_facts", {}),
            depth="deep",
        )
        research_brief = await self._get_researcher().run(researcher_input)
        result: dict[str, Any] = {"research_brief": research_brief}
        if research_brief and research_brief.brief_id:
            result["research_brief_id"] = research_brief.brief_id
        return result

    async def _node_coach(self, state: TurnState) -> dict[str, Any]:
        """Flow A/B/C Phase 1b: Coach → CoachOutput.

        For Flow C, runs in parallel with ``researcher`` via fan-out.
        For Flow B, re-uses ``critic_feedback_ab`` on retry iterations.
        """
        flow = state.get("flow", "B")
        attempt = state.get("coach_attempt", 0)
        # Only provide critic_feedback on Flow B retries (not Flow A or C).
        critic_feedback = state.get("critic_feedback_ab") if flow == "B" else None

        coach_input = CoachInput(
            intent_packet=state["intent_packet"],
            user_facts=state.get("user_facts", {}),
            active_hypotheses=state.get("active_hypotheses", []),
            recent_turns=state.get("recent_turns", []),
            critic_feedback=critic_feedback,
        )
        coach_out = await self._coach.run(coach_input)
        return {"coach_out": coach_out, "coach_attempt": attempt + 1}

    async def _node_coach_escalation(self, state: TurnState) -> dict[str, Any]:
        """Flow B: Coach retry limit reached — use escalation response."""
        logger.warning("Coach retry limit reached; using escalation response.")
        coach_out = await self._coach.run_escalation(
            CoachInput(
                intent_packet=state["intent_packet"],
                user_facts=state.get("user_facts", {}),
                active_hypotheses=state.get("active_hypotheses", []),
                recent_turns=state.get("recent_turns", []),
                critic_feedback=state.get("critic_feedback_ab"),
            )
        )
        return {"coach_out": coach_out}

    async def _node_devils_advocate(self, state: TurnState) -> dict[str, Any]:
        """Flow C Phase 2: DevilsAdvocate → challenges the coach output.

        Fan-in point: LangGraph waits for both ``researcher`` and ``coach``
        to complete before executing this node.
        """
        da_input = DevilsAdvocateInput(
            coach_output=state["coach_out"],
            user_facts=state.get("user_facts", {}),
            active_hypotheses=state.get("active_hypotheses", []),
            intent_packet=state["intent_packet"],
            research_brief=state.get("research_brief"),
        )
        da_out = await self._devils_advocate.run(da_input)
        return {
            "da_out": da_out,
            "da_output_dict": da_out.model_dump(mode="json"),
        }

    async def _node_synthesizer(self, state: TurnState) -> dict[str, Any]:
        """Flow C Phase 3a: Synthesizer produces the final deliberation response.

        On retry (``synth_attempt > 0``), receives ``critic_feedback_c``.
        """
        attempt = state.get("synth_attempt", 0)
        # Only pass critic feedback on retry iterations.
        critic_feedback = state.get("critic_feedback_c") if attempt > 0 else None

        synth_input = SynthesizerInput(
            coach_output=state["coach_out"],
            da_output=state["da_out"],
            research_brief=state.get("research_brief"),
            user_facts=state.get("user_facts", {}),
            active_hypotheses=state.get("active_hypotheses", []),
            intent_packet=state["intent_packet"],
            critic_feedback=critic_feedback,
        )
        synth_out = await self._synthesizer.run(synth_input)
        return {
            "synth_out": synth_out,
            "synth_attempt": attempt + 1,
            "synth_output_dict": synth_out.model_dump(mode="json"),
            "chart_specs_list": [c.model_dump(mode="json") for c in synth_out.chart_specs],
        }

    async def _node_critic(self, state: TurnState) -> dict[str, Any]:
        """Flow B/C: Critic evaluates the coach/synthesizer output.

        Populates ``critic_feedback_ab`` (Flow B) or ``critic_feedback_c``
        (Flow C) for the next retry iteration.
        """
        is_flow_c = state.get("is_flow_c", False)

        if is_flow_c:
            critic_input = CriticInput(
                coach_output=state["coach_out"],
                synthesizer_output=state.get("synth_out"),
                da_output=state.get("da_out"),
                user_facts=state.get("user_facts", {}),
                active_hypotheses=state.get("active_hypotheses", []),
                intent_packet=state["intent_packet"],
                is_flow_c=True,
            )
        else:
            critic_input = CriticInput(
                coach_output=state["coach_out"],
                user_facts=state.get("user_facts", {}),
                active_hypotheses=state.get("active_hypotheses", []),
                intent_packet=state["intent_packet"],
            )

        verdict = await self._critic.run(critic_input)
        result: dict[str, Any] = {"critic_verdicts": [verdict.model_dump()]}

        if verdict.verdict != "pass":
            feedback = verdict.suggested_fix or "; ".join(verdict.specific_complaints)
            logger.debug(
                "Critic rejected attempt (flow_c=%s): %s", is_flow_c, feedback
            )
            if is_flow_c:
                result["critic_feedback_c"] = feedback
            else:
                result["critic_feedback_ab"] = feedback

        return result

    async def _node_supervisor(self, state: TurnState) -> dict[str, Any]:
        """Step 6.5: Supervisor — final pre-response safety/correctness check."""
        synth_out: SynthesizedResponse | None = state.get("synth_out")
        coach_out: CoachOutput = state["coach_out"]

        initial_response = (
            synth_out.response_text
            if synth_out is not None and synth_out.response_text
            else coach_out.response_text
        )

        final_response, supervisor_event, new_synth_out = await self._apply_supervisor(
            response_text=initial_response,
            user_message=state["user_message"],
            user_facts=state.get("user_facts", {}),
            specific_ask=state["intent_packet"].specific_ask,
            flow=state.get("flow", "B"),
            intent_packet=state["intent_packet"],
            active_hypotheses=state.get("active_hypotheses", []),
            recent_turns=state.get("recent_turns", []),
            coach_out_obj=coach_out,
            da_out_obj=state.get("da_out"),
            research_brief_obj=state.get("research_brief"),
        )

        result: dict[str, Any] = {
            "final_response": final_response,
            "supervisor_event": supervisor_event,
        }

        if new_synth_out is not None:
            # Supervisor retry produced a new synthesizer output — update state.
            result["synth_out"] = new_synth_out
            result["synth_output_dict"] = new_synth_out.model_dump(mode="json")
            result["chart_specs_list"] = [
                c.model_dump(mode="json") for c in new_synth_out.chart_specs
            ]
        elif (
            state.get("is_flow_c")
            and final_response != initial_response
            and state.get("synth_output_dict")
        ):
            # Warn action changed the response text — keep synth_output_dict in sync.
            existing_dict: dict[str, Any] = state.get("synth_output_dict") or {}
            result["synth_output_dict"] = dict(existing_dict, response_text=final_response)

        return result

    async def _node_persist(self, state: TurnState) -> dict[str, Any]:
        """Step 7: Persist turn; Step 7.5: Persist non-pass Supervisor event."""
        session: Session = state["session"]
        turn_id = await self._episodic.save_turn(
            session_id=session.session_id,
            user_id=state["user_id"],
            user_message=state["user_message"],
            assistant_message=state.get("final_response", ""),
            intent_packet=state["intent_packet"].model_dump(),
            flow_used=state.get("flow", "B"),
            critic_verdicts=state.get("critic_verdicts") or None,
            tokens_used=None,
            research_brief_id=state.get("research_brief_id"),
            devils_advocate_output=state.get("da_output_dict"),
            synthesizer_output=state.get("synth_output_dict"),
            chart_specs=state.get("chart_specs_list"),
        )

        # Supervisor events are persisted after the turn row exists (valid FK).
        supervisor_event: SupervisorEvent | None = state.get("supervisor_event")
        if supervisor_event is not None and supervisor_event.action != "pass":
            await self._save_supervisor_event(turn_id, supervisor_event)

        return {"turn_id": turn_id}

    async def _node_session_update(self, state: TurnState) -> dict[str, Any]:
        """Step 8: Update session theory."""
        session: Session = state["session"]
        intent: IntentPacket = state["intent_packet"]
        await self._episodic.update_session_theory(session.session_id, intent.session_theory)
        return {}

    async def _node_profiler_dispatch(self, state: TurnState) -> dict[str, Any]:
        """Step 9: Queue Profiler as a fire-and-forget background task."""
        turn_id: UUID | None = state.get("turn_id")
        if turn_id is None:
            logger.warning("Profiler dispatch: turn_id is None; skipping profiler.")
            return {}
        task = asyncio.create_task(
            self._profiler.run_and_save(
                user_id=state["user_id"],
                turn_id=turn_id,
                user_message=state["user_message"],
                assistant_message=state.get("final_response", ""),
                existing_facts=state.get("user_facts", {}),
            ),
            name=f"profiler-{turn_id}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return {}

    # -------------------------------------------------------------------------
    # Routing functions (conditional edges)
    # -------------------------------------------------------------------------

    def _route_understander(self, state: TurnState) -> str:
        intent: IntentPacket | None = state.get("intent_packet")
        if intent and intent.needs_clarification and intent.clarification_question:
            logger.debug("Understander requested clarification.")
            return "clarification_reply"
        return "orchestrator"

    def _route_orchestrator(self, state: TurnState) -> list[str] | str:
        """Fan-out to researcher + coach for Flow C; single route otherwise."""
        flow = state.get("flow", "B")
        if flow == "onboarding":
            return "onboarding"
        if flow == "C":
            # Parallel fan-out: researcher and coach run simultaneously.
            # Fan-in at devils_advocate (waits for both edges).
            return ["researcher", "coach"]
        return "coach"  # Flow A or B

    def _route_coach(self, state: TurnState) -> str:
        flow = state.get("flow", "B")
        if flow == "A":
            return "supervisor"
        if flow == "C":
            # Flow C fan-in: coach result joins researcher at devils_advocate.
            return "devils_advocate"
        # Flow B → run Critic.
        return "critic"

    def _route_critic(self, state: TurnState) -> str:
        """Route after Critic: pass → supervisor, retry or exhausted → loop/escalate."""
        is_flow_c = state.get("is_flow_c", False)
        verdicts: list[dict[str, Any]] = state.get("critic_verdicts", [])
        last = verdicts[-1] if verdicts else None

        if is_flow_c:
            if not last or last.get("verdict") == "pass":
                return "supervisor"
            synth_attempt = state.get("synth_attempt", 0)
            if synth_attempt >= _MAX_SYNTH_RETRIES:
                logger.warning(
                    "Flow C Synthesizer retry limit reached; sending to supervisor anyway."
                )
                return "supervisor"
            return "synthesizer"
        else:
            if not last or last.get("verdict") == "pass":
                return "supervisor"
            coach_attempt = state.get("coach_attempt", 0)
            if coach_attempt >= _MAX_COACH_RETRIES:
                return "coach_escalation"
            return "coach"

    # -------------------------------------------------------------------------
    # Supervisor helpers
    # -------------------------------------------------------------------------

    async def _apply_supervisor(
        self,
        *,
        response_text: str,
        user_message: str,
        user_facts: dict[str, Any],
        specific_ask: str,
        flow: str,
        intent_packet: Any,
        active_hypotheses: list[Any],
        recent_turns: list[Any],
        coach_out_obj: Any = None,
        da_out_obj: Any = None,
        research_brief_obj: Any = None,
    ) -> tuple[str, SupervisorEvent, SynthesizedResponse | None]:
        """Run Supervisor check and handle the action.

        Args:
            coach_out_obj: Coach output object for retry (Flow A/B/C).
            da_out_obj: DA output for retry (Flow C only).
            research_brief_obj: Research brief for retry (Flow C only).

        Returns:
            ``(final_response_text, supervisor_event, updated_synth_out | None)``.
            ``updated_synth_out`` is non-None only when the supervisor retry
            produced a new Synthesizer output (Flow C retry path).
        """
        sup_input = SupervisorInput(
            user_message=user_message,
            response_text=response_text,
            user_facts=user_facts,
            specific_ask=specific_ask,
            flow_used=flow,
        )
        event = await self._supervisor.run(sup_input)

        if event.action == "pass":
            return response_text, event, None

        if event.action == "block":
            override = event.scripted_override or response_text
            logger.warning("Supervisor BLOCK on flow %s: %s", flow, event.details)
            return override, event, None

        if event.action == "warn":
            logger.info("Supervisor WARN on flow %s: %s", flow, event.details)
            return response_text + _SUPERVISOR_CAVEAT, event, None

        # action == "retry"
        logger.info(
            "Supervisor RETRY on flow %s: %s — re-running agent once.",
            flow, event.details,
        )
        retry_response, new_synth_out = await self._supervisor_rerun(
            flow=flow,
            supervisor_feedback=event.details or "Supervisor flagged an issue.",
            intent_packet=intent_packet,
            user_facts=user_facts,
            active_hypotheses=active_hypotheses,
            recent_turns=recent_turns,
            original_response=response_text,
            coach_out_obj=coach_out_obj,
            da_out_obj=da_out_obj,
            research_brief_obj=research_brief_obj,
        )

        # Re-check the retry response.
        retry_event = await self._supervisor.run(
            SupervisorInput(
                user_message=user_message,
                response_text=retry_response,
                user_facts=user_facts,
                specific_ask=specific_ask,
                flow_used=flow,
            )
        )
        if retry_event.action == "pass":
            return retry_response, event, new_synth_out

        # Retry still tripped — fall through to warn.
        logger.info(
            "Supervisor retry also tripped (%s); falling through to warn.",
            retry_event.action,
        )
        return retry_response + _SUPERVISOR_CAVEAT, event, new_synth_out

    async def _supervisor_rerun(
        self,
        *,
        flow: str,
        supervisor_feedback: str,
        intent_packet: Any,
        user_facts: dict[str, Any],
        active_hypotheses: list[Any],
        recent_turns: list[Any],
        original_response: str,
        coach_out_obj: Any = None,
        da_out_obj: Any = None,
        research_brief_obj: Any = None,
    ) -> tuple[str, SynthesizedResponse | None]:
        """Re-run the appropriate agent once with supervisor feedback.

        Returns:
            ``(response_text, new_synth_out | None)``.
            ``new_synth_out`` is only non-None for Flow C.
        """
        if flow in ("A", "B"):
            retry_out = await self._coach.run(
                CoachInput(
                    intent_packet=intent_packet,
                    user_facts=user_facts,
                    active_hypotheses=active_hypotheses,
                    recent_turns=recent_turns,
                    critic_feedback=supervisor_feedback,
                )
            )
            return retry_out.response_text, None

        if flow == "C" and coach_out_obj is not None and da_out_obj is not None:
            retry_synth = await self._synthesizer.run(
                SynthesizerInput(
                    coach_output=coach_out_obj,
                    da_output=da_out_obj,
                    research_brief=research_brief_obj,
                    user_facts=user_facts,
                    active_hypotheses=active_hypotheses,
                    intent_packet=intent_packet,
                    critic_feedback=supervisor_feedback,
                )
            )
            return retry_synth.response_text, retry_synth

        # Onboarding or state unavailable — no retry.
        return original_response, None

    async def _save_supervisor_event(
        self, turn_id: UUID, event: SupervisorEvent
    ) -> None:
        """Persist a non-pass SupervisorEvent to the supervisor_events table."""
        if event.action == "pass":
            return
        if event.event_type is None or event.severity is None:
            logger.warning(
                "Supervisor non-pass event missing event_type/severity; skipping persist."
            )
            return
        details_json = json.dumps({"details": event.details}) if event.details else None
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO supervisor_events
                        (turn_id, event_type, severity, details, action_taken)
                    VALUES ($1, $2, $3, $4::jsonb, $5)
                    """,
                    turn_id,
                    event.event_type,
                    event.severity,
                    details_json,
                    event.action,
                )
        except Exception:
            logger.exception("Failed to persist supervisor_event for turn %s", turn_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_researcher(self) -> Researcher:
        """Return the shared Researcher, creating it lazily on first call."""
        if self._researcher is None:
            from career_coach.web.factory import get_client  # deferred: reads Tavily key

            self._researcher = Researcher(
                factory=self._factory,
                web_client=get_client("researcher"),
            )
        return self._researcher

    async def _get_or_create_session(
        self, user_id: UUID, session_id: UUID | None
    ) -> Session:
        if session_id is not None:
            session = await self._episodic.get_session(session_id)
            if session is not None:
                return session
        return await self._episodic.create_session(user_id)
