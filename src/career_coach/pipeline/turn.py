"""Core turn processing pipeline.

Implements the ``process_turn`` pseudocode from SPEC §10. Agents are wired
together here; this module owns no LLM logic itself.

Flow:
  1. Get or create session.
  2. Load context (facts, hypotheses, recent turns).
  3. Understander → IntentPacket.
  4. If clarification needed → short-circuit with the question.
  5. Orchestrator decides flow A, B, or C.
  6a. Flow A: Coach only, no Critic.
  6b. Flow B: Coach + Critic retry loop (max 3 attempts).
  6c. Flow C: parallel Researcher + Coach → DA → Synthesizer + Critic retry
      (max 2 attempts), all within a 25 s latency budget.
  7. Persist turn (with Flow C extras if applicable).
  8. Update session theory.
  9. Queue Profiler as a background task (not awaited).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from career_coach.agents.coach import Coach
from career_coach.agents.critic import Critic
from career_coach.agents.devils_advocate import DevilsAdvocate
from career_coach.agents.orchestrator import Orchestrator
from career_coach.agents.profiler import Profiler
from career_coach.agents.researcher import Researcher
from career_coach.agents.synthesizer import Synthesizer
from career_coach.agents.understander import Understander
from career_coach.kb.repo import WorldKBRepo
from career_coach.llm.factory import LLMFactory
from career_coach.memory.episodic import EpisodicRepo, Session
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo
from career_coach.models.agent_io import (
    CoachInput,
    CriticInput,
    DevilsAdvocateInput,
    ResearcherInput,
    SynthesizerInput,
    UnderstanderInput,
)
from career_coach.pipeline.onboarding import OnboardingPipeline, OnboardingPolicy

logger = logging.getLogger("career_coach.pipeline.turn")

_MAX_COACH_RETRIES = 3   # Flow B: 1 attempt + 2 retries
_MAX_SYNTH_RETRIES = 2   # Flow C Synthesizer loop: 1 attempt + 1 retry
_FLOW_C_BUDGET_S = 25.0  # §7.2 latency budget


@dataclass(slots=True)
class TurnResult:
    """Result returned by :func:`process_turn`."""

    response: str
    turn_id: UUID | None
    session_id: UUID
    clarification_only: bool = False


class TurnPipeline:
    """Stateless service that processes one user turn end-to-end.

    Instantiate once at app start-up and reuse across requests.
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
        self._profiler = Profiler(factory)
        # Flow C agents.  Researcher is initialised lazily on first use so the
        # Tavily API key is not required at startup (e.g. in test environments
        # or when only Flow A/B turns are processed).  DA and Synthesizer have
        # no external key dependency and are created eagerly.
        self._researcher: Researcher | None = None
        self._devils_advocate = DevilsAdvocate(factory)
        self._synthesizer = Synthesizer(factory)
        # Keeps strong references to background tasks so the GC can't collect
        # them before they complete (Python GC collects unreferenced Tasks).
        self._background_tasks: set[asyncio.Task[Any]] = set()
        # Scratch dict for Flow C extras; reset at the start of every Flow C turn.
        self._flow_c_state: dict[str, Any] = {}

    async def process_turn(
        self,
        user_id: UUID,
        user_message: str,
        session_id: UUID | None = None,
    ) -> TurnResult:
        """Run one user turn through the full agent pipeline.

        Args:
            user_id: The user whose profile is used and updated.
            user_message: Raw text from the user.
            session_id: Optional existing session. Creates a new one if ``None``.

        Returns:
            :class:`TurnResult` with the assistant response and turn metadata.
        """
        # 1. Session
        session = await self._get_or_create_session(user_id, session_id)

        # 2. Load context
        user_facts = await self._structured.get_all(user_id)
        active_hypotheses = await self._semantic.get_active(user_id)
        recent_turns = await self._episodic.get_recent(session.session_id, limit=5)
        fact_count = len(user_facts)

        # 3. Understander
        intent_packet = await self._understander.run(
            UnderstanderInput(
                user_message=user_message,
                session_id=session.session_id,
                session_theory=session.session_theory,
                recent_turns=recent_turns,
                user_facts=user_facts,
            )
        )

        # 4. Short-circuit: clarification needed
        if intent_packet.needs_clarification and intent_packet.clarification_question:
            logger.debug("Understander requested clarification.")
            turn_id = await self._episodic.save_turn(
                session_id=session.session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=intent_packet.clarification_question,
                intent_packet=intent_packet.model_dump(),
                flow_used="clarification",
                critic_verdicts=None,
                tokens_used=None,
            )
            return TurnResult(
                response=intent_packet.clarification_question,
                turn_id=turn_id,
                session_id=session.session_id,
                clarification_only=True,
            )

        # 5. Onboarding check + Orchestrator
        is_new = await self._onboarding_policy.is_new_user(user_id)
        flow = self._orchestrator.decide(
            intent_packet, is_new_user=is_new, fact_count=fact_count
        )
        logger.debug("Orchestrator chose flow %s (is_new=%s, fact_count=%d).", flow, is_new, fact_count)

        # 6. Route to onboarding, Flow A, B, or C
        critic_verdicts: list[dict[str, Any]] = []
        coach_out = None
        # Flow C extras — populated on flow "C" only
        research_brief_id: UUID | None = None
        da_output_dict: dict[str, Any] | None = None
        synth_output_dict: dict[str, Any] | None = None
        chart_specs_list: list[dict[str, Any]] | None = None

        if flow == "onboarding":
            # Onboarding bypasses the Critic — probe questions need no grounding check.
            coach_out = await self._onboarding_pipeline.process_turn(
                user_id=user_id,
                session_id=session.session_id,
                user_message=user_message,
                intent_packet=intent_packet,
                user_facts=user_facts,
                recent_turns=recent_turns,
            )

        elif flow == "C":
            coach_out = await self._run_flow_c(
                user_id=user_id,
                user_message=user_message,
                intent_packet=intent_packet,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                recent_turns=recent_turns,
                critic_verdicts=critic_verdicts,
                session_id=session.session_id,
                # out-params populated by reference via mutable lists
                _research_brief_id_out=[None],
                _da_output_dict_out=[None],
                _synth_output_dict_out=[None],
                _chart_specs_list_out=[None],
            )
            # Un-pack from mutable carriers
            research_brief_id = self._flow_c_state.get("research_brief_id")
            da_output_dict = self._flow_c_state.get("da_output_dict")
            synth_output_dict = self._flow_c_state.get("synth_output_dict")
            chart_specs_list = self._flow_c_state.get("chart_specs_list")

        else:
            coach_out = await self._run_flow_ab(
                flow=flow,
                intent_packet=intent_packet,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                recent_turns=recent_turns,
                critic_verdicts=critic_verdicts,
            )

        assert coach_out is not None  # always set by every branch above

        # 7. Persist turn
        response_text = (
            synth_output_dict["response_text"]
            if synth_output_dict and "response_text" in synth_output_dict
            else coach_out.response_text
        )
        turn_id = await self._episodic.save_turn(
            session_id=session.session_id,
            user_id=user_id,
            user_message=user_message,
            assistant_message=response_text,
            intent_packet=intent_packet.model_dump(),
            flow_used=flow,
            critic_verdicts=critic_verdicts if critic_verdicts else None,
            tokens_used=None,
            research_brief_id=research_brief_id,
            devils_advocate_output=da_output_dict,
            synthesizer_output=synth_output_dict,
            chart_specs=chart_specs_list,
        )

        # 8. Update session theory
        await self._episodic.update_session_theory(
            session.session_id, intent_packet.session_theory
        )

        # 9. Queue Profiler as a background task (non-blocking).
        task = asyncio.create_task(
            self._profiler.run_and_save(
                user_id=user_id,
                turn_id=turn_id,
                user_message=user_message,
                assistant_message=response_text,
                existing_facts=user_facts,
            ),
            name=f"profiler-{turn_id}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return TurnResult(
            response=response_text,
            turn_id=turn_id,
            session_id=session.session_id,
        )

    # ---- Flow A/B ----------------------------------------------------------

    async def _run_flow_ab(
        self,
        *,
        flow: str,
        intent_packet: Any,
        user_facts: dict[str, Any],
        active_hypotheses: list[Any],
        recent_turns: list[Any],
        critic_verdicts: list[dict[str, Any]],
    ) -> Any:
        """Run Flow A (Coach only) or Flow B (Coach + Critic retry loop)."""
        critic_feedback: str | None = None

        for attempt in range(_MAX_COACH_RETRIES):
            coach_input = CoachInput(
                intent_packet=intent_packet,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                recent_turns=recent_turns,
                critic_feedback=critic_feedback,
            )
            coach_out = await self._coach.run(coach_input)

            if flow == "A":
                # Flow A skips the Critic entirely.
                return coach_out

            # Flow B: run Critic
            critic_input = CriticInput(
                coach_output=coach_out,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                intent_packet=intent_packet,
            )
            verdict = await self._critic.run(critic_input)
            critic_verdicts.append(verdict.model_dump())

            if verdict.verdict == "pass":
                return coach_out

            # Rejected — build feedback for next attempt
            critic_feedback = verdict.suggested_fix or "; ".join(
                verdict.specific_complaints
            )
            logger.debug(
                "Critic rejected attempt %d/%d: %s",
                attempt + 1,
                _MAX_COACH_RETRIES,
                critic_feedback,
            )

        # All retries exhausted — force an honest escalation response.
        logger.warning("Coach retry limit reached; using escalation response.")
        return await self._coach.run_escalation(
            CoachInput(
                intent_packet=intent_packet,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                recent_turns=recent_turns,
                critic_feedback=critic_feedback,
            )
        )

    # ---- Flow C ------------------------------------------------------------

    async def _run_flow_c(
        self,
        *,
        user_id: UUID,
        user_message: str,
        intent_packet: Any,
        user_facts: dict[str, Any],
        active_hypotheses: list[Any],
        recent_turns: list[Any],
        critic_verdicts: list[dict[str, Any]],
        session_id: UUID,
        # unused positional out-param dicts kept for clarity
        _research_brief_id_out: list[Any],
        _da_output_dict_out: list[Any],
        _synth_output_dict_out: list[Any],
        _chart_specs_list_out: list[Any],
    ) -> Any:
        """Run Flow C: parallel Researcher + Coach → DA → Synthesizer + Critic.

        Stores Flow C extras in ``self._flow_c_state`` before returning the
        Coach output (used as fallback or context).  The caller reads:
          ``self._flow_c_state["research_brief_id"]``
          ``self._flow_c_state["da_output_dict"]``
          ``self._flow_c_state["synth_output_dict"]``
          ``self._flow_c_state["chart_specs_list"]``

        On timeout, falls back to Coach output and logs ``flow_c_timeout``.
        """
        self._flow_c_state = {}
        t0 = asyncio.get_event_loop().time()

        try:
            async with asyncio.timeout(_FLOW_C_BUDGET_S):
                return await self._flow_c_inner(
                    intent_packet=intent_packet,
                    user_facts=user_facts,
                    active_hypotheses=active_hypotheses,
                    recent_turns=recent_turns,
                    critic_verdicts=critic_verdicts,
                )
        except TimeoutError:
            elapsed = asyncio.get_event_loop().time() - t0
            logger.warning(
                "Flow C timeout after %.1fs (budget=%.0fs); falling back to Coach output.",
                elapsed,
                _FLOW_C_BUDGET_S,
            )
            # Return a Coach-only fallback response.
            coach_out = await self._coach.run(
                CoachInput(
                    intent_packet=intent_packet,
                    user_facts=user_facts,
                    active_hypotheses=active_hypotheses,
                    recent_turns=recent_turns,
                    critic_feedback=None,
                )
            )
            # Record fallback reason in state so it can be persisted.
            self._flow_c_state["fallback_reason"] = "flow_c_timeout"
            return coach_out

    async def _flow_c_inner(
        self,
        *,
        intent_packet: Any,
        user_facts: dict[str, Any],
        active_hypotheses: list[Any],
        recent_turns: list[Any],
        critic_verdicts: list[dict[str, Any]],
    ) -> Any:
        """Inner coroutine for Flow C — wrapped by timeout in _run_flow_c."""
        # --- Phase 1: parallel Researcher + Coach ---
        researcher_input = ResearcherInput(
            question=intent_packet.specific_ask,
            user_facts=user_facts,
            depth="deep",
        )
        coach_input = CoachInput(
            intent_packet=intent_packet,
            user_facts=user_facts,
            active_hypotheses=active_hypotheses,
            recent_turns=recent_turns,
            critic_feedback=None,
        )
        research_brief, coach_out = await asyncio.gather(
            self._get_researcher().run(researcher_input),
            self._coach.run(coach_input),
        )

        # --- Phase 2: Devil's Advocate ---
        da_input = DevilsAdvocateInput(
            coach_output=coach_out,
            user_facts=user_facts,
            active_hypotheses=active_hypotheses,
            intent_packet=intent_packet,
            research_brief=research_brief,
        )
        da_out = await self._devils_advocate.run(da_input)

        # Persist DA output in state
        self._flow_c_state["da_output_dict"] = da_out.model_dump(mode="json")
        if research_brief and research_brief.brief_id:
            self._flow_c_state["research_brief_id"] = research_brief.brief_id

        # --- Phase 3: Synthesizer + Critic retry loop ---
        critic_feedback: str | None = None
        synth_out = None

        for attempt in range(_MAX_SYNTH_RETRIES):
            synth_input = SynthesizerInput(
                coach_output=coach_out,
                da_output=da_out,
                research_brief=research_brief,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                intent_packet=intent_packet,
                critic_feedback=critic_feedback,
            )
            synth_out = await self._synthesizer.run(synth_input)

            # Flow C Critic check
            critic_input = CriticInput(
                coach_output=coach_out,
                synthesizer_output=synth_out,
                da_output=da_out,
                user_facts=user_facts,
                active_hypotheses=active_hypotheses,
                intent_packet=intent_packet,
                is_flow_c=True,
            )
            verdict = await self._critic.run(critic_input)
            critic_verdicts.append(verdict.model_dump())

            if verdict.verdict == "pass":
                break

            critic_feedback = verdict.suggested_fix or "; ".join(verdict.specific_complaints)
            logger.debug(
                "Flow C Critic rejected Synthesizer attempt %d/%d: %s",
                attempt + 1,
                _MAX_SYNTH_RETRIES,
                critic_feedback,
            )

        assert synth_out is not None  # loop always runs at least once

        # Persist Synthesizer output in state
        self._flow_c_state["synth_output_dict"] = synth_out.model_dump(mode="json")
        self._flow_c_state["chart_specs_list"] = [
            c.model_dump(mode="json") for c in synth_out.chart_specs
        ]

        return coach_out  # pipeline uses coach_out for Profiler; response comes from synth_out

    # ---- helpers -----------------------------------------------------------

    def _get_researcher(self) -> Researcher:
        """Return the shared Researcher, creating it lazily on first call.

        Lazy creation keeps the Tavily API key out of the startup path — the
        key is only required when Flow C actually fires.
        """
        if self._researcher is None:
            from career_coach.web.factory import get_client  # deferred: reads Tavily key

            self._researcher = Researcher(
                factory=self._factory,
                web_client=get_client("researcher"),
                kb_repo=WorldKBRepo(),
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
