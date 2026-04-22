"""Core turn processing pipeline.

Implements the ``process_turn`` pseudocode from SPEC §10. Agents are wired
together here; this module owns no LLM logic itself.

Flow:
  1. Get or create session.
  2. Load context (facts, hypotheses, recent turns).
  3. Understander → IntentPacket.
  4. If clarification needed → short-circuit with the question.
  5. Orchestrator decides flow A or B.
  6. Coach + Critic retry loop (max 3 attempts on flow B).
  7. Persist turn.
  8. Update session theory.
  9. Queue Profiler as a background task (not awaited).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from career_coach.agents.coach import Coach
from career_coach.agents.critic import Critic
from career_coach.agents.orchestrator import Orchestrator
from career_coach.agents.profiler import Profiler
from career_coach.agents.understander import Understander
from career_coach.llm.factory import LLMFactory
from career_coach.memory.episodic import EpisodicRepo, Session
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo
from career_coach.models.agent_io import CoachInput, CriticInput, UnderstanderInput

logger = logging.getLogger("career_coach.pipeline.turn")

_MAX_COACH_RETRIES = 3  # 1 attempt + 2 retries


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
        self._coach = Coach(factory)
        self._critic = Critic(factory)
        self._profiler = Profiler(factory)

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

        # 5. Orchestrator
        flow = self._orchestrator.decide(intent_packet)
        logger.debug("Orchestrator chose flow %s.", flow)

        # 6. Coach + Critic retry loop
        critic_feedback: str | None = None
        critic_verdicts: list[dict] = []
        coach_out = None

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
                break

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
                break

            # Rejected — build feedback for next attempt
            critic_feedback = verdict.suggested_fix or "; ".join(verdict.specific_complaints)
            logger.debug(
                "Critic rejected attempt %d/%d: %s",
                attempt + 1,
                _MAX_COACH_RETRIES,
                critic_feedback,
            )
        else:
            # All retries exhausted — force an honest escalation response.
            logger.warning("Coach retry limit reached; using escalation response.")
            coach_out = await self._coach.run_escalation(
                CoachInput(
                    intent_packet=intent_packet,
                    user_facts=user_facts,
                    active_hypotheses=active_hypotheses,
                    recent_turns=recent_turns,
                    critic_feedback=critic_feedback,
                )
            )

        assert coach_out is not None  # always set by loop or escalation

        # 7. Persist turn
        turn_id = await self._episodic.save_turn(
            session_id=session.session_id,
            user_id=user_id,
            user_message=user_message,
            assistant_message=coach_out.response_text,
            intent_packet=intent_packet.model_dump(),
            flow_used=flow,
            critic_verdicts=critic_verdicts if critic_verdicts else None,
            tokens_used=None,
        )

        # 8. Update session theory
        await self._episodic.update_session_theory(
            session.session_id, intent_packet.session_theory
        )

        # 9. Queue Profiler as a background task (non-blocking)
        _task = asyncio.create_task(  # noqa: RUF006 — intentionally fire-and-forget
            self._profiler.run_and_save(
                user_id=user_id,
                turn_id=turn_id,
                user_message=user_message,
                assistant_message=coach_out.response_text,
                existing_facts=user_facts,
            ),
            name=f"profiler-{turn_id}",
        )

        return TurnResult(
            response=coach_out.response_text,
            turn_id=turn_id,
            session_id=session.session_id,
        )

    # ---- helpers -----------------------------------------------------------

    async def _get_or_create_session(
        self, user_id: UUID, session_id: UUID | None
    ) -> Session:
        if session_id is not None:
            session = await self._episodic.get_session(session_id)
            if session is not None:
                return session
        return await self._episodic.create_session(user_id)
