"""Onboarding pipeline and policy.

New users (< 5 facts OR < 3 turns) are routed here instead of the standard
Coach pipeline. The onboarder asks structured probe questions to gather the
context the coach needs before it can give grounded advice.

See SPEC_v0.5 §4.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from career_coach.agents.base import Agent
from career_coach.db import get_pool
from career_coach.llm.client import Message
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import CoachOutput
from career_coach.models.intent import IntentPacket
from career_coach.models.user_model import TurnSummary

logger = logging.getLogger("career_coach.pipeline.onboarding")

_NEW_USER_FACT_THRESHOLD = 5
_NEW_USER_TURN_THRESHOLD = 3

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY a JSON object matching this schema — "
    "no prose, no code fences, no think-blocks:\n"
    '{"response_text": "string", "referenced_facts": [], '
    '"referenced_hypotheses": [], "proposed_challenge": null, "uncertainty_flags": []}'
)

_FALLBACK_RESPONSE = (
    "Thanks for reaching out! To help you well, I'd like to understand a bit about "
    "where you're coming from. Could you tell me: what are you currently studying or "
    "working on, and what prompted this question today?"
)


class OnboardingPolicy:
    """Determines whether a user should be routed to the onboarding flow.

    Kept separate from the Orchestrator so the Orchestrator stays LLM-free
    and stateless (pure synchronous logic). This class owns the DB query.
    """

    async def is_new_user(self, user_id: UUID) -> bool:
        """Return ``True`` if the user needs onboarding.

        A user is "new" when EITHER:
        - ``structured_facts`` count < :data:`_NEW_USER_FACT_THRESHOLD` (5), OR
        - total ``turns`` count < :data:`_NEW_USER_TURN_THRESHOLD` (3).
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            fact_count: int = await conn.fetchval(
                "SELECT COUNT(*) FROM structured_facts WHERE user_id = $1",
                user_id,
            )
            if fact_count >= _NEW_USER_FACT_THRESHOLD:
                return False
            turn_count: int = await conn.fetchval(
                "SELECT COUNT(*) FROM turns WHERE user_id = $1",
                user_id,
            )
        return turn_count < _NEW_USER_TURN_THRESHOLD


class OnboardingPipeline(Agent):
    """Runs a structured onboarding probe instead of the standard Coach.

    Shares the Coach's model tier (Qwen3-235B) but uses a dedicated prompt
    template that cycles through three probe archetypes based on how many
    turns the user has had so far.
    """

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("onboarder", factory)

    async def process_turn(
        self,
        user_id: UUID,
        session_id: UUID,
        user_message: str,
        intent_packet: IntentPacket,
        user_facts: dict[str, Any],
        recent_turns: list[TurnSummary],
        *,
        turn_id: UUID | None = None,
    ) -> CoachOutput:
        """Generate an onboarding probe response.

        Args:
            user_id: Used to compute the onboarding turn index.
            session_id: Current session (not used in the prompt, kept for API symmetry).
            user_message: The user's raw message.
            intent_packet: Parsed intent (used for context but not for flow routing here).
            user_facts: Current known facts about the user.
            recent_turns: Last few turns in this session.
            turn_id: Optional turn_id for the agent_calls log.

        Returns:
            A :class:`CoachOutput` — same schema as the standard Coach so
            downstream persistence is unchanged.
        """
        turn_index = await self._get_onboarding_turn_index(user_id)

        prompt = self.render_prompt(
            "onboarder.j2",
            user_message=user_message,
            user_facts=user_facts,
            recent_turns=recent_turns,
            onboarding_turn_index=turn_index,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None
        response = await self.complete(messages, response_format="json")

        try:
            output = _parse_coach_output(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "Onboarder parse error on first attempt (%s); retrying once.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty response)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            retry_count = 1
            try:
                output = _parse_coach_output(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                error_str = str(exc)
                fallback_reason = "retry_exhausted"
                logger.warning(
                    "Onboarder parse error after retry (%s); using fallback.", exc
                )
                output = _fallback_output()

        latency = self.now_ms() - t0

        await self.log_call(
            turn_id=turn_id,
            input_payload={
                "user_message": user_message,
                "onboarding_turn_index": turn_index,
                "num_facts": len(user_facts),
                "num_recent_turns": len(recent_turns),
            },
            output_payload=output.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
            retry_count=retry_count,
            fallback_reason=fallback_reason,
        )
        return output

    async def _get_onboarding_turn_index(self, user_id: UUID) -> int:
        """Return the number of turns this user has already had (0-indexed probe selector)."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            count: int = await conn.fetchval(
                "SELECT COUNT(*) FROM turns WHERE user_id = $1",
                user_id,
            )
        return count


# ---- helpers ---------------------------------------------------------------


def _parse_coach_output(raw: str) -> CoachOutput:
    """Parse LLM response as a CoachOutput (same format as Coach agent)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    return CoachOutput(**data)


def _fallback_output() -> CoachOutput:
    """Safe fallback when parse fails even after retry."""
    return CoachOutput(
        response_text=_FALLBACK_RESPONSE,
        referenced_facts=[],
        referenced_hypotheses=[],
        proposed_challenge=None,
        uncertainty_flags=["onboarder_parse_error"],
    )
