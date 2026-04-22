"""Critic agent.

Quality gate between the Coach and the user. Runs a structured 4-point rubric
and returns a pass/reject verdict with specific complaints on rejection.
Uses Claude Haiku for speed — it's evaluating, not generating.

See SPEC §6.4.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from career_coach.agents.base import Agent
from career_coach.llm.client import Message
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import CriticInput, CriticVerdict

logger = logging.getLogger("career_coach.agents.critic")


class Critic(Agent):
    """Evaluates a :class:`CoachOutput` and returns a :class:`CriticVerdict`."""

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("critic", factory)

    async def run(
        self,
        input_data: CriticInput,
        *,
        turn_id: UUID | None = None,
    ) -> CriticVerdict:
        """Evaluate a Coach response and return a pass/reject verdict.

        Fails *open* on parse errors: a malformed Critic response is treated
        as a pass so the retry loop doesn't get stuck on a broken Critic.

        Args:
            input_data: Typed input containing the Coach output + user model.
            turn_id: Optional turn_id for the agent_calls log.

        Returns:
            A validated :class:`CriticVerdict`.
        """
        prompt = self.render_prompt(
            "critic.j2",
            coach_output=input_data.coach_output,
            user_facts=input_data.user_facts,
            active_hypotheses=input_data.active_hypotheses,
            intent_packet=input_data.intent_packet,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        response = await self.complete(messages, response_format="json")
        latency = self.now_ms() - t0

        try:
            verdict = _parse_verdict(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
            error_str = str(exc)
            logger.warning("Critic parse error (%s); failing open with pass.", exc)
            # Fail open: a broken Critic shouldn't block the user.
            verdict = CriticVerdict(
                verdict="pass",
                failure_modes=[],
                specific_complaints=[],
                suggested_fix=None,
            )

        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=verdict.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
        )
        return verdict


# ---- helpers -----------------------------------------------------------


def _parse_verdict(raw: str) -> CriticVerdict:
    """Strip optional markdown fences and parse JSON → CriticVerdict."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    return CriticVerdict(**data)


def _serialize_input(inp: CriticInput) -> dict[str, Any]:
    return {
        "turn_intent": inp.intent_packet.turn_intent,
        "specific_ask": inp.intent_packet.specific_ask,
        "num_facts": len(inp.user_facts),
        "num_hypotheses": len(inp.active_hypotheses),
        "referenced_facts_count": len(inp.coach_output.referenced_facts),
        "referenced_hypotheses_count": len(inp.coach_output.referenced_hypotheses),
    }
