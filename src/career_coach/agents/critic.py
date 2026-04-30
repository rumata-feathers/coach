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

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY a JSON object matching this schema — "
    "no prose, no code fences, no think-blocks:\n"
    '{"verdict": "pass|reject", "failure_modes": [], '
    '"specific_complaints": [], "suggested_fix": null}'
)


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
            is_flow_c=input_data.is_flow_c,
            synthesizer_output=input_data.synthesizer_output,
            da_output=input_data.da_output,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None
        response = await self.complete(messages, response_format="json")

        try:
            verdict = _parse_verdict(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "Critic parse error on first attempt (%s); retrying once.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty response)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            retry_count = 1
            try:
                verdict = _parse_verdict(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                error_str = str(exc)
                fallback_reason = "retry_exhausted"
                logger.warning(
                    "Critic parse error after retry (%s); failing open with pass.", exc
                )
                # Fail open: a broken Critic shouldn't block the user.
                verdict = CriticVerdict(
                    verdict="pass",
                    failure_modes=[],
                    specific_complaints=[],
                    suggested_fix=None,
                )

        latency = self.now_ms() - t0

        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=verdict.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
            retry_count=retry_count,
            fallback_reason=fallback_reason,
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
        "is_flow_c": inp.is_flow_c,
        "has_synthesizer_output": inp.synthesizer_output is not None,
    }
