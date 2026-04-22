"""Coach agent.

Generates the user-facing response. The only agent whose output (post-Critic
approval) the user actually reads. Uses Claude Sonnet for quality.

See SPEC §6.3.
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
from career_coach.models.agent_io import CoachInput, CoachOutput
from career_coach.models.user_model import Challenge

logger = logging.getLogger("career_coach.agents.coach")

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY a JSON object matching this schema — "
    "no prose, no code fences, no think-blocks:\n"
    '{"response_text": "string", "referenced_facts": [], '
    '"referenced_hypotheses": [], "proposed_challenge": null, "uncertainty_flags": []}'
)

_ESCALATION_RESPONSE = (
    "I want to give you a genuinely useful response here, but I'm running into "
    "difficulty shaping one that I'm confident is actually grounded in what I know "
    "about you and addresses what you're asking. Could you help me by sharing a bit "
    "more context? For example: what specifically prompted this question, and what "
    "would a useful answer look like for you?"
)


class Coach(Agent):
    """Produces a :class:`CoachOutput` from an :class:`IntentPacket` + user model."""

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("coach", factory)

    async def run(
        self,
        input_data: CoachInput,
        *,
        turn_id: UUID | None = None,
    ) -> CoachOutput:
        """Generate a coaching response.

        Args:
            input_data: Typed input including intent, user model, and optional
                Critic feedback for retries.
            turn_id: Optional turn_id for the agent_calls log.

        Returns:
            A validated :class:`CoachOutput`. Falls back to a safe canned
            response on persistent parse failures.
        """
        prompt = self.render_prompt(
            "coach.j2",
            intent_packet=input_data.intent_packet,
            user_facts=input_data.user_facts,
            active_hypotheses=input_data.active_hypotheses,
            recent_turns=input_data.recent_turns,
            critic_feedback=input_data.critic_feedback,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        response = await self.complete(messages, response_format="json")

        try:
            output = _parse_coach_output(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "Coach parse error on first attempt (%s); retrying once.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty response)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            try:
                output = _parse_coach_output(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                error_str = str(exc)
                logger.warning("Coach parse error after retry (%s); using fallback output.", exc)
                output = _error_output(str(exc))

        latency = self.now_ms() - t0

        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=output.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
        )
        return output

    async def run_escalation(
        self,
        input_data: CoachInput,
        *,
        turn_id: UUID | None = None,
    ) -> CoachOutput:
        """Return an honest 'I need more context' response after retry exhaustion.

        Bypasses the Critic loop. Used by the turn pipeline after 3 failed
        Coach+Critic cycles to avoid an infinite loop.
        """
        output = CoachOutput(
            response_text=_ESCALATION_RESPONSE,
            referenced_facts=[],
            referenced_hypotheses=[],
            proposed_challenge=None,
            uncertainty_flags=["insufficient_context_for_grounded_response"],
        )
        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=output.model_dump(),
            latency_ms=0,
            tokens_in=None,
            tokens_out=None,
            error="escalation — critic retry limit reached",
        )
        return output


# ---- helpers -----------------------------------------------------------


def _parse_coach_output(raw: str) -> CoachOutput:
    """Strip optional markdown fences, parse JSON, and validate."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    # Pydantic coerces proposed_challenge from a dict if present.
    if data.get("proposed_challenge") and isinstance(data["proposed_challenge"], dict):
        data["proposed_challenge"] = Challenge(**data["proposed_challenge"])
    return CoachOutput(**data)


def _error_output(reason: str) -> CoachOutput:
    """Safe fallback output when parse fails even after stripping."""
    return CoachOutput(
        response_text=(
            "I ran into an issue preparing my response. Could you try rephrasing your question?"
        ),
        referenced_facts=[],
        referenced_hypotheses=[],
        proposed_challenge=None,
        uncertainty_flags=[f"internal_parse_error: {reason[:100]}"],
    )


def _serialize_input(inp: CoachInput) -> dict[str, Any]:
    return {
        "turn_intent": inp.intent_packet.turn_intent,
        "specific_ask": inp.intent_packet.specific_ask,
        "num_facts": len(inp.user_facts),
        "num_hypotheses": len(inp.active_hypotheses),
        "has_critic_feedback": inp.critic_feedback is not None,
    }
