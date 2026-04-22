"""Understander agent.

Compresses a user message + session context into a structured
:class:`IntentPacket`. Cheap, fast model; one job: interpret, not advise.

See SPEC §6.1.
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
from career_coach.models.agent_io import UnderstanderInput
from career_coach.models.intent import IntentPacket

logger = logging.getLogger("career_coach.agents.understander")


class Understander(Agent):
    """Produces an :class:`IntentPacket` from a user message."""

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("understander", factory)

    async def run(
        self,
        input_data: UnderstanderInput,
        *,
        turn_id: UUID | None = None,
    ) -> IntentPacket:
        """Run the Understander and return a validated :class:`IntentPacket`.

        Args:
            input_data: Typed input from the turn pipeline.
            turn_id: Optional turn_id for the agent_calls log (``None`` on
                the first pass before the turn row is written).

        Returns:
            A validated :class:`IntentPacket`.

        Raises:
            ValueError: If the LLM returns invalid JSON or a schema that
                fails Pydantic validation after the retry.
        """
        prompt = self.render_prompt(
            "understander.j2",
            user_message=input_data.user_message,
            session_theory=input_data.session_theory,
            user_facts=input_data.user_facts,
            recent_turns=input_data.recent_turns,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        response = await self.complete(messages, response_format="json")
        latency = self.now_ms() - t0

        try:
            packet = _parse_intent_packet(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            error_str = str(exc)
            logger.warning("Understander parse error (%s); returning fallback.", exc)
            packet = _fallback_packet(input_data)

        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=packet.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
        )
        return packet


# ---- helpers -----------------------------------------------------------


def _parse_intent_packet(raw: str) -> IntentPacket:
    """Strip optional markdown fences and parse JSON → IntentPacket."""
    text = raw.strip()
    if text.startswith("```"):
        # Remove code fences if the model added them despite instructions.
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    return IntentPacket(**data)


def _fallback_packet(inp: UnderstanderInput) -> IntentPacket:
    """Conservative fallback used when the LLM returns malformed output.

    Returns a 'standard' packet that routes to the Coach without blocking the
    turn — better to proceed with degraded understanding than to crash.
    """
    return IntentPacket(
        session_theory=inp.session_theory or "Unknown intent — parse error.",
        turn_intent="explore",
        specific_ask=inp.user_message[:200],
        emotional_tenor="unknown",
        clarity_score=0.5,
        needs_clarification=False,
        clarification_question=None,
        inferred_constraints=[],
        budget_hint="standard",
    )


def _serialize_input(inp: UnderstanderInput) -> dict[str, Any]:
    """Lightweight serialization for the agent_calls log."""
    return {
        "user_message": inp.user_message,
        "session_id": str(inp.session_id),
        "has_session_theory": inp.session_theory is not None,
        "num_recent_turns": len(inp.recent_turns),
        "num_known_facts": len(inp.user_facts),
    }
