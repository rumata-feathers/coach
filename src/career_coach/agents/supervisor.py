"""Supervisor agent.

Cross-cutting safety and correctness monitor. Runs on every flow as the final
pre-response check. Three checks: fact_contradiction, off_topic, unsafe.

Fails *open* on parse errors (returns "pass") so a broken Supervisor never
blocks the user. Early-returns "pass" for empty responses to stay within the
p50 < 500 ms latency budget.

See SPEC_v1.md §10.
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
from career_coach.models.agent_io import SupervisorEvent, SupervisorInput

logger = logging.getLogger("career_coach.agents.supervisor")

_PASS_EVENT = SupervisorEvent(action="pass")

_RETRY_PROMPT = (
    "Your previous response was not valid JSON matching the SupervisorEvent schema. "
    "Return ONLY a JSON object — no prose, no code fences, no think-blocks.\n"
    "Required shape:\n"
    '{"event_type": null|"fact_contradiction"|"off_topic"|"unsafe", '
    '"severity": null|"low"|"med"|"high", '
    '"details": null|"<one sentence>", '
    '"action": "pass"|"warn"|"retry"|"block", '
    '"scripted_override": null|"<text>"}'
)


class Supervisor(Agent):
    """Final pre-response safety/correctness check.

    Returns a :class:`SupervisorEvent` with an ``action`` that the pipeline
    uses to decide what to do with the response.

    Args:
        factory: Shared LLM factory.
    """

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("supervisor", factory)

    async def run(
        self,
        input_data: SupervisorInput,
        *,
        turn_id: UUID | None = None,
    ) -> SupervisorEvent:
        """Run the three-check safety monitor.

        Args:
            input_data: Supervisor input including user message + response.
            turn_id: For agent_calls logging.

        Returns:
            A :class:`SupervisorEvent`. On parse failure, returns a ``pass``
            event (fail-open so a broken Supervisor never blocks the user).
        """
        # Early-return pass for empty responses (latency discipline).
        if not input_data.response_text.strip():
            logger.debug("Supervisor: empty response_text — returning pass immediately.")
            return _PASS_EVENT

        prompt = self.render_prompt(
            "supervisor.j2",
            user_message=input_data.user_message,
            response_text=input_data.response_text,
            user_facts=input_data.user_facts,
            specific_ask=input_data.specific_ask,
            flow_used=input_data.flow_used,
        )

        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None
        # Initialise so log_call is always reached even if complete() raises.
        event = _PASS_EVENT
        tokens_in: int | None = None
        tokens_out: int | None = None

        try:
            response = await self.complete(messages, response_format="json")
            tokens_in = response.tokens_in
            tokens_out = response.tokens_out
            try:
                event = _parse_event(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
                logger.warning(
                    "Supervisor parse error on first attempt (%s); retrying.", first_exc
                )
                retry_messages = [
                    *messages,
                    Message(role="assistant", content=response.text or "(empty)"),
                    Message(role="user", content=_RETRY_PROMPT),
                ]
                response = await self.complete(retry_messages, response_format="json")
                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                retry_count = 1
                try:
                    event = _parse_event(response.text)
                except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                    error_str = str(exc)
                    fallback_reason = "retry_exhausted"
                    logger.warning(
                        "Supervisor parse error after retry (%s); failing open with pass.", exc
                    )
                    event = _PASS_EVENT
        except Exception as llm_exc:
            # LLM call itself failed (e.g. BadRequestError). Fail open — a
            # broken Supervisor must never block the pipeline. Always log so
            # the failure is visible in agent_calls.
            error_str = str(llm_exc)
            fallback_reason = "llm_exception"
            logger.warning(
                "Supervisor LLM call raised (%s); failing open with pass.", llm_exc
            )

        latency = self.now_ms() - t0
        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=event.model_dump(mode="json"),
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            error=error_str,
            retry_count=retry_count,
            fallback_reason=fallback_reason,
        )
        return event


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------


def _parse_event(raw: str) -> SupervisorEvent:
    """Strip optional markdown fences, parse JSON, validate."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    return SupervisorEvent(**data)


def _serialize_input(inp: SupervisorInput) -> dict[str, Any]:
    return {
        "flow_used": inp.flow_used,
        "specific_ask": inp.specific_ask[:200],
        "num_facts": len(inp.user_facts),
        "response_len": len(inp.response_text),
        "user_message_len": len(inp.user_message),
    }
