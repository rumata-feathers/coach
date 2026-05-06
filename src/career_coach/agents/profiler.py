"""Profiler agent.

Runs asynchronously after each turn to extract new facts and
hypothesis-relevant signals from the conversation. Never blocks the
user-facing response path.

See SPEC §6.5.
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
from career_coach.memory.semantic import SemanticRepo
from career_coach.memory.structured import StructuredFactsRepo
from career_coach.models.agent_io import ProfilerInput, ProfilerOutput
from career_coach.models.user_model import EvidenceDraft, FactUpdate

logger = logging.getLogger("career_coach.agents.profiler")

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY a JSON object matching this schema — "
    "no prose, no code fences, no think-blocks:\n"
    '{"new_facts": [], "fact_updates": [], "hypothesis_evidence": []}'
)


class Profiler(Agent):
    """Extracts facts and hypothesis evidence from a completed turn."""

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("profiler", factory)
        self._structured = StructuredFactsRepo()
        self._semantic = SemanticRepo()

    async def run(
        self,
        input_data: ProfilerInput,
        *,
        turn_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> ProfilerOutput:
        """Call the LLM and return a :class:`ProfilerOutput`.

        Does not persist anything — call :meth:`run_and_save` for that.
        """
        prompt = self.render_prompt(
            "profiler.j2",
            user_facts=input_data.existing_facts,
            user_message=input_data.user_message,
            assistant_message=input_data.assistant_message,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None
        response = await self.complete(messages, response_format="json")

        try:
            output = _parse_output(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "Profiler parse error on first attempt (%s); retrying once.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty response)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            retry_count = 1
            try:
                output = _parse_output(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                error_str = str(exc)
                fallback_reason = "retry_exhausted"
                logger.warning(
                    "Profiler parse error after retry (%s); returning empty output.", exc
                )
                output = ProfilerOutput()

        latency = self.now_ms() - t0

        await self.log_call(
            turn_id=turn_id,
            user_id=user_id,
            input_payload=_serialize_input(input_data),
            output_payload=output.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
            retry_count=retry_count,
            fallback_reason=fallback_reason,
        )
        return output

    async def run_and_save(
        self,
        user_id: UUID,
        turn_id: UUID,
        user_message: str,
        assistant_message: str,
        existing_facts: dict[str, Any],
    ) -> ProfilerOutput:
        """Run extraction and persist results to the structured store.

        Called as a background task from the turn pipeline. Failures are
        logged but do not propagate — the turn result is already committed.
        """
        inp = ProfilerInput(
            turn_id=turn_id,
            user_message=user_message,
            assistant_message=assistant_message,
            existing_facts=existing_facts,
        )
        try:
            output = await self.run(inp, turn_id=turn_id, user_id=user_id)
            all_facts = output.new_facts + output.fact_updates
            if all_facts:
                await self._structured.bulk_upsert(user_id, all_facts)
                logger.info(
                    "Profiler wrote %d fact(s) for user %s (turn %s).",
                    len(all_facts),
                    user_id,
                    turn_id,
                )
            # Queue evidence drafts for later distillation (hypothesis_id = NULL)
            for evidence in output.hypothesis_evidence:
                await self._semantic.queue_evidence(evidence, turn_id=turn_id)
            if output.hypothesis_evidence:
                logger.info(
                    "Profiler queued %d evidence draft(s) for user %s (turn %s).",
                    len(output.hypothesis_evidence),
                    user_id,
                    turn_id,
                )
                # Auto-trigger distillation whenever new evidence is queued.
                # This runs inside the existing background task so it never
                # blocks the user-facing response. The job returns early if
                # the evidence queue is empty, making it safe to call every turn.
                await self._run_distillation(user_id)
        except Exception as exc:
            logger.error("Profiler background task failed: %s", exc, exc_info=True)
            return ProfilerOutput()
        return output

    async def _run_distillation(self, user_id: UUID) -> None:
        """Trigger distillation in-process after evidence is queued.

        Imported lazily to avoid a module-level circular dependency between
        the agents and jobs packages.  Failures are logged and swallowed so
        a broken distillation job never kills the profiler background task.
        """
        try:
            from career_coach.jobs.distillation import run_distillation

            result = await run_distillation(user_id, factory=self._factory)
            logger.info(
                "Auto-distillation for user %s: processed=%d created=%d matched=%d deduped=%d",
                user_id,
                result.get("processed", 0),
                result.get("created", 0),
                result.get("matched", 0),
                result.get("deduped", 0),
            )
        except Exception as exc:
            logger.error(
                "Auto-distillation failed for user %s: %s", user_id, exc, exc_info=True
            )


# ---- helpers ---------------------------------------------------------------


def _parse_output(raw: str) -> ProfilerOutput:
    """Strip fences, parse JSON, validate against the schema."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)
    new_facts = [FactUpdate(**f) for f in data.get("new_facts", [])]
    fact_updates = [FactUpdate(**f) for f in data.get("fact_updates", [])]
    hypothesis_evidence = [
        EvidenceDraft(**e) for e in data.get("hypothesis_evidence", [])
    ]
    return ProfilerOutput(
        new_facts=new_facts,
        fact_updates=fact_updates,
        hypothesis_evidence=hypothesis_evidence,
    )


def _serialize_input(inp: ProfilerInput) -> dict[str, Any]:
    return {
        "turn_id": str(inp.turn_id),
        "user_message_len": len(inp.user_message),
        "assistant_message_len": len(inp.assistant_message),
        "num_existing_facts": len(inp.existing_facts),
    }
