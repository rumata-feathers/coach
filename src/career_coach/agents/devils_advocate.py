"""Devil's Advocate agent.

Applies structured contrarian pressure to the Coach's response. Grounds
counter-points in the user's profile and research findings where possible.
Honest agreement (``agrees_with_coach=True``) is a first-class outcome.

See SPEC_v1.md §8.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from career_coach.agents.base import Agent
from career_coach.llm.client import Message
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import (
    CounterPoint,
    DevilsAdvocateInput,
    DevilsAdvocateOutput,
    Risk,
)

logger = logging.getLogger("career_coach.agents.devils_advocate")

_RETRY_PROMPT = (
    "Your previous response was not valid JSON matching the DevilsAdvocateOutput schema. "
    "Return ONLY a JSON object — no prose, no code fences, no think-blocks.\n"
    "Required fields: counter_points (list), blind_spots (list), risks (list), "
    "agrees_with_coach (bool).\n"
    "Every counter_point MUST be an object with ALL FOUR of these fields:\n"
    '  "point": "...",\n'
    '  "reasoning": "one sentence explaining WHY this matters for this user",\n'
    '  "severity": "low" | "med" | "high",\n'
    '  "source_type": "user_profile" | "research" | "general"\n'
    "Every risk MUST be an object (NOT a plain string) with ALL THREE fields:\n"
    '  "scenario": "concrete description of what could go wrong",\n'
    '  "likelihood": "low" | "med" | "high",\n'
    '  "impact": "low" | "med" | "high"\n'
    "If agrees_with_coach is false you need ≥2 counter_points.\n"
    "Example risk object: "
    '{"scenario": "The startup could fail within 2 years", '
    '"likelihood": "med", "impact": "high"}'
)

# Honest-agreement fallback: returned when the model's agrees_with_coach path
# produces a structurally valid but empty response after a parse failure.
_EMPTY_AGREEMENT = DevilsAdvocateOutput(
    counter_points=[],
    blind_spots=[],
    risks=[],
    agrees_with_coach=True,
)


class DevilsAdvocate(Agent):
    """Applies structured critical pressure to :class:`CoachOutput`.

    Args:
        factory: Shared LLM factory.
    """

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("devils_advocate", factory)

    async def run(
        self,
        input_data: DevilsAdvocateInput,
        *,
        turn_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> DevilsAdvocateOutput:
        """Critique the Coach's response and return a structured DA output.

        Args:
            input_data: Typed DA input including coach output, optional research
                brief, user facts, and intent.
            turn_id: For agent_calls logging.
            user_id: Owning user; stored in agent_calls for per-user token aggregation.

        Returns:
            A validated :class:`DevilsAdvocateOutput`. Falls back to an honest
            agreement output on persistent parse failures (safe default — the
            Synthesizer will still have Coach + Research to work with).
        """
        research_findings = (
            [
                {
                    "claim": f.claim,
                    "confidence": f.confidence,
                    "is_numeric": f.is_numeric,
                    "citations": [
                        {"source_type": c.source_type, "title": c.title}
                        for c in f.citations
                    ],
                }
                for f in input_data.research_brief.findings
            ]
            if input_data.research_brief
            else []
        )

        prompt = self.render_prompt(
            "devils_advocate.j2",
            coach_response_text=input_data.coach_output.response_text,
            coach_referenced_facts=input_data.coach_output.referenced_facts,
            user_facts=input_data.user_facts,
            active_hypotheses=[
                {
                    "id": str(h.id),
                    "statement": h.statement,
                    "confidence": h.confidence,
                }
                for h in input_data.active_hypotheses
            ],
            intent_specific_ask=input_data.intent_packet.specific_ask,
            research_findings=research_findings,
            current_date=str(date.today()),
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None

        response = await self.complete(messages, response_format="json")

        try:
            output = _parse_da_output(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "DA parse error on first attempt (%s); retrying.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            retry_count = 1
            try:
                output = _parse_da_output(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                error_str = str(exc)
                fallback_reason = "retry_exhausted"
                logger.warning(
                    "DA parse error after retry (%s); falling back to honest agreement.", exc
                )
                output = _EMPTY_AGREEMENT

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


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------


def _parse_da_output(raw: str) -> DevilsAdvocateOutput:
    """Strip optional markdown fences, parse JSON, and validate.

    Applies defensive coercions for common model output deviations:
    - CounterPoint dict missing ``reasoning`` → filled from ``point`` text.
    - Risk element is a plain string → wrapped into a Risk dict with defaults.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)

    # Coerce counter_points: fill missing "reasoning" field when absent.
    raw_cps = data.get("counter_points", [])
    counter_points: list[CounterPoint] = []
    for cp in raw_cps:
        if isinstance(cp, dict):
            if "reasoning" not in cp:
                # Model omitted reasoning — synthesise from point text.
                cp = dict(cp, reasoning=cp.get("point", "No reasoning provided."))
            counter_points.append(CounterPoint(**cp))
        # Non-dict entries are silently skipped (model output error).

    # Coerce risks: plain strings become Risk objects with neutral defaults.
    raw_risks = data.get("risks", [])
    risks: list[Risk] = []
    for r in raw_risks:
        if isinstance(r, str):
            risks.append(Risk(scenario=r, likelihood="med", impact="med"))
        elif isinstance(r, dict):
            risks.append(Risk(**r))
        # Non-dict, non-string entries silently skipped.

    return DevilsAdvocateOutput(
        counter_points=counter_points,
        blind_spots=data.get("blind_spots", []),
        risks=risks,
        agrees_with_coach=data.get("agrees_with_coach", False),
    )


def _serialize_input(inp: DevilsAdvocateInput) -> dict[str, Any]:
    return {
        "specific_ask": inp.intent_packet.specific_ask[:200],
        "num_facts": len(inp.user_facts),
        "num_hypotheses": len(inp.active_hypotheses),
        "has_research": inp.research_brief is not None and bool(inp.research_brief.findings),
        "coach_response_len": len(inp.coach_output.response_text),
    }
