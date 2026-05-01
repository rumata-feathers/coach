"""Synthesizer agent.

Integrates Coach output, Devil's Advocate critique, and Researcher findings
into a single coherent user-facing response. Flow C only.

Prompt encodes §6.4 chart emission gate and §9 reference budget rules.
Retry-on-parse-failure with one re-attempt (same as Coach pattern).

See SPEC_v1.md §9.
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
    Challenge,
    ChartSpec,
    Citation,
    SynthesizedResponse,
    SynthesizerInput,
)

logger = logging.getLogger("career_coach.agents.synthesizer")

_RETRY_PROMPT = (
    "Your previous response was not valid JSON matching the SynthesizedResponse schema. "
    "Return ONLY a JSON object — no prose, no code fences, no think-blocks.\n"
    "Key requirements:\n"
    "- integrated_from must include 'coach'.\n"
    "- Every Citation must have exactly one of 'url' or 'kb_path' (not both, not neither).\n"
    "- chart_specs must be [] or contain exactly 1 ChartSpec with non-empty "
    "  source_citation_indices.\n"
    "- surfaced_tradeoffs must be non-empty if the DA disagreed with the Coach.\n"
    "- response_text must be non-empty."
)

_ESCALATION_RESPONSE = SynthesizedResponse(
    response_text=(
        "I was working on a detailed integrated response for you, but ran into "
        "difficulty producing one that meets the quality standards. "
        "Here is the Coach's core analysis, which remains valid:\n\n"
        "{coach_text}"
    ),
    referenced_facts=[],
    referenced_hypotheses=[],
    referenced_findings=[],
    citations=[],
    surfaced_tradeoffs=[],
    integrated_from=["coach"],
    chart_specs=[],
    proposed_challenge=None,
    uncertainty_flags=["synthesis_failed_fallback_to_coach"],
)


class Synthesizer(Agent):
    """Produces a :class:`SynthesizedResponse` from Coach + DA + Research outputs.

    Args:
        factory: Shared LLM factory.
    """

    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("synthesizer", factory)

    async def run(
        self,
        input_data: SynthesizerInput,
        *,
        turn_id: UUID | None = None,
    ) -> SynthesizedResponse:
        """Synthesise Coach + DA + Research into a single user-facing response.

        Args:
            input_data: Typed synthesizer input.
            turn_id: For agent_calls logging.

        Returns:
            A validated :class:`SynthesizedResponse`. Falls back to a Coach-text
            wrapper on persistent parse failures.
        """
        research_findings = (
            [
                {
                    "claim": f.claim,
                    "confidence": f.confidence,
                    "is_numeric": f.is_numeric,
                    "citations": [
                        {"source_type": c.source_type, "title": c.title,
                         "url": c.url, "kb_path": c.kb_path}
                        for c in f.citations
                    ],
                }
                for f in input_data.research_brief.findings
            ]
            if input_data.research_brief
            else []
        )
        research_caveats = (
            input_data.research_brief.caveats if input_data.research_brief else []
        )
        has_research = bool(research_findings)

        prompt = self.render_prompt(
            "synthesizer.j2",
            coach_response_text=input_data.coach_output.response_text,
            coach_referenced_facts=input_data.coach_output.referenced_facts,
            da_output={
                "agrees_with_coach": input_data.da_output.agrees_with_coach,
                "counter_points": [
                    {
                        "point": cp.point,
                        "reasoning": cp.reasoning,
                        "severity": cp.severity,
                        "source_type": cp.source_type,
                    }
                    for cp in input_data.da_output.counter_points
                ],
                "blind_spots": input_data.da_output.blind_spots,
                "risks": [
                    {"scenario": r.scenario, "likelihood": r.likelihood, "impact": r.impact}
                    for r in input_data.da_output.risks
                ],
            },
            research_findings=research_findings,
            research_caveats=research_caveats,
            user_facts=input_data.user_facts,
            active_hypotheses=[
                {"id": str(h.id), "statement": h.statement, "confidence": h.confidence}
                for h in input_data.active_hypotheses
            ],
            intent_specific_ask=input_data.intent_packet.specific_ask,
            has_research=has_research,
            critic_feedback=input_data.critic_feedback,
            current_date=str(date.today()),
        )

        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        retry_count = 0
        fallback_reason: str | None = None

        response = await self.complete(messages, response_format="json")

        try:
            output = _parse_synthesized(response.text)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as first_exc:
            logger.warning(
                "Synthesizer parse error on first attempt (%s); retrying.", first_exc
            )
            retry_messages = [
                *messages,
                Message(role="assistant", content=response.text or "(empty)"),
                Message(role="user", content=_RETRY_PROMPT),
            ]
            response = await self.complete(retry_messages, response_format="json")
            retry_count = 1
            try:
                output = _parse_synthesized(response.text)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                error_str = str(exc)
                fallback_reason = "retry_exhausted"
                logger.warning(
                    "Synthesizer parse error after retry (%s); using Coach fallback.", exc
                )
                output = _coach_fallback(input_data)

        latency = self.now_ms() - t0
        await self.log_call(
            turn_id=turn_id,
            input_payload=_serialize_input(input_data),
            output_payload=output.model_dump(mode="json"),
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


def _parse_synthesized(raw: str) -> SynthesizedResponse:
    """Strip optional markdown fences, parse JSON, and validate.

    Applies defensive coercions:
    - ``referenced_hypotheses`` entries that are not valid UUIDs are silently
      dropped. The LLM sometimes returns slug strings instead of UUIDs.
    """
    import uuid as _uuid

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    data = json.loads(text)

    # Coerce nested dicts to typed models
    raw_citations = data.get("citations", [])
    citations = [Citation(**c) if isinstance(c, dict) else c for c in raw_citations]

    raw_charts = data.get("chart_specs", [])
    chart_specs: list[ChartSpec] = []
    for c in raw_charts:
        if isinstance(c, dict):
            # Coerce nested AxisSpec dicts
            chart_specs.append(ChartSpec(**c))
        else:
            chart_specs.append(c)

    # Coerce proposed_challenge if dict
    raw_challenge = data.get("proposed_challenge")
    proposed_challenge = (
        Challenge(**raw_challenge)
        if isinstance(raw_challenge, dict)
        else raw_challenge
    )

    # Filter referenced_hypotheses: drop any entry that is not a valid UUID.
    # The model sometimes returns descriptive slugs rather than UUID strings.
    valid_hyp_ids: list[str] = []
    for h in data.get("referenced_hypotheses", []):
        if isinstance(h, str):
            try:
                _uuid.UUID(h)
                valid_hyp_ids.append(h)
            except ValueError:
                pass  # silently drop non-UUID strings

    return SynthesizedResponse(
        response_text=data.get("response_text", ""),
        referenced_facts=data.get("referenced_facts", []),
        referenced_hypotheses=valid_hyp_ids,
        referenced_findings=data.get("referenced_findings", []),
        citations=citations,
        surfaced_tradeoffs=data.get("surfaced_tradeoffs", []),
        integrated_from=data.get("integrated_from", ["coach"]),
        chart_specs=chart_specs,
        proposed_challenge=proposed_challenge,
        uncertainty_flags=data.get("uncertainty_flags", []),
    )


def _coach_fallback(input_data: SynthesizerInput) -> SynthesizedResponse:
    """Minimal fallback that wraps the Coach text when parse repeatedly fails."""
    return SynthesizedResponse(
        response_text=input_data.coach_output.response_text,
        referenced_facts=input_data.coach_output.referenced_facts,
        referenced_hypotheses=list(input_data.coach_output.referenced_hypotheses),
        referenced_findings=[],
        citations=[],
        surfaced_tradeoffs=[],
        integrated_from=["coach"],
        chart_specs=[],
        proposed_challenge=input_data.coach_output.proposed_challenge,
        uncertainty_flags=[
            *input_data.coach_output.uncertainty_flags,
            "synthesis_failed_fallback_to_coach",
        ],
    )


def _serialize_input(inp: SynthesizerInput) -> dict[str, Any]:
    return {
        "specific_ask": inp.intent_packet.specific_ask[:200],
        "num_facts": len(inp.user_facts),
        "num_hypotheses": len(inp.active_hypotheses),
        "has_research": inp.research_brief is not None
        and bool(inp.research_brief.findings),
        "da_agreed": inp.da_output.agrees_with_coach,
        "num_da_counter_points": len(inp.da_output.counter_points),
        "has_critic_feedback": inp.critic_feedback is not None,
    }
