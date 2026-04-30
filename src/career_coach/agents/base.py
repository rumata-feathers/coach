"""Agent base class and utilities.

Every agent:
  1. Has a ``name`` (matches its key in ``config/models.yaml``).
  2. Owns an :class:`LLMClient` and an :class:`AgentLLMConfig`.
  3. Loads its Jinja2 prompt template from ``config/prompts/<name>.j2``.
  4. Exposes a typed ``run()`` method (defined in concrete subclasses).
  5. Logs every call to ``agent_calls`` via :meth:`log_call`.

Prompts MUST live in ``config/prompts/*.j2``. Hardcoding prompt text in
Python is explicitly banned (see ``CLAUDE.md``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from career_coach.db import get_pool
from career_coach.llm.client import LLMClient, LLMResponse
from career_coach.llm.factory import AgentLLMConfig, LLMFactory

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"


def _build_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=select_autoescape([]),  # no HTML escaping for LLM prompts
        undefined=StrictUndefined,  # fail loudly on missing variables
        trim_blocks=True,
        lstrip_blocks=True,
    )


_JINJA_ENV = _build_jinja_env()


async def log_agent_call(
    *,
    agent_name: str,
    model_used: str | None,
    turn_id: UUID | None,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any] | None,
    latency_ms: int,
    tokens_in: int | None,
    tokens_out: int | None,
    error: str | None = None,
    retry_count: int = 0,
    fallback_reason: str | None = None,
) -> None:
    """Write a row to ``agent_calls``.

    This standalone function allows non-LLM components (e.g. the web search
    adapter) to log calls without subclassing :class:`Agent`. :meth:`Agent.log_call`
    delegates to this function.

    Payloads are round-tripped through JSON to coerce non-serialisable types
    (UUID, datetime, …) to strings before asyncpg's JSONB codec sees them.
    """
    pool = await get_pool()
    safe_input = _jsonify(input_payload)
    safe_output = _jsonify(output_payload) if output_payload is not None else None
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_calls
                (turn_id, agent_name, model_used, input_payload, output_payload,
                 latency_ms, tokens_in, tokens_out, error,
                 retry_count, fallback_reason)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            turn_id,
            agent_name,
            model_used,
            safe_input,
            safe_output,
            latency_ms,
            tokens_in,
            tokens_out,
            error,
            retry_count,
            fallback_reason,
        )


class Agent:
    """Base class for all career-coach agents.

    Concrete subclasses implement ``run()`` with typed input/output. They call
    :meth:`complete` (which wraps :meth:`LLMClient.complete`) and
    :meth:`log_call` (which writes to ``agent_calls``).
    """

    def __init__(self, name: str, factory: LLMFactory) -> None:
        """Initialise the agent.

        Args:
            name: Key used in ``config/models.yaml`` (e.g. ``"understander"``).
            factory: Shared factory that caches provider clients.
        """
        self.name = name
        self._factory = factory
        self._llm_config: AgentLLMConfig = factory.config_for(name)
        self._client: LLMClient = factory.client_for(name)

    def render_prompt(self, template_name: str, **ctx: Any) -> str:
        """Render a Jinja2 template from ``config/prompts/``.

        Args:
            template_name: Filename relative to ``config/prompts/``,
                e.g. ``"understander.j2"``.
            **ctx: Variables injected into the template.

        Raises:
            jinja2.UndefinedError: If the template references a missing variable.
        """
        template = _JINJA_ENV.get_template(template_name)
        return template.render(**ctx)

    async def complete(
        self,
        messages: Any,
        *,
        response_format: Any = "text",
    ) -> LLMResponse:
        """Run an LLM call using this agent's configured model and defaults."""
        return await self._client.complete(
            messages,
            model=self._llm_config.model,
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            response_format=response_format,
            extra_body=self._llm_config.extra_body,
        )

    async def log_call(
        self,
        *,
        turn_id: UUID | None,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any] | None,
        latency_ms: int,
        tokens_in: int | None,
        tokens_out: int | None,
        error: str | None = None,
        retry_count: int = 0,
        fallback_reason: str | None = None,
    ) -> None:
        """Append a row to ``agent_calls`` for observability.

        Delegates to the module-level :func:`log_agent_call` function.

        Args:
            retry_count: Number of retry attempts made (0 = first attempt succeeded).
            fallback_reason: Populated when a fallback path was used, e.g.
                ``"empty_llm_response"``, ``"schema_validation_failed"``,
                ``"retry_exhausted"``.  ``None`` means the happy path was taken.
        """
        await log_agent_call(
            agent_name=self.name,
            model_used=self._llm_config.model,
            turn_id=turn_id,
            input_payload=input_payload,
            output_payload=output_payload,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            error=error,
            retry_count=retry_count,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def now_ms() -> int:
        """Wall-clock time in milliseconds for latency measurement.

        Uses ``perf_counter_ns`` so two calls within the same millisecond
        still produce a measurable non-zero difference.
        """
        return time.perf_counter_ns() // 1_000_000


def _jsonify(data: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON to coerce UUID/datetime to strings."""
    result: dict[str, Any] = json.loads(json.dumps(data, default=str))
    return result
