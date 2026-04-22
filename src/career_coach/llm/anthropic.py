"""Anthropic provider implementation of :class:`LLMClient`.

Thin wrapper around the official ``anthropic`` async SDK. Translates between
our provider-agnostic :class:`Message` shape and Anthropic's
``system`` + ``messages`` split (Anthropic expects the system prompt as a
top-level argument, not a message with ``role="system"``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock

from career_coach.llm.client import LLMClient, LLMResponse, Message, ResponseFormat


class AnthropicClient(LLMClient):
    """LLM client backed by the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, client: AsyncAnthropic | None = None):
        """Create a client.

        Either pass ``api_key`` (picked up from env if ``None``) or inject a
        pre-built ``AsyncAnthropic`` — the latter is convenient for tests.
        """
        if client is not None:
            self._client = client
        else:
            self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: ResponseFormat = "text",
    ) -> LLMResponse:
        system_prompt, chat_messages = _split_system(messages)
        if response_format == "json":
            # Anthropic has no strict JSON mode; prompts must instruct JSON
            # explicitly. We append a terse reminder so callers who forget in
            # the template still get usable output.
            system_prompt = (
                (system_prompt or "") + "\n\nRespond with a single JSON object and no prose."
            ).strip()

        response = await self._client.messages.create(
            model=model,
            system=system_prompt or "",
            messages=[
                {
                    "role": cast(Literal["user", "assistant"], m.role),
                    "content": m.content,
                }
                for m in chat_messages
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = "".join(block.text for block in response.content if isinstance(block, TextBlock))

        return LLMResponse(
            text=text,
            model=model,
            tokens_in=getattr(response.usage, "input_tokens", None),
            tokens_out=getattr(response.usage, "output_tokens", None),
            raw=_safe_dump(response),
        )


def _split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """Pull out the first system message and return ``(system, rest)``.

    Multiple system messages are concatenated with blank lines. Non-system
    messages keep their original order.
    """
    system_parts: list[str] = []
    rest: list[Message] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            rest.append(msg)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def _safe_dump(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK response to a plain dict for logging."""
    if hasattr(obj, "model_dump"):
        try:
            return dict(obj.model_dump())
        except Exception:  # pragma: no cover — log-only path
            return {}
    return {}
