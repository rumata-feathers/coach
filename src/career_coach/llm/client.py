"""Core LLM interface shared by every provider implementation.

Every provider (Anthropic, Hugging Face, future OpenAI, …) exposes a single
async method :meth:`LLMClient.complete` that accepts a list of
:class:`Message` objects and returns a :class:`LLMResponse`. Agents depend on
this protocol — not on any SDK — so providers are swappable via configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]
ResponseFormat = Literal["text", "json"]


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message.

    ``content`` is a plain string in v0 — no multipart content blocks. Provider
    adapters translate to whatever their SDK expects.
    """

    role: Role
    content: str


@dataclass(slots=True)
class LLMResponse:
    """Structured response shared by all providers.

    ``text`` is the assistant's reply. ``tokens_in`` / ``tokens_out`` are used
    for the ``agent_calls`` observability log and may be ``None`` if the
    provider does not expose token counts.
    """

    text: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Protocol every provider implementation must satisfy."""

    async def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: ResponseFormat = "text",
    ) -> LLMResponse:
        """Run a chat completion and return the assistant's reply.

        Implementations must not raise for normal "model said no" outcomes;
        they must raise for transport / auth / provider errors so the turn
        loop can surface them in ``agent_calls.error``.
        """
        ...
