"""In-memory mock LLM client for unit tests.

Provide either a single fixed response or a queue of responses. Every call is
recorded on ``calls`` so tests can assert on model selection, messages, and
parameters without touching a real provider.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from career_coach.llm.client import LLMClient, LLMResponse, Message, ResponseFormat


@dataclass
class RecordedCall:
    """One observed call to :meth:`MockLLMClient.complete`."""

    messages: list[Message]
    model: str
    temperature: float
    max_tokens: int
    response_format: ResponseFormat
    extra_body: dict[str, Any] | None = None


@dataclass
class MockLLMClient(LLMClient):
    """Deterministic client used by agent unit tests."""

    responses: deque[str] = field(default_factory=deque)
    default_response: str = '{"ok": true}'
    calls: list[RecordedCall] = field(default_factory=list)
    delay_ms: int = 0
    """If non-zero, each ``complete`` call sleeps this many milliseconds.

    Useful for latency measurement tests and concurrency stress tests.
    """

    def queue(self, *texts: str) -> None:
        """Queue one or more textual responses to return in order."""
        self.responses.extend(texts)

    async def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: ResponseFormat = "text",
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        self.calls.append(
            RecordedCall(
                messages=list(messages),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )
        )
        text = self.responses.popleft() if self.responses else self.default_response
        return LLMResponse(
            text=text,
            model=model,
            tokens_in=len(" ".join(m.content for m in messages).split()),
            tokens_out=len(text.split()),
            raw={"mock": True},
        )
