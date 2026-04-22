"""Live Anthropic smoke test.

Skipped when ``ANTHROPIC_API_KEY`` is not set. Asserts only response shape
(text non-empty, token counts populated) to keep it robust to content drift.
"""

from __future__ import annotations

import os

import pytest

from career_coach.llm.anthropic import AnthropicClient
from career_coach.llm.client import Message


async def test_anthropic_haiku_responds() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live Anthropic call")

    client = AnthropicClient()
    response = await client.complete(
        [
            Message(role="system", content="Reply with a single short sentence."),
            Message(role="user", content="Say hello."),
        ],
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        max_tokens=50,
    )
    assert response.text.strip(), "expected non-empty text"
    assert response.model == "claude-haiku-4-5-20251001"
    assert response.tokens_in is not None
    assert response.tokens_out is not None
