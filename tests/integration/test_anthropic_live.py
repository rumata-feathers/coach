"""Live HuggingFace smoke test.

Skipped when ``HUGGINGFACE_API_TOKEN`` is not set. Asserts only response shape
(text non-empty, token counts populated) to keep it robust to content drift.
"""

from __future__ import annotations

import os

import pytest

from career_coach.llm.client import Message
from career_coach.llm.huggingface import HuggingFaceClient


async def test_huggingface_responds() -> None:
    if not os.environ.get("HUGGINGFACE_API_TOKEN"):
        pytest.skip("HUGGINGFACE_API_TOKEN not set — skipping live HuggingFace call")

    client = HuggingFaceClient()
    response = await client.complete(
        [
            Message(role="system", content="Reply with a single short sentence."),
            Message(role="user", content="Say hello."),
        ],
        model="Qwen/Qwen3-32B",
        temperature=0.0,
        max_tokens=50,
    )
    assert response.text.strip(), "expected non-empty text"
    assert response.tokens_in is not None
    assert response.tokens_out is not None
