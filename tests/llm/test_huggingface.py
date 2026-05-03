"""Unit tests for HuggingFaceClient think-block stripping logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from career_coach.llm.huggingface import HuggingFaceClient
from career_coach.llm.client import Message


def _make_client() -> HuggingFaceClient:
    return HuggingFaceClient(client=MagicMock(spec=AsyncOpenAI))


def _mock_response(content: str, reasoning_content: str | None = None) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.model_extra = (
        {"reasoning_content": reasoning_content} if reasoning_content is not None else {}
    )
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


async def _complete(
    client: HuggingFaceClient,
    content: str,
    reasoning_content: str | None = None,
) -> str:
    mock_create = AsyncMock(return_value=_mock_response(content, reasoning_content))
    client._client.chat.completions.create = mock_create  # type: ignore[attr-defined]
    result = await client.complete(
        [Message(role="user", content="hi")],
        model="test-model",
    )
    return result.text


async def test_no_think_block_passes_through() -> None:
    """Plain response with no think block is returned unchanged."""
    client = _make_client()
    text = await _complete(client, '{"ok": true}')
    assert text == '{"ok": true}'


async def test_think_block_stripped_when_answer_follows() -> None:
    """Normal path: answer appears after </think>."""
    client = _make_client()
    raw = "<think>Let me think...</think>\n{\"answer\": 42}"
    text = await _complete(client, raw)
    assert text == '{"answer": 42}'


async def test_reasoning_content_json_rescued_when_content_empty() -> None:
    """HF Novita path: content='', reasoning_content has the JSON at the end."""
    client = _make_client()
    # Mirrors real DeepSeek-V4-Flash thinking_mode=thinking API response
    reasoning = 'Let me think about this. The answer is {"response_text": "Hi!", "referenced_facts": []}'
    text = await _complete(client, "", reasoning_content=reasoning)
    assert '"response_text"' in text
    assert '"Hi!"' in text


async def test_think_block_json_rescued_when_nothing_after() -> None:
    """Inline think-block path: model put JSON inside <think>; adapter extracts it."""
    client = _make_client()
    raw = (
        "<think>I'll produce the JSON here:\n"
        '{"response_text": "Hello!", "referenced_facts": []}\n'
        "That should be good.</think>"
    )
    text = await _complete(client, raw)
    # The last {...} in the think block should be extracted
    assert '"response_text"' in text
    assert '"Hello!"' in text


async def test_think_block_picks_last_json_object() -> None:
    """When think block has multiple JSON objects, the last one is used."""
    client = _make_client()
    raw = (
        '<think>First attempt: {"bad": true}\n'
        'Second attempt: {"good": true}\n'
        "</think>"
    )
    text = await _complete(client, raw)
    assert '"good"' in text
    assert '"bad"' not in text


async def test_empty_response_after_think_with_no_json_inside() -> None:
    """Edge case: think block has no JSON → empty string returned (let caller handle)."""
    client = _make_client()
    raw = "<think>Just prose, no JSON here.</think>"
    text = await _complete(client, raw)
    assert text == ""
