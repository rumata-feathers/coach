"""Smoke tests for :class:`MockLLMClient`.

The mock is the backbone of every agent unit test — if these break, every
other agent test is suspect.
"""

from __future__ import annotations

from career_coach.llm.client import Message
from career_coach.llm.mock import MockLLMClient


async def test_mock_returns_default_when_queue_empty() -> None:
    mock = MockLLMClient(default_response='{"hello": "world"}')
    response = await mock.complete(
        [Message(role="user", content="ping")],
        model="model-x",
    )
    assert response.text == '{"hello": "world"}'
    assert response.model == "model-x"


async def test_mock_queue_is_consumed_in_order() -> None:
    mock = MockLLMClient()
    mock.queue("first", "second")

    r1 = await mock.complete([Message(role="user", content="a")], model="m")
    r2 = await mock.complete([Message(role="user", content="b")], model="m")
    r3 = await mock.complete([Message(role="user", content="c")], model="m")

    assert (r1.text, r2.text) == ("first", "second")
    # Falls back to default once queue is empty.
    assert r3.text == mock.default_response


async def test_mock_records_calls() -> None:
    mock = MockLLMClient()
    await mock.complete(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        model="m",
        temperature=0.3,
        max_tokens=123,
        response_format="json",
    )
    assert len(mock.calls) == 1
    call = mock.calls[0]
    assert call.model == "m"
    assert call.temperature == 0.3
    assert call.max_tokens == 123
    assert call.response_format == "json"
    assert [m.role for m in call.messages] == ["system", "user"]
