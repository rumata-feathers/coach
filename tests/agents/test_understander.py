"""Unit tests for :class:`Understander`.

All LLM calls are intercepted by :class:`MockLLMClient` — no real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from career_coach.agents.understander import Understander, _fallback_packet, _parse_intent_packet
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import UnderstanderInput
from career_coach.models.intent import IntentPacket

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _valid_packet_json(**overrides: object) -> str:
    base: dict[str, object] = {
        "session_theory": "User is exploring career options.",
        "turn_intent": "explore",
        "specific_ask": "Help me think about whether economics suits me.",
        "emotional_tenor": "curious",
        "clarity_score": 0.8,
        "needs_clarification": False,
        "clarification_question": None,
        "inferred_constraints": ["undergrad", "UK"],
        "budget_hint": "standard",
    }
    base.update(overrides)
    return json.dumps(base)


def _make_input() -> UnderstanderInput:
    return UnderstanderInput(
        user_message="I'm not sure economics is for me.",
        session_id=uuid4(),
        session_theory=None,
        recent_turns=[],
        user_facts={"age": 19, "location": "London"},
    )


async def test_understander_returns_valid_intent_packet() -> None:
    mock = MockLLMClient(default_response=_valid_packet_json())
    understander = Understander(_make_factory(mock))

    packet = await understander.run(_make_input())

    assert isinstance(packet, IntentPacket)
    assert packet.turn_intent == "explore"
    assert packet.budget_hint == "standard"
    assert not packet.needs_clarification


async def test_understander_uses_correct_model() -> None:
    mock = MockLLMClient(default_response=_valid_packet_json())
    understander = Understander(_make_factory(mock))

    await understander.run(_make_input())

    assert len(mock.calls) == 1
    assert mock.calls[0].model == "Qwen/Qwen3-32B"


async def test_understander_requests_json_format() -> None:
    mock = MockLLMClient(default_response=_valid_packet_json())
    understander = Understander(_make_factory(mock))

    await understander.run(_make_input())

    assert mock.calls[0].response_format == "json"


async def test_understander_strips_markdown_fences() -> None:
    fenced = "```json\n" + _valid_packet_json() + "\n```"
    mock = MockLLMClient(default_response=fenced)
    understander = Understander(_make_factory(mock))

    packet = await understander.run(_make_input())

    assert isinstance(packet, IntentPacket)
    assert packet.turn_intent == "explore"


async def test_understander_returns_fallback_on_malformed_json() -> None:
    mock = MockLLMClient(default_response="this is not json at all")
    understander = Understander(_make_factory(mock))

    packet = await understander.run(_make_input())

    # Fallback should not crash and should pass Pydantic validation.
    assert isinstance(packet, IntentPacket)
    assert not packet.needs_clarification


async def test_understander_returns_fallback_on_schema_mismatch() -> None:
    # Valid JSON but missing required fields.
    mock = MockLLMClient(default_response='{"turn_intent": "explore"}')
    understander = Understander(_make_factory(mock))

    packet = await understander.run(_make_input())

    assert isinstance(packet, IntentPacket)


async def test_understander_clarification_packet_is_valid() -> None:
    clarification_json = _valid_packet_json(
        clarity_score=0.3,
        needs_clarification=True,
        clarification_question="What specific aspect of economics concerns you?",
        budget_hint="standard",
    )
    mock = MockLLMClient(default_response=clarification_json)
    understander = Understander(_make_factory(mock))

    packet = await understander.run(_make_input())

    assert packet.needs_clarification
    assert packet.clarification_question is not None


async def test_parse_intent_packet_parses_valid_json() -> None:
    raw = _valid_packet_json()
    packet = _parse_intent_packet(raw)
    assert packet.turn_intent == "explore"


def test_fallback_packet_is_always_valid() -> None:
    inp = _make_input()
    packet = _fallback_packet(inp)
    assert isinstance(packet, IntentPacket)
    assert not packet.needs_clarification


async def test_understander_integration_with_real_prompt() -> None:
    """Verify the Jinja2 template renders without UndefinedError."""
    mock = MockLLMClient(default_response=_valid_packet_json())
    understander = Understander(_make_factory(mock))

    inp = UnderstanderInput(
        user_message="Should I study law or economics?",
        session_id=uuid4(),
        session_theory="User weighing two options.",
        recent_turns=[],
        user_facts={"age": 19, "education_stage": "undergrad"},
    )
    await understander.run(inp)
    # Template rendered without error, LLM was called once.
    assert len(mock.calls) == 1
    # The prompt should contain the user message.
    prompt_content = mock.calls[0].messages[0].content
    assert "Should I study law or economics?" in prompt_content
