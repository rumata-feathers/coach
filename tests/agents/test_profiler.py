"""Unit tests for :class:`Profiler` with :class:`MockLLMClient`."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from career_coach.agents.profiler import Profiler
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.models.agent_io import ProfilerInput, ProfilerOutput

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _make_input() -> ProfilerInput:
    return ProfilerInput(
        turn_id=uuid4(),
        user_message="I'm 19 and I really enjoy maths and logic puzzles.",
        assistant_message="That's a great foundation for quantitative fields like CS or economics.",
        existing_facts={"location": "London", "education_stage": "undergrad"},
    )


def _good_output_json() -> str:
    return json.dumps(
        {
            "new_facts": [
                {"key": "age", "value": 19, "confidence": 0.95, "source": "user_stated"},
                {
                    "key": "interests",
                    "value": ["maths", "logic_puzzles"],
                    "confidence": 0.9,
                    "source": "user_stated",
                },
            ],
            "fact_updates": [],
            "hypothesis_evidence": [
                {
                    "source_type": "user_statement",
                    "excerpt": "I really enjoy maths and logic puzzles.",
                    "weight": 0.8,
                    "hypothesis_hint": "analytical_aptitude",
                }
            ],
        }
    )


async def test_profiler_returns_valid_output() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    profiler = Profiler(_make_factory(mock))
    output = await profiler.run(_make_input())

    assert isinstance(output, ProfilerOutput)
    assert len(output.new_facts) == 2
    assert output.new_facts[0].key == "age"
    assert output.new_facts[0].value == 19
    assert len(output.hypothesis_evidence) == 1
    assert output.hypothesis_evidence[0].weight == pytest.approx(0.8)


async def test_profiler_empty_output_on_malformed_json() -> None:
    mock = MockLLMClient(default_response="not json at all")
    profiler = Profiler(_make_factory(mock))
    output = await profiler.run(_make_input())

    assert isinstance(output, ProfilerOutput)
    assert output.new_facts == []
    assert output.fact_updates == []
    assert output.hypothesis_evidence == []


async def test_profiler_strips_markdown_fences() -> None:
    fenced = "```json\n" + _good_output_json() + "\n```"
    mock = MockLLMClient(default_response=fenced)
    profiler = Profiler(_make_factory(mock))
    output = await profiler.run(_make_input())

    assert isinstance(output, ProfilerOutput)
    assert len(output.new_facts) == 2


async def test_profiler_uses_correct_model() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    profiler = Profiler(_make_factory(mock))
    await profiler.run(_make_input())

    assert mock.calls[0].model == "XiaomiMiMo/MiMo-V2-Flash"


async def test_profiler_prompt_contains_user_message() -> None:
    mock = MockLLMClient(default_response=_good_output_json())
    profiler = Profiler(_make_factory(mock))
    await profiler.run(_make_input())

    prompt = mock.calls[0].messages[0].content
    assert "maths and logic puzzles" in prompt


async def test_profiler_empty_arrays_are_valid() -> None:
    empty_json = json.dumps(
        {"new_facts": [], "fact_updates": [], "hypothesis_evidence": []}
    )
    mock = MockLLMClient(default_response=empty_json)
    profiler = Profiler(_make_factory(mock))
    output = await profiler.run(_make_input())

    assert output.new_facts == []
    assert output.fact_updates == []
    assert output.hypothesis_evidence == []


async def test_profiler_retries_on_empty_response() -> None:
    """First LLM call returns empty string; retry returns valid JSON."""
    mock = MockLLMClient()
    mock.queue("", _good_output_json())  # empty first, valid second
    profiler = Profiler(_make_factory(mock))

    output = await profiler.run(_make_input())

    assert len(mock.calls) == 2, "Expected exactly 2 LLM calls (original + retry)"
    assert isinstance(output, ProfilerOutput)
    assert len(output.new_facts) == 2
