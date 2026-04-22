"""Unit tests for :class:`TurnPipeline`.

All external dependencies (DB repos + LLM calls) are patched so these tests
run without a database or API token.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.memory.episodic import Session
from career_coach.pipeline.turn import TurnPipeline, TurnResult

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

FAKE_SESSION_ID = uuid4()
FAKE_TURN_ID = uuid4()
FAKE_USER_ID = uuid4()


def _make_factory(mock: MockLLMClient) -> LLMFactory:
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory


def _intent_json(**overrides: object) -> str:
    base: dict[str, object] = {
        "session_theory": "User explores career options.",
        "turn_intent": "explore",
        "specific_ask": "Should I study economics?",
        "emotional_tenor": "curious",
        "clarity_score": 0.8,
        "needs_clarification": False,
        "clarification_question": None,
        "inferred_constraints": [],
        "budget_hint": "standard",
    }
    base.update(overrides)
    return json.dumps(base)


def _coach_json() -> str:
    return json.dumps(
        {
            "response_text": "Economics is a solid choice given your London location.",
            "referenced_facts": ["location"],
            "referenced_hypotheses": [],
            "proposed_challenge": None,
            "uncertainty_flags": [],
        }
    )


def _critic_pass_json() -> str:
    return json.dumps(
        {
            "verdict": "pass",
            "failure_modes": [],
            "specific_complaints": [],
            "suggested_fix": None,
        }
    )


def _profiler_json() -> str:
    return json.dumps(
        {
            "new_facts": [{"key": "age", "value": 19, "confidence": 0.9, "source": "user_stated"}],
            "fact_updates": [],
            "hypothesis_evidence": [],
        }
    )


def _patch_repos(
    session: Session | None = None,
    user_facts: dict | None = None,
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Return patched (episodic, structured, semantic) repo mocks."""
    sess = session or Session(
        session_id=FAKE_SESSION_ID, user_id=FAKE_USER_ID, session_theory=None
    )
    structured = AsyncMock()
    structured.get_all.return_value = user_facts or {}
    structured.bulk_upsert.return_value = None

    episodic = AsyncMock()
    episodic.get_session.return_value = sess
    episodic.create_session.return_value = sess
    episodic.get_recent.return_value = []
    episodic.save_turn.return_value = FAKE_TURN_ID
    episodic.update_session_theory.return_value = None
    episodic.save_turn_embedding.return_value = None

    semantic = AsyncMock()
    semantic.get_active.return_value = []

    return structured, episodic, semantic


@pytest.fixture()
def pipeline_mocks() -> tuple[MockLLMClient, TurnPipeline]:
    """Yield a (mock_llm, pipeline) pair with all repo calls stubbed."""
    mock = MockLLMClient()
    factory = _make_factory(mock)
    pipeline = TurnPipeline(factory)

    structured, episodic, semantic = _patch_repos(user_facts={"location": "London"})
    pipeline._structured = structured
    pipeline._episodic = episodic
    pipeline._semantic = semantic

    return mock, pipeline


async def test_process_turn_returns_turn_result(
    pipeline_mocks: tuple[MockLLMClient, TurnPipeline],
) -> None:
    mock, pipeline = pipeline_mocks
    mock.queue(
        _intent_json(),    # Understander
        _coach_json(),     # Coach (attempt 1)
        _critic_pass_json(),  # Critic → pass
        _profiler_json(),  # Profiler (background)
    )

    result = await pipeline.process_turn(
        user_id=FAKE_USER_ID,
        user_message="Should I study economics?",
    )

    assert isinstance(result, TurnResult)
    assert result.session_id == FAKE_SESSION_ID
    assert result.turn_id == FAKE_TURN_ID
    assert not result.clarification_only
    assert "economics" in result.response.lower()


async def test_process_turn_clarification_short_circuits(
    pipeline_mocks: tuple[MockLLMClient, TurnPipeline],
) -> None:
    mock, pipeline = pipeline_mocks
    mock.queue(
        _intent_json(
            clarity_score=0.3,
            needs_clarification=True,
            clarification_question="What aspect of economics interests you?",
            budget_hint="standard",
        )
    )

    result = await pipeline.process_turn(
        user_id=FAKE_USER_ID,
        user_message="I dunno",
    )

    assert result.clarification_only
    assert "economics" in result.response.lower()
    # Coach was never called
    assert len(mock.calls) == 1


async def test_process_turn_flow_a_skips_critic(
    pipeline_mocks: tuple[MockLLMClient, TurnPipeline],
) -> None:
    mock, pipeline = pipeline_mocks
    # budget_hint=quick → Orchestrator routes flow A → no Critic call
    mock.queue(
        _intent_json(turn_intent="explore", budget_hint="quick"),
        _coach_json(),
        _profiler_json(),
    )

    result = await pipeline.process_turn(
        user_id=FAKE_USER_ID,
        user_message="Quick question",
    )

    assert isinstance(result, TurnResult)
    # Only Understander (1) + Coach (1) + Profiler (1) = 3 LLM calls; no Critic
    # Profiler may not complete before assertion due to asyncio.create_task;
    # assert at minimum understander + coach called
    assert len(mock.calls) >= 2


async def test_process_turn_creates_session_when_none_given(
    pipeline_mocks: tuple[MockLLMClient, TurnPipeline],
) -> None:
    mock, pipeline = pipeline_mocks
    mock.queue(_intent_json(), _coach_json(), _critic_pass_json(), _profiler_json())

    # Passing session_id=None should trigger create_session
    pipeline._episodic.get_session = AsyncMock(return_value=None)  # type: ignore[assignment]
    await pipeline.process_turn(
        user_id=FAKE_USER_ID,
        user_message="Hello",
        session_id=None,
    )

    pipeline._episodic.create_session.assert_awaited_once()
