"""Tests for the deployment_version stamp.

Covers:
  - Settings.deployment_version property (all combinations of env vars)
  - EpisodicRepo.save_turn accepts and passes deployment_version to the DB
  - TurnPipeline._node_persist passes the current deployment_version
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from career_coach.config import Settings, get_settings
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient
from career_coach.memory.episodic import Session
from career_coach.pipeline.turn import TurnPipeline

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

FAKE_SESSION_ID = uuid4()
FAKE_TURN_ID = uuid4()
FAKE_USER_ID = uuid4()


# ---------------------------------------------------------------------------
# Settings.deployment_version property
# ---------------------------------------------------------------------------


def _settings(**env_overrides: str | None) -> Settings:
    """Build a Settings instance with specific env values, bypassing the cache."""
    return Settings(**{k: v for k, v in env_overrides.items() if v is not None})  # type: ignore[arg-type]


def test_deployment_version_tag_and_sha() -> None:
    s = Settings(
        GIT_SHA="abc1234def5678",  # type: ignore[call-arg]
        DEPLOYMENT_TAG="v0.5.1",  # type: ignore[call-arg]
    )
    assert s.deployment_version == "v0.5.1@abc1234"


def test_deployment_version_sha_only() -> None:
    s = Settings(GIT_SHA="abc1234def5678")  # type: ignore[call-arg]
    assert s.deployment_version == "abc1234"


def test_deployment_version_railway_sha_fallback() -> None:
    """RAILWAY_GIT_COMMIT_SHA is used when GIT_SHA is absent."""
    s = Settings(RAILWAY_GIT_COMMIT_SHA="feed0001cafe")  # type: ignore[call-arg]
    assert s.deployment_version == "feed000"


def test_deployment_version_prefers_git_sha_over_railway() -> None:
    """GIT_SHA (baked at build time) takes priority over Railway runtime SHA."""
    s = Settings(
        GIT_SHA="aaaaaaa1111111",  # type: ignore[call-arg]
        RAILWAY_GIT_COMMIT_SHA="bbbbbbb2222222",  # type: ignore[call-arg]
    )
    assert s.deployment_version == "aaaaaaa"


def test_deployment_version_local_fallback() -> None:
    """With no SHA at all, returns 'local'."""
    s = Settings()
    assert s.deployment_version == "local"


def test_deployment_version_sha_truncates_to_7_chars() -> None:
    s = Settings(GIT_SHA="0123456789abcdef")  # type: ignore[call-arg]
    assert s.deployment_version == "0123456"


# ---------------------------------------------------------------------------
# EpisodicRepo.save_turn — DB receives deployment_version
# ---------------------------------------------------------------------------


def _mock_pool(turn_id: object = None) -> AsyncMock:
    """Return a mock pool whose connection emulates a turn insert."""
    mock_conn = AsyncMock()
    # fetchval returns next_index (0) then turn_id
    mock_conn.fetchval.side_effect = [0, turn_id or FAKE_TURN_ID]

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    # transaction() context manager
    tx_ctx = AsyncMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=None)
    tx_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=tx_ctx)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)
    return AsyncMock(return_value=pool)


@pytest.mark.asyncio
async def test_save_turn_passes_deployment_version_to_db() -> None:
    """save_turn must include deployment_version in the INSERT statement."""
    from career_coach.memory.episodic import EpisodicRepo

    repo = EpisodicRepo()
    mock_get_pool = _mock_pool()

    with patch("career_coach.memory.episodic.get_pool", new=mock_get_pool):
        await repo.save_turn(
            session_id=FAKE_SESSION_ID,
            user_id=FAKE_USER_ID,
            user_message="hello",
            assistant_message="hi",
            intent_packet=None,
            flow_used="B",
            critic_verdicts=None,
            tokens_used=None,
            deployment_version="v0.5.1@abc1234",
        )

    pool = await mock_get_pool()
    conn = await pool.acquire().__aenter__()

    # The INSERT call is the second fetchval call — check that the SQL and
    # params include the deployment_version string.
    insert_call = conn.fetchval.call_args_list[1]
    sql: str = insert_call.args[0]
    params: tuple = insert_call.args[1:]

    assert "deployment_version" in sql
    assert "v0.5.1@abc1234" in params


# ---------------------------------------------------------------------------
# TurnPipeline — deployment_version propagated from settings
# ---------------------------------------------------------------------------


def _intent_json(**overrides: object) -> str:
    base: dict[str, object] = {
        "session_theory": "User explores career options.",
        "turn_intent": "explore",
        "specific_ask": "What career should I pick?",
        "emotional_tenor": "curious",
        "clarity_score": 0.9,
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
            "response_text": "Economics is a good choice.",
            "referenced_facts": [],
            "referenced_hypotheses": [],
            "proposed_challenge": None,
            "uncertainty_flags": [],
        }
    )


def _critic_pass_json() -> str:
    return json.dumps(
        {"verdict": "pass", "failure_modes": [], "specific_complaints": [], "suggested_fix": None}
    )


def _supervisor_pass_json() -> str:
    return json.dumps(
        {"event_type": None, "severity": None, "details": None, "action": "pass", "scripted_override": None}
    )


@pytest.mark.asyncio
async def test_pipeline_stamps_deployment_version_on_turn() -> None:
    """_node_persist must call save_turn with the settings deployment_version."""
    mock_llm = MockLLMClient()
    mock_llm.queue(
        _intent_json(),
        _coach_json(),
        _critic_pass_json(),
        _supervisor_pass_json(),
    )

    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock_llm)
    pipeline = TurnPipeline(factory)

    sess = Session(session_id=FAKE_SESSION_ID, user_id=FAKE_USER_ID, session_theory=None)
    episodic = AsyncMock()
    episodic.get_session.return_value = sess
    episodic.create_session.return_value = sess
    episodic.get_recent.return_value = []
    episodic.save_turn.return_value = FAKE_TURN_ID
    episodic.update_session_theory.return_value = None

    structured = AsyncMock()
    structured.get_all.return_value = {"location": "London"}
    structured.bulk_upsert.return_value = None

    semantic = AsyncMock()
    semantic.get_active.return_value = []

    pipeline._structured = structured
    pipeline._episodic = episodic
    pipeline._semantic = semantic
    pipeline._onboarding_policy = AsyncMock()
    pipeline._onboarding_policy.is_new_user.return_value = False

    # Patch settings to return a known version string
    fake_settings = MagicMock()
    fake_settings.deployment_version = "v0.5.1@abc1234"

    with patch("career_coach.pipeline.turn.get_settings", return_value=fake_settings):
        await pipeline.process_turn(
            user_id=FAKE_USER_ID,
            user_message="What career should I pick?",
        )

    # save_turn must have been called exactly once (main persist path, not clarification)
    episodic.save_turn.assert_awaited_once()
    call_kwargs = episodic.save_turn.call_args.kwargs
    assert call_kwargs["deployment_version"] == "v0.5.1@abc1234", (
        f"Expected deployment_version='v0.5.1@abc1234', got {call_kwargs.get('deployment_version')!r}"
    )
