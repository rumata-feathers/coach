"""Unit tests for :class:`LLMFactory`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(dedent(body))
    return path


def test_factory_loads_agent_configs(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        understander:
          provider: anthropic
          model: claude-haiku-4-5-20251001
          temperature: 0.2
          max_tokens: 1500
        coach:
          provider: anthropic
          model: claude-sonnet-4-6
        """,
    )
    factory = LLMFactory(config_path=config_path)
    understander_cfg = factory.config_for("understander")
    assert understander_cfg.provider == "anthropic"
    assert understander_cfg.model == "claude-haiku-4-5-20251001"
    assert understander_cfg.temperature == pytest.approx(0.2)
    assert understander_cfg.max_tokens == 1500

    coach_cfg = factory.config_for("coach")
    # Defaults apply when fields are omitted.
    assert coach_cfg.temperature == pytest.approx(0.7)
    assert coach_cfg.max_tokens == 2000


def test_factory_rejects_unknown_provider(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        understander:
          provider: bogus
          model: some-model
        """,
    )
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMFactory(config_path=config_path)


def test_register_client_overrides_provider(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        understander:
          provider: anthropic
          model: claude-haiku-4-5-20251001
        """,
    )
    factory = LLMFactory(config_path=config_path)
    mock = MockLLMClient()
    factory.register_client("anthropic", mock)

    assert factory.client_for("understander") is mock


def test_missing_agent_raises(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        understander:
          provider: anthropic
          model: claude-haiku-4-5-20251001
        """,
    )
    factory = LLMFactory(config_path=config_path)
    with pytest.raises(KeyError):
        factory.config_for("coach")
