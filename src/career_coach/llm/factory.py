"""Factory that wires per-agent LLM clients from ``config/models.yaml``.

The agent code never imports a provider SDK. It asks the factory for its
client by name and receives a configured :class:`LLMClient` plus the defaults
(model, temperature, max tokens) it should use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from career_coach.config import get_settings
from career_coach.llm.anthropic import AnthropicClient
from career_coach.llm.client import LLMClient
from career_coach.llm.huggingface import HuggingFaceClient

Provider = Literal["anthropic", "huggingface"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "models.yaml"


@dataclass(frozen=True, slots=True)
class AgentLLMConfig:
    """Resolved configuration for a single agent."""

    provider: Provider
    model: str
    temperature: float
    max_tokens: int


class LLMFactory:
    """Build and cache one :class:`LLMClient` per provider."""

    def __init__(self, config_path: Path | None = None):
        """Load config from ``config/models.yaml`` (or override for tests)."""
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._agent_configs: dict[str, AgentLLMConfig] = {}
        self._clients: dict[Provider, LLMClient] = {}
        self._load()

    def _load(self) -> None:
        if not self._config_path.exists():
            raise FileNotFoundError(f"LLM config not found: {self._config_path}")
        raw = yaml.safe_load(self._config_path.read_text()) or {}
        for agent_name, cfg in raw.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Malformed LLM config for agent '{agent_name}'")
            provider = cfg.get("provider")
            if provider not in ("anthropic", "huggingface"):
                raise ValueError(f"Unsupported provider '{provider}' for agent '{agent_name}'")
            self._agent_configs[agent_name] = AgentLLMConfig(
                provider=provider,
                model=str(cfg["model"]),
                temperature=float(cfg.get("temperature", 0.7)),
                max_tokens=int(cfg.get("max_tokens", 2000)),
            )

    # -- public API ---------------------------------------------------------

    def config_for(self, agent_name: str) -> AgentLLMConfig:
        """Return the resolved config for an agent, or raise ``KeyError``."""
        try:
            return self._agent_configs[agent_name]
        except KeyError as exc:  # pragma: no cover — trivial re-raise
            raise KeyError(f"No LLM config for agent '{agent_name}'") from exc

    def client_for(self, agent_name: str) -> LLMClient:
        """Return the :class:`LLMClient` configured for the given agent."""
        return self._client_for_provider(self.config_for(agent_name).provider)

    def register_client(self, provider: Provider, client: LLMClient) -> None:
        """Inject a client (used by tests to substitute :class:`MockLLMClient`)."""
        self._clients[provider] = client

    # -- internals ----------------------------------------------------------

    def _client_for_provider(self, provider: Provider) -> LLMClient:
        if provider in self._clients:
            return self._clients[provider]

        settings = get_settings()
        client: LLMClient
        if provider == "anthropic":
            client = AnthropicClient(api_key=settings.anthropic_api_key)
        elif provider == "huggingface":
            client = HuggingFaceClient(api_token=settings.huggingface_api_token)
        else:  # pragma: no cover — exhaustively handled above
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[provider] = client
        return client
