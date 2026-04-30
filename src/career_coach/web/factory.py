"""Config-driven web search client factory.

Reads ``config/web_search.yaml`` and returns the configured client per agent.
The cache and quota tracker are shared across all clients returned for the
same provider within a process.

See SPEC_v1.md §4.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from career_coach.web.cache import WebSearchCache
from career_coach.web.client import WebSearchClient
from career_coach.web.quota import QuotaTracker
from career_coach.web.tavily import TavilyClient

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "web_search.yaml"

# Module-level singletons: one cache and one quota tracker per provider.
_CACHES: dict[str, WebSearchCache] = {}
_QUOTAS: dict[str, QuotaTracker] = {}


class WebSearchConfig:
    """Parsed per-agent web search configuration.

    Attributes:
        provider: Name of the search provider (e.g. ``"tavily"``).
        max_results_default: Default *max_results* for :meth:`WebSearchClient.search`.
        recency_days_default: Default *recency_days* (``None`` means no restriction).
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.provider: str = raw["provider"]
        self.max_results_default: int = int(raw.get("max_results_default", 5))
        self.recency_days_default: int | None = raw.get("recency_days_default")


def get_client(
    agent_name: str,
    *,
    config_path: Path = _CONFIG_PATH,
    api_key: str | None = None,
) -> WebSearchClient:
    """Return a configured :class:`WebSearchClient` for *agent_name*.

    The cache and quota tracker are module-level singletons shared across
    calls, so repeated invocations for the same agent reuse the same client
    state.

    Args:
        agent_name: Key in ``config/web_search.yaml`` (e.g. ``"researcher"``).
        config_path: Override path to the YAML config (used in tests).
        api_key: Override the API key (used in tests). When ``None`` the
            appropriate env-var is read from ``career_coach.config``.

    Raises:
        KeyError: If *agent_name* is not present in the config.
        ValueError: If the configured provider is unsupported.
    """
    raw_config = yaml.safe_load(config_path.read_text())
    agent_cfg = WebSearchConfig(raw_config[agent_name])
    provider = agent_cfg.provider

    if provider == "tavily":
        resolved_key = api_key or _get_tavily_key()
        cache = _CACHES.setdefault(provider, WebSearchCache())
        quota = _QUOTAS.setdefault(provider, QuotaTracker())
        return TavilyClient(resolved_key, cache=cache, quota=quota)

    raise ValueError(
        f"Unsupported web search provider: {provider!r}. "
        "Supported providers in v1: 'tavily'."
    )


def _get_tavily_key() -> str:
    """Read TAVILY_API_KEY from environment / .env via career_coach.config."""
    from career_coach.config import get_settings

    settings = get_settings()
    key = settings.tavily_api_key
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Add it to your .env file or set it in the environment."
        )
    return key
