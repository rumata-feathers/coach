"""Unit-test configuration for the agents package.

``Agent.log_call`` writes to the ``agent_calls`` table via :func:`get_pool`.
Unit tests run without a live database, so we patch it to a no-op here.
Integration tests in ``tests/integration/`` use the real ``log_call`` path
and are unaffected because this conftest only applies to ``tests/agents/``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
async def _patch_agent_log_call() -> AsyncIterator[None]:
    """Silence ``Agent.log_call`` so unit tests need no Postgres connection."""
    with patch("career_coach.agents.base.Agent.log_call", new_callable=AsyncMock):
        yield
