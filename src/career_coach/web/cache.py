"""In-process LRU cache for web search results.

Cache key: ``(provider, query, max_results, recency_days)``.
TTL: 24 hours (configurable). Cuts API cost during eval runs where the same
queries fire repeatedly, and during Critic retry loops.

The cache is per-process — sufficient for v1. v2 can introduce a shared
Redis-backed cache if multi-process / multi-worker setups are needed.

See SPEC_v1.md §4.4.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from career_coach.web.client import SearchResult

# Default cache parameters
_DEFAULT_TTL_SECONDS: int = 86_400  # 24 hours
_DEFAULT_MAX_SIZE: int = 256         # max distinct key-sets to keep


class WebSearchCache:
    """LRU cache with TTL for :class:`~career_coach.web.client.SearchResult` lists.

    Args:
        max_size: Maximum number of entries to retain (LRU eviction when exceeded).
        ttl_seconds: Time-to-live per entry in seconds.
    """

    def __init__(
        self,
        *,
        max_size: int = _DEFAULT_MAX_SIZE,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        # OrderedDict gives O(1) move-to-end (LRU update) and O(1) pop oldest.
        self._store: OrderedDict[tuple[Any, ...], tuple[float, list[SearchResult]]] = (
            OrderedDict()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        provider: str,
        query: str,
        max_results: int,
        recency_days: int | None,
    ) -> list[SearchResult] | None:
        """Return cached results or ``None`` on cache miss / expiry.

        A hit moves the entry to the end of the LRU order.
        """
        key = self._make_key(provider, query, max_results, recency_days)
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, results = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._store[key]
            return None
        # Refresh LRU position
        self._store.move_to_end(key)
        return results

    def set(
        self,
        provider: str,
        query: str,
        max_results: int,
        recency_days: int | None,
        results: list[SearchResult],
    ) -> None:
        """Store *results* under the given key, evicting LRU if at capacity."""
        key = self._make_key(provider, query, max_results, recency_days)
        self._store[key] = (time.monotonic(), results)
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)  # remove oldest

    def clear(self) -> None:
        """Clear all entries. Intended for test isolation."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        provider: str,
        query: str,
        max_results: int,
        recency_days: int | None,
    ) -> tuple[str, str, int, int | None]:
        return (provider, query, max_results, recency_days)
