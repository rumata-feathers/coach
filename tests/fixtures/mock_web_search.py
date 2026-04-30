"""In-memory mock web search client for unit tests.

Provides a queue-based and default-response-based mock that matches the
:class:`~career_coach.web.client.WebSearchClient` protocol.
Every call is recorded on ``search_calls`` / ``fetch_calls`` so tests can
assert on query strings, max_results, and other parameters.

Usage::

    mock = MockWebSearchClient()
    mock.queue_search_results([SearchResult(url="https://example.com", title="Test", snippet="...")])
    results = await mock.search("quant finance")
    assert mock.search_calls[0]["query"] == "quant finance"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from career_coach.web.client import SearchResult


@dataclass
class MockWebSearchClient:
    """Deterministic web search client for unit tests.

    Attributes:
        default_results: Returned by :meth:`search` when the queue is empty.
        default_fetch_content: Returned by :meth:`fetch` when the queue is empty.
        search_calls: Recorded arguments from every :meth:`search` call.
        fetch_calls: Recorded URLs from every :meth:`fetch` call.
    """

    default_results: list[SearchResult] = field(default_factory=list)
    default_fetch_content: str = "Extracted page content."
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    fetch_calls: list[str] = field(default_factory=list)
    _search_queue: deque[list[SearchResult]] = field(default_factory=deque)
    _fetch_queue: deque[str] = field(default_factory=deque)

    def queue_search_results(self, *batches: list[SearchResult]) -> None:
        """Enqueue one or more :meth:`search` responses to return in order."""
        for batch in batches:
            self._search_queue.append(batch)

    def queue_fetch_content(self, *contents: str) -> None:
        """Enqueue one or more :meth:`fetch` responses to return in order."""
        self._fetch_queue.extend(contents)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
        turn_id: Any = None,
    ) -> list[SearchResult]:
        """Return queued results or :attr:`default_results`."""
        self.search_calls.append(
            {"query": query, "max_results": max_results, "recency_days": recency_days}
        )
        if self._search_queue:
            return self._search_queue.popleft()
        return list(self.default_results)

    async def fetch(self, url: str) -> str:
        """Return queued content or :attr:`default_fetch_content`."""
        self.fetch_calls.append(url)
        if self._fetch_queue:
            return self._fetch_queue.popleft()
        return self.default_fetch_content
