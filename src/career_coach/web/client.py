"""Core web search interface shared by every provider implementation.

Every provider (Tavily, Exa, …) exposes two async methods:
- :meth:`WebSearchClient.search` — keyword search, returns ranked results.
- :meth:`WebSearchClient.fetch` — fetch a URL and return clean extracted text.

Agents depend on this protocol, not on any SDK, so providers are swappable
via ``config/web_search.yaml``.

See SPEC_v1.md §4.2 for the canonical contract.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single result returned by :meth:`WebSearchClient.search`.

    Attributes:
        url: Canonical URL of the source page.
        title: Page/article title.
        snippet: Short excerpt returned by the search provider.
        content: Full extracted text of the page, populated by a subsequent
            :meth:`WebSearchClient.fetch` call (or by the provider during
            search, if the provider supports it). ``None`` until populated.
        published_date: Publication date of the page when available.
        score: Provider relevance score in [0, 1], optional.
    """

    url: str
    title: str
    snippet: str
    content: str | None = None
    published_date: date | None = None
    score: float | None = None


class WebSearchClient(Protocol):
    """Protocol every web search provider must satisfy.

    Implementations must not raise for graceful failures (no results, quota
    exceeded). They must raise for auth / transport errors so callers can
    surface them.
    """

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        """Search for *query* and return up to *max_results* ranked results.

        Args:
            query: Natural-language search query.
            max_results: Maximum number of results to return.
            recency_days: If set, restrict results to the last N days.

        Returns:
            List of search results (may be empty on quota exhaustion or when
            no results are found; never raises for these cases).
        """
        ...

    async def fetch(self, url: str) -> str:
        """Fetch *url* and return the main text content.

        Returns clean plain text (no HTML tags). Raises on network / HTTP
        errors.

        Args:
            url: The page URL to fetch and extract.
        """
        ...
