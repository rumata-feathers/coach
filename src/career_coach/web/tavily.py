"""Tavily web search and fetch client.

Uses Tavily's REST API directly via httpx (no Tavily Python SDK dependency).
Wraps the cache and quota tracker.

Tavily free tier: ~1 000 calls/month. The :class:`QuotaTracker` default (200
per process per day) keeps us well inside this.

See SPEC_v1.md §4.3.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from datetime import date
from uuid import UUID

import httpx

from career_coach.web.cache import WebSearchCache
from career_coach.web.client import SearchResult
from career_coach.web.quota import QuotaTracker

logger = logging.getLogger("career_coach.web.tavily")

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"

# Maximum characters to return from fetch() to avoid flooding the context window.
_MAX_FETCH_CHARS: int = 8_000

# Compiled HTML stripper patterns (module-level, no re-compile per call).
_RE_SCRIPT_STYLE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_RE_TAGS = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")


class TavilyClient:
    """Tavily-backed :class:`~career_coach.web.client.WebSearchClient`.

    Args:
        api_key: Tavily API key.
        cache: Shared search result cache. If ``None`` a fresh instance is
            created (mostly for unit-test isolation).
        quota: Shared quota tracker. If ``None`` a fresh instance is created.
        timeout: HTTP request timeout in seconds.
    """

    provider: str = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        cache: WebSearchCache | None = None,
        quota: QuotaTracker | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._cache = cache if cache is not None else WebSearchCache()
        self._quota = quota if quota is not None else QuotaTracker()
        self._http = httpx.AsyncClient(timeout=timeout)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
        turn_id: UUID | None = None,
    ) -> list[SearchResult]:
        """Search Tavily and return ranked results.

        Returns an empty list (with a warning log) on quota exhaustion.
        Cache is checked before any HTTP call. Every live API call is logged
        to ``agent_calls`` per SPEC_v1.md §4.6.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            recency_days: Restrict to results published within this many days.
            turn_id: If provided, associate the ``agent_calls`` log row with
                this turn.
        """
        cached = self._cache.get(self.provider, query, max_results, recency_days)
        if cached is not None:
            logger.debug("cache hit for query=%r", query)
            return cached

        if not self._quota.try_consume():
            logger.warning(
                "Tavily daily quota exhausted (%d/%d); returning empty results",
                self._quota._count,
                self._quota._limit,
            )
            return []

        payload: dict[str, object] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if recency_days is not None:
            payload["days"] = recency_days

        t0 = time.monotonic()
        error_str: str | None = None
        results: list[SearchResult] = []
        latency_ms: int = 0
        try:
            resp = await self._http.post(_SEARCH_URL, json=payload)
            latency_ms = int((time.monotonic() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()
            results = [_parse_result(r) for r in data.get("results", [])]
            logger.debug(
                "Tavily search query=%r → %d results in %dms", query, len(results), latency_ms
            )
            self._cache.set(self.provider, query, max_results, recency_days, results)
        except Exception as exc:
            latency_ms = latency_ms or int((time.monotonic() - t0) * 1000)
            error_str = str(exc)
            logger.warning("Tavily search failed for query=%r: %s", query, exc)
            raise
        finally:
            # Log every live API call to agent_calls per spec §4.6.
            # Import deferred to avoid circular import at module load time.
            # Wrapped in try/except so a missing DB in unit tests never breaks search.
            try:
                from career_coach.agents.base import log_agent_call

                await log_agent_call(
                    agent_name="web_search",
                    model_used=self.provider,
                    turn_id=turn_id,
                    input_payload={"query": query, "max_results": max_results,
                                   "recency_days": recency_days},
                    output_payload={"result_count": len(results)},
                    latency_ms=latency_ms,
                    tokens_in=None,
                    tokens_out=None,
                    error=error_str,
                )
            except Exception:
                # Never let observability logging break the search path.
                pass

        return results

    async def fetch(self, url: str) -> str:
        """Fetch *url* using Tavily's /extract endpoint, falling back to raw HTTP + HTML strip.

        Returns at most :data:`_MAX_FETCH_CHARS` characters of clean plain text.
        """
        # Try Tavily's extract endpoint first (returns pre-cleaned content)
        try:
            resp = await self._http.post(
                _EXTRACT_URL,
                json={"api_key": self._api_key, "urls": [url]},
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results and results[0].get("raw_content"):
                    text = results[0]["raw_content"]
                    return text[:_MAX_FETCH_CHARS]
        except httpx.HTTPError:
            pass

        # Fall back: raw GET + HTML strip
        resp = await self._http.get(url, follow_redirects=True)
        resp.raise_for_status()
        return _strip_html(resp.text)[:_MAX_FETCH_CHARS]

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(raw: dict[str, object]) -> SearchResult:
    """Map a Tavily result dict to a :class:`SearchResult`."""
    pub = raw.get("published_date")
    published: date | None = None
    if isinstance(pub, str) and pub:
        with contextlib.suppress(ValueError):
            published = date.fromisoformat(pub[:10])

    score_raw = raw.get("score")
    score: float | None = float(score_raw) if score_raw is not None else None

    return SearchResult(
        url=str(raw.get("url", "")),
        title=str(raw.get("title", "")),
        snippet=str(raw.get("content", "")),
        content=str(raw.get("content", "")) or None,
        published_date=published,
        score=score,
    )


def _strip_html(html: str) -> str:
    """Strip HTML tags and return normalised plain text.

    Uses regex only — no external HTML parser. Sufficient for extracting the
    main prose content from a page.
    """
    # Remove <script> and <style> blocks with their content
    text = _RE_SCRIPT_STYLE.sub(" ", html)
    # Replace remaining tags with spaces
    text = _RE_TAGS.sub(" ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Collapse whitespace
    return _RE_WS.sub(" ", text).strip()
