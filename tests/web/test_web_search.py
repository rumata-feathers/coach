"""Unit tests for the web search adapter.

All tests mock the underlying httpx client — no live API calls.
The integration test (test_tavily_live_*) is skipped if TAVILY_API_KEY is not set.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from career_coach.web.cache import WebSearchCache
from career_coach.web.client import SearchResult
from career_coach.web.quota import QuotaTracker
from career_coach.web.tavily import TavilyClient, _strip_html

_FIXTURE_PATH = Path(__file__).parent / "tavily_search_fixture.json"


# ---- helpers ----------------------------------------------------------------


def _make_client(
    quota: QuotaTracker | None = None,
    cache: WebSearchCache | None = None,
) -> TavilyClient:
    return TavilyClient(
        api_key="test-key",
        quota=quota or QuotaTracker(daily_limit=100),
        cache=cache or WebSearchCache(),
    )


def _mock_http_response(fixture_path: Path) -> MagicMock:
    """Return a mock httpx.AsyncClient whose post() returns the fixture JSON."""
    fixture = json.loads(fixture_path.read_text())
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = fixture
    mock_resp.raise_for_status = MagicMock()

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=mock_resp)
    return mock_http


# ---- tests ------------------------------------------------------------------


async def test_tavily_search_returns_results() -> None:
    """search() with a fixture response returns correctly parsed SearchResults."""
    client = _make_client()
    client._http = _mock_http_response(_FIXTURE_PATH)

    results = await client.search("quant finance careers UK 2026")

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].url == "https://example.com/quant-finance-2026"
    assert results[0].title == "Quantitative Finance Careers in the UK: 2026 Guide"
    assert "quant" in results[0].snippet.lower()
    assert results[0].score == pytest.approx(0.92)
    assert results[0].published_date == date(2026, 1, 15)


async def test_tavily_search_uses_correct_payload() -> None:
    """search() sends the query, max_results, and recency_days in the request body."""
    client = _make_client()
    client._http = _mock_http_response(_FIXTURE_PATH)

    await client.search("test query", max_results=3, recency_days=30)

    call_args = client._http.post.call_args
    payload = call_args.kwargs.get("json") or call_args.args[1]
    assert payload["query"] == "test query"
    assert payload["max_results"] == 3
    assert payload["days"] == 30


async def test_quota_exhaustion_returns_empty() -> None:
    """When daily quota is exhausted, search() returns an empty list without calling the API."""
    quota = QuotaTracker(daily_limit=1)
    client = _make_client(quota=quota)
    client._http = _mock_http_response(_FIXTURE_PATH)

    # First call: within quota
    results_first = await client.search("first query")
    assert len(results_first) == 3
    assert client._http.post.call_count == 1

    # Second call: quota exceeded
    results_second = await client.search("second query")
    assert results_second == []
    assert client._http.post.call_count == 1  # no additional HTTP call
    assert quota.is_exceeded


async def test_quota_exhaustion_flag_is_set() -> None:
    """QuotaTracker.is_exceeded is True after the limit is reached."""
    quota = QuotaTracker(daily_limit=2)
    assert not quota.is_exceeded
    assert quota.try_consume()   # 1/2
    assert not quota.is_exceeded
    assert quota.try_consume()   # 2/2
    assert quota.is_exceeded
    assert not quota.try_consume()  # over limit


async def test_cache_hit_does_not_call_api() -> None:
    """Second search with identical params returns cached results without an HTTP call."""
    cache = WebSearchCache()
    client = _make_client(cache=cache)
    client._http = _mock_http_response(_FIXTURE_PATH)

    results_first = await client.search("quant finance", max_results=5)
    assert client._http.post.call_count == 1

    results_second = await client.search("quant finance", max_results=5)
    assert client._http.post.call_count == 1  # no additional call
    assert results_second == results_first


async def test_cache_miss_on_different_params() -> None:
    """Different max_results means a cache miss — the API is called again."""
    cache = WebSearchCache()
    client = _make_client(cache=cache)
    client._http = _mock_http_response(_FIXTURE_PATH)

    await client.search("quant finance", max_results=5)
    await client.search("quant finance", max_results=3)
    assert client._http.post.call_count == 2


async def test_cache_lru_eviction() -> None:
    """Cache evicts oldest entries when max_size is exceeded."""
    cache = WebSearchCache(max_size=2)
    result = SearchResult(url="https://example.com", title="T", snippet="S")
    cache.set("tavily", "q1", 5, None, [result])
    cache.set("tavily", "q2", 5, None, [result])
    cache.set("tavily", "q3", 5, None, [result])  # evicts q1

    assert cache.get("tavily", "q1", 5, None) is None
    assert cache.get("tavily", "q2", 5, None) is not None
    assert cache.get("tavily", "q3", 5, None) is not None
    assert cache.size == 2


async def test_fetch_extracts_main_content() -> None:
    """fetch() strips HTML tags and returns clean plain text."""
    html_page = """<!DOCTYPE html>
<html>
<head><title>Test</title>
<style>body { color: red; }</style>
<script>var x = 1;</script>
</head>
<body>
<nav>Navigation stuff</nav>
<main>
<h1>Quant Finance Guide</h1>
<p>Quantitative analysts work on pricing models &amp; risk systems.</p>
<p>Typical salary: &gt;£60k for junior roles.</p>
</main>
</body>
</html>"""

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = html_page
    mock_resp.raise_for_status = MagicMock()

    # Make extract endpoint fail so we fall through to raw fetch
    def post_side_effect(*args: object, **kwargs: object) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        return resp

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(side_effect=post_side_effect)
    mock_http.get = AsyncMock(return_value=mock_resp)

    client = _make_client()
    client._http = mock_http

    content = await client.fetch("https://example.com/quant-guide")

    assert "<html>" not in content
    assert "<p>" not in content
    assert "<script>" not in content
    assert "<style>" not in content
    assert "Quant Finance Guide" in content
    assert "Quantitative analysts" in content
    assert "&amp;" not in content  # entity decoded
    assert "£60k" in content


async def test_strip_html_utility() -> None:
    """_strip_html helper strips tags, decodes entities, collapses whitespace."""
    html = "<p>Hello &amp; <b>world</b>!</p><script>bad()</script>"
    result = _strip_html(html)
    assert result == "Hello & world !"


async def test_mock_web_search_client_records_calls() -> None:
    """MockWebSearchClient records every search call and returns queued results."""
    from tests.fixtures.mock_web_search import MockWebSearchClient

    mock = MockWebSearchClient()
    result1 = SearchResult(url="https://a.com", title="A", snippet="snippet A")
    result2 = SearchResult(url="https://b.com", title="B", snippet="snippet B")
    mock.queue_search_results([result1, result2])

    results = await mock.search("test query", max_results=3)

    assert len(results) == 2
    assert results[0].url == "https://a.com"
    assert len(mock.search_calls) == 1
    assert mock.search_calls[0]["query"] == "test query"
    assert mock.search_calls[0]["max_results"] == 3


async def test_mock_web_search_client_default_results() -> None:
    """MockWebSearchClient returns default_results when queue is empty."""
    from tests.fixtures.mock_web_search import MockWebSearchClient

    default = [SearchResult(url="https://default.com", title="Default", snippet="D")]
    mock = MockWebSearchClient(default_results=default)

    results = await mock.search("anything")
    assert results == default


async def test_mock_web_search_client_fetch() -> None:
    """MockWebSearchClient records fetch calls and returns queued / default content."""
    from tests.fixtures.mock_web_search import MockWebSearchClient

    mock = MockWebSearchClient()
    mock.queue_fetch_content("Custom fetched content.")

    content = await mock.fetch("https://example.com")
    assert content == "Custom fetched content."
    assert mock.fetch_calls == ["https://example.com"]

    # Falls back to default when queue is empty
    content2 = await mock.fetch("https://example2.com")
    assert content2 == "Extracted page content."


# ---- integration test (skipped without API key) ----------------------------


async def test_tavily_live_search() -> None:
    """Live Tavily integration: search returns ≥3 results with valid URLs."""
    from career_coach.config import get_settings

    settings = get_settings()
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY not set — skipping live Tavily test")

    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        results = await client.search("quant finance careers UK 2026", max_results=5)
        assert len(results) >= 3, f"Expected ≥3 results, got {len(results)}"
        for r in results:
            assert r.url.startswith("http"), f"Invalid URL: {r.url!r}"
            assert r.title
            assert r.snippet

        # Fetch top result
        top_url = results[0].url
        content = await client.fetch(top_url)
        assert len(content) > 100, "Fetched content suspiciously short"
    finally:
        await client.aclose()
