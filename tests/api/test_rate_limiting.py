"""Tests for per-IP rate limiting and admin token gate.

Uses a minimal FastAPI test app wired via httpx.AsyncClient + ASGITransport so
the tests are compatible with pytest-asyncio's session-scoped event loop.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, Request
from httpx import ASGITransport, AsyncClient
from slowapi.errors import RateLimitExceeded

from career_coach.api.limiter import _client_ip, limiter
from career_coach.api.main import _handle_rate_limit
from career_coach.api.routes import _require_admin


# ── Minimal test app ──────────────────────────────────────────────────────────

def _make_test_app(chat_limit: str = "60/hour", users_limit: str = "5/hour") -> FastAPI:
    """Build a stripped-down FastAPI app that exercises the limiter logic."""
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)  # type: ignore[arg-type]

    @app.post("/chat")
    @limiter.limit(chat_limit)
    async def _chat(request: Request) -> dict[str, str]:
        return {"ok": "true"}

    @app.post("/users")
    @limiter.limit(users_limit)
    async def _create_user(request: Request) -> dict[str, str]:
        return {"ok": "true"}

    @app.post("/admin/test", dependencies=[Depends(_require_admin)])
    async def _admin_ep(request: Request) -> dict[str, str]:
        return {"ok": "true"}

    return app


# ── X-Forwarded-For key function ──────────────────────────────────────────────


def test_client_ip_reads_x_forwarded_for() -> None:
    """_client_ip uses the first X-Forwarded-For value, not request.client."""
    from unittest.mock import MagicMock

    req = MagicMock()
    req.headers = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    req.client = MagicMock(host="10.0.0.1")

    assert _client_ip(req) == "203.0.113.5"


def test_client_ip_falls_back_to_host() -> None:
    """_client_ip falls back to request.client.host when no XFF header."""
    from unittest.mock import MagicMock

    req = MagicMock()
    req.headers = {}
    req.client = MagicMock(host="192.0.2.1")

    assert _client_ip(req) == "192.0.2.1"


# ── Rate limit behaviour ──────────────────────────────────────────────────────


async def test_users_rate_limit_blocks_after_limit() -> None:
    """/users allows up to the limit then returns 429 with correct JSON."""
    app = _make_test_app(users_limit="3/minute")
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ip = "203.0.113.10"
        for i in range(3):
            r = await client.post("/users", headers={"X-Forwarded-For": ip})
            assert r.status_code == 200, f"Expected 200 on attempt {i + 1}"

        r = await client.post("/users", headers={"X-Forwarded-For": ip})
        assert r.status_code == 429
        body = r.json()
        assert body["error"] == "rate_limit"
        assert isinstance(body["retry_after_seconds"], int)
        assert body["retry_after_seconds"] > 0
        assert "Retry-After" in r.headers


async def test_chat_rate_limit_different_ips_are_independent() -> None:
    """Two different IPs do not share a rate-limit bucket."""
    app = _make_test_app(chat_limit="2/minute")
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(2):
            r = await client.post("/chat", headers={"X-Forwarded-For": "1.1.1.1"})
            assert r.status_code == 200

        # First IP is now at the limit
        assert (await client.post("/chat", headers={"X-Forwarded-For": "1.1.1.1"})).status_code == 429
        # Second IP is still free
        assert (await client.post("/chat", headers={"X-Forwarded-For": "2.2.2.2"})).status_code == 200


async def test_429_body_has_correct_shape() -> None:
    """429 response body always has error + retry_after_seconds keys."""
    app = _make_test_app(users_limit="1/minute")
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ip = "203.0.113.99"
        await client.post("/users", headers={"X-Forwarded-For": ip})  # consume quota
        r = await client.post("/users", headers={"X-Forwarded-For": ip})

        assert r.status_code == 429
        body = r.json()
        assert set(body.keys()) == {"error", "retry_after_seconds"}
        assert body["error"] == "rate_limit"


# ── Admin token gate ──────────────────────────────────────────────────────────


async def test_admin_gate_rejects_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """X-Admin-Token mismatch → 403."""
    monkeypatch.setenv("ADMIN_TOKEN", "real-secret")
    from career_coach.config import get_settings

    get_settings.cache_clear()
    try:
        app = _make_test_app()
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/admin/test", headers={"X-Admin-Token": "wrong"})
            assert r.status_code == 403

            r = await client.post("/admin/test", headers={"X-Admin-Token": "real-secret"})
            assert r.status_code == 200
    finally:
        get_settings.cache_clear()


async def test_admin_gate_open_when_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ADMIN_TOKEN is not set, admin routes are open (local dev)."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    from career_coach.config import get_settings

    get_settings.cache_clear()
    try:
        app = _make_test_app()
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/admin/test")  # no token header
            assert r.status_code == 200
    finally:
        get_settings.cache_clear()
