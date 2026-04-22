"""Smoke test: the FastAPI app boots and ``/health`` returns OK."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from career_coach.api.main import app


async def test_health_endpoint_returns_ok() -> None:
    """``GET /health`` must return ``{"status": "ok", ...}`` with HTTP 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
