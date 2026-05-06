"""Integration tests for the FastAPI HTTP surface.

Tests the real app with a live Postgres DB. The LLM layer is intercepted by
a MockLLMClient injected via the factory to keep tests fast and deterministic.

All tests use httpx.AsyncClient with ASGITransport (no real TCP).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from career_coach.api.main import create_app
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient

pytestmark = pytest.mark.asyncio

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def _intent_json() -> str:
    return json.dumps(
        {
            "session_theory": "User exploring career options.",
            "turn_intent": "explore",
            "specific_ask": "Best path for an economics student.",
            "emotional_tenor": "curious",
            "clarity_score": 0.8,
            "needs_clarification": False,
            "clarification_question": None,
            "inferred_constraints": [],
            "budget_hint": "standard",
        }
    )


def _coach_json() -> str:
    return json.dumps(
        {
            "response_text": "Economics is a great choice for an analytical thinker in London.",
            "referenced_facts": ["location"],
            "referenced_hypotheses": [],
            "proposed_challenge": None,
            "uncertainty_flags": [],
        }
    )


def _critic_pass_json() -> str:
    return json.dumps(
        {"verdict": "pass", "failure_modes": [], "specific_complaints": [], "suggested_fix": None}
    )


def _profiler_json() -> str:
    return json.dumps(
        {
            "new_facts": [],
            "fact_updates": [],
            "hypothesis_evidence": [],
        }
    )


def _parse_sse(resp: httpx.Response) -> dict:
    """Extract the JSON payload from a Server-Sent Events response body.

    The /chat endpoint emits optional keepalive comment lines followed by a
    single ``data: {...}`` event.  Plain ``.json()`` fails because the body
    starts with ``: keepalive``.
    """
    text = resp.text
    match = re.search(r"^data: (.+)$", text, re.MULTILINE)
    assert match, f"No 'data:' line found in SSE body: {text!r}"
    return json.loads(match.group(1))


def _make_mock_factory() -> tuple[MockLLMClient, LLMFactory]:
    mock = MockLLMClient()
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return mock, factory


@pytest.fixture()
async def api_client(migrated_db: str):  # type: ignore[no-untyped-def]
    """Yield an httpx.AsyncClient wired to the real app with a mock LLM."""
    mock, factory = _make_mock_factory()
    # Pre-fill enough responses for a typical flow: understander, coach, critic, profiler
    mock.queue(_intent_json(), _coach_json(), _critic_pass_json(), _profiler_json())

    app = create_app()

    # Patch the routes module so it uses our mock factory
    with patch("career_coach.api.routes._factory", factory):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, mock


# ---- /users ------------------------------------------------------------------


async def test_create_user_returns_201_with_uuid(migrated_db: str) -> None:
    _mock, factory = _make_mock_factory()
    app = create_app()
    with patch("career_coach.api.routes._factory", factory):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/users", json={"display_name": "Test Student"})

    assert response.status_code == 201
    body = response.json()
    assert "user_id" in body
    UUID(body["user_id"])  # Ensure it's a valid UUID


async def test_create_user_returns_422_on_empty_name(migrated_db: str) -> None:
    _mock, factory = _make_mock_factory()
    app = create_app()
    with patch("career_coach.api.routes._factory", factory):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/users", json={"display_name": ""})

    assert response.status_code == 422


# ---- /chat -------------------------------------------------------------------


async def test_chat_returns_response(api_client) -> None:  # type: ignore[no-untyped-def]
    client, mock = api_client
    # First create a user
    user_resp = await client.post("/users", json={"display_name": f"chat_test_{uuid4().hex[:6]}", "is_test": True})
    user_id = user_resp.json()["user_id"]

    # Reset mock queue for this test
    mock.queue(_intent_json(), _coach_json(), _critic_pass_json(), _profiler_json())

    resp = await client.post(
        "/chat",
        json={"user_id": user_id, "message": "Should I study economics?"},
    )

    assert resp.status_code == 200
    body = _parse_sse(resp)
    assert "response" in body
    assert body["response"]
    assert "turn_id" in body
    assert "session_id" in body
    UUID(body["session_id"])


async def test_chat_returns_404_for_unknown_user(api_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = api_client
    resp = await client.post(
        "/chat",
        json={"user_id": str(uuid4()), "message": "Hello"},
    )
    assert resp.status_code == 404


async def test_chat_preserves_session_id(api_client) -> None:  # type: ignore[no-untyped-def]
    client, mock = api_client
    user_resp = await client.post(
        "/users", json={"display_name": f"session_test_{uuid4().hex[:6]}", "is_test": True}
    )
    user_id = user_resp.json()["user_id"]

    # Turn 1
    mock.queue(_intent_json(), _coach_json(), _critic_pass_json(), _profiler_json())
    resp1 = await client.post("/chat", json={"user_id": user_id, "message": "Hello"})
    session_id = _parse_sse(resp1)["session_id"]

    # Turn 2 — pass the same session_id back
    mock.queue(_intent_json(), _coach_json(), _critic_pass_json(), _profiler_json())
    resp2 = await client.post(
        "/chat",
        json={"user_id": user_id, "message": "Follow-up", "session_id": session_id},
    )

    assert resp2.status_code == 200
    assert _parse_sse(resp2)["session_id"] == session_id


# ---- /admin/distill ----------------------------------------------------------


async def test_distill_dry_run_returns_200(api_client) -> None:  # type: ignore[no-untyped-def]
    client, _mock = api_client
    user_resp = await client.post(
        "/users", json={"display_name": f"distill_{uuid4().hex[:6]}", "is_test": True}
    )
    user_id = user_resp.json()["user_id"]

    resp = await client.post(f"/admin/distill/{user_id}", json={"dry_run": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "dry_run"
    assert body["user_id"] == user_id


async def test_distill_returns_404_for_unknown_user(api_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = api_client
    resp = await client.post(f"/admin/distill/{uuid4()}", json={})
    assert resp.status_code == 404
