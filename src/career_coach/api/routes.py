"""Conversation and admin API routes.

Endpoints:
  POST /users            — create a user, return their UUID
  POST /chat             — process one turn, return the assistant response
  POST /admin/distill/{user_id} — trigger the distillation job for a user
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from career_coach.api.limiter import limiter
from career_coach.config import get_settings
from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.pipeline.turn import TurnPipeline

logger = logging.getLogger("career_coach.api.routes")


# ---- Admin gate ----------------------------------------------------------

async def _require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Dependency: validate X-Admin-Token against ADMIN_TOKEN env var.

    If ADMIN_TOKEN is unset the check is skipped so local dev works without
    any additional configuration.
    """
    expected = get_settings().admin_token
    if expected is None:
        return
    if x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Token")


# ---- Shared factory (one per process) ------------------------------------

_factory: LLMFactory | None = None


def _get_factory() -> LLMFactory:
    global _factory
    if _factory is None:
        _factory = LLMFactory()
    return _factory


# ---- Request / Response schemas -----------------------------------------


class CreateUserRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=200)
    is_test: bool = Field(default=False, description="Mark as a test/dev user; excluded from daily reports by default.")


class CreateUserResponse(BaseModel):
    user_id: UUID


class ChatRequest(BaseModel):
    user_id: UUID
    message: str = Field(..., min_length=1)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    response: str
    turn_id: UUID | None
    session_id: UUID
    clarification_only: bool = False
    chart_specs: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class DistillRequest(BaseModel):
    """Optional run parameters for the distillation job."""

    dry_run: bool = False


class DistillResponse(BaseModel):
    status: str
    user_id: UUID
    message: str


# ---- Router --------------------------------------------------------------

router = APIRouter()


@router.post("/users", response_model=CreateUserResponse, status_code=201, tags=["users"])
@limiter.limit("5/hour")
async def create_user(request: Request, body: CreateUserRequest) -> CreateUserResponse:
    """Create a new user and return their UUID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            "INSERT INTO users (display_name, is_test) VALUES ($1, $2) RETURNING user_id",
            body.display_name,
            body.is_test,
        )
    logger.info("Created user %s ('%s', is_test=%s).", user_id, body.display_name, body.is_test)
    return CreateUserResponse(user_id=user_id)


@router.post("/chat", tags=["conversation"])
@limiter.limit("60/hour")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Process one user turn and stream the assistant response as SSE.

    Uses Server-Sent Events so Railway's 30-second idle timeout does not
    disconnect long-running Flow C pipeline calls.  The stream emits:
      • ``": keepalive"`` comment lines every 5 s while the pipeline runs
      • A single ``data: {…JSON…}`` line with the full ChatResponse payload

    Clients that do not speak SSE can read the raw body: the final line is
    always a valid ``data:`` event they can parse directly.

    Errors are emitted as ``data: {"error": "…", …}`` so the client gets a
    structured payload even on failure (never a naked 500 that the frontend
    cannot display).
    """
    # Verify user exists — do this before entering the SSE stream so a 404
    # can still be returned as a proper HTTP error (not an SSE error event).
    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM users WHERE user_id = $1)", body.user_id
        )
    if not exists:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found.")

    async def _stream() -> AsyncIterator[str]:
        task = asyncio.create_task(
            TurnPipeline(_get_factory()).process_turn(
                user_id=body.user_id,
                user_message=body.message,
                session_id=body.session_id,
            )
        )
        try:
            # Keep Railway's load-balancer alive with periodic comment lines.
            while not task.done():
                yield ": keepalive\n\n"
                await asyncio.sleep(5)
            result = await task
        except Exception as exc:
            logger.exception("Pipeline error for user %s: %s", body.user_id, exc)
            payload = json.dumps({"error": "pipeline_error", "message": str(exc)})
            yield f"data: {payload}\n\n"
            return

        resp = ChatResponse(
            response=result.response,
            turn_id=result.turn_id,
            session_id=result.session_id,
            clarification_only=result.clarification_only,
            chart_specs=result.chart_specs,
            citations=result.citations,
        )
        yield f"data: {resp.model_dump_json()}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            # Instruct proxies not to buffer the stream.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/admin/distill/{user_id}",
    response_model=DistillResponse,
    tags=["admin"],
    dependencies=[Depends(_require_admin)],
)
async def distill(
    user_id: UUID,
    body: DistillRequest,
    background_tasks: BackgroundTasks,
) -> DistillResponse:
    """Trigger the distillation job for a specific user.

    In v0 this is a stub — the actual distillation logic lives in
    :mod:`career_coach.jobs.distillation` (Task 10). The endpoint exists so
    the HTTP interface is stable while the job is being developed.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM users WHERE user_id = $1)", user_id
        )
    if not exists:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    if not body.dry_run:
        background_tasks.add_task(_run_distillation, user_id)

    return DistillResponse(
        status="queued" if not body.dry_run else "dry_run",
        user_id=user_id,
        message=(
            "Distillation job queued in background."
            if not body.dry_run
            else "Dry run — no job scheduled."
        ),
    )


async def _run_distillation(user_id: UUID) -> None:
    """Background wrapper for the distillation job (Task 10 implementation)."""
    try:
        from career_coach.jobs.distillation import (
            run_distillation,
        )

        await run_distillation(user_id)
    except ImportError:
        logger.warning(
            "Distillation job not yet implemented (Task 10). "
            "Skipping distillation for user %s.",
            user_id,
        )
    except Exception as exc:
        logger.exception("Distillation failed for user %s: %s", user_id, exc)
