"""Conversation and admin API routes.

Endpoints:
  POST /users            — create a user, return their UUID
  POST /chat             — process one turn, return the assistant response
  POST /admin/distill/{user_id} — trigger the distillation job for a user
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from career_coach.db import get_pool
from career_coach.llm.factory import LLMFactory
from career_coach.pipeline.turn import TurnPipeline

logger = logging.getLogger("career_coach.api.routes")

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
async def create_user(body: CreateUserRequest) -> CreateUserResponse:
    """Create a new user and return their UUID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            "INSERT INTO users (display_name) VALUES ($1) RETURNING user_id",
            body.display_name,
        )
    logger.info("Created user %s ('%s').", user_id, body.display_name)
    return CreateUserResponse(user_id=user_id)


@router.post("/chat", response_model=ChatResponse, tags=["conversation"])
async def chat(body: ChatRequest) -> ChatResponse:
    """Process one user turn and return the assistant response.

    Runs the full agent pipeline: Understander → Orchestrator → Coach →
    Critic (flow B) → persists turn → background Profiler.

    Errors in the agent pipeline are surfaced as HTTP 500 with a structured
    ``detail`` payload.
    """
    # Verify user exists
    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM users WHERE user_id = $1)", body.user_id
        )
    if not exists:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found.")

    try:
        pipeline = TurnPipeline(_get_factory())
        result = await pipeline.process_turn(
            user_id=body.user_id,
            user_message=body.message,
            session_id=body.session_id,
        )
    except Exception as exc:
        logger.exception("Pipeline error for user %s: %s", body.user_id, exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "pipeline_error",
                "message": str(exc),
                "user_id": str(body.user_id),
            },
        ) from exc

    return ChatResponse(
        response=result.response,
        turn_id=result.turn_id,
        session_id=result.session_id,
        clarification_only=result.clarification_only,
    )


@router.post(
    "/admin/distill/{user_id}",
    response_model=DistillResponse,
    tags=["admin"],
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
