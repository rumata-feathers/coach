"""FastAPI application entry point.

Exposes:
  GET  /health  — liveness probe (no DB dependency; always fast)
  (full conversation routes registered via :mod:`career_coach.api.routes`)

Lifespan events:
  startup  — initialise the asyncpg connection pool so the first request is
             not penalised with cold-pool latency, and to fail fast if the DB
             is unreachable at deploy time.
  shutdown — close the pool gracefully so Railway's SIGTERM drain completes
             without leaving idle server-side connections.

The app reads all configuration from environment variables via
:func:`~career_coach.config.get_settings`.  There is no hard dependency on
docker-compose: set ``SUPABASE_DB_URL`` in the environment (or Railway's env
var panel) to point at any reachable Postgres instance.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from career_coach import __version__
from career_coach.api.routes import router
from career_coach.config import get_settings
from career_coach.db import close_pool, get_pool

logger = logging.getLogger("career_coach.api")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """ASGI lifespan: open pool on startup, close on shutdown."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info(
        "Career Coach %s starting (git=%s).",
        __version__,
        settings.git_sha or "local",
    )

    # Connect to DB pool at startup so we fail fast if the DB is unreachable
    # rather than discovering it on the first live request.
    await get_pool()
    logger.info("DB pool ready.")

    yield  # application is live

    await close_pool()
    logger.info("DB pool closed.")


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Using a factory keeps tests (and future ASGI workers) free to spin up
    isolated instances without relying on import-time side effects.
    """
    app = FastAPI(
        title="Career Coach",
        version=__version__,
        description="Longitudinal, personalised career coaching — see SPEC.md.",
        lifespan=_lifespan,
    )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        """Liveness probe.

        Returns a constant 200 whenever the process is up.  Deliberately does
        **not** check the DB connection — a transient DB blip must not cause
        Railway to restart a healthy app instance.  DB reachability is
        validated at startup (see lifespan above).
        """
        settings = get_settings()
        return {
            "status": "ok",
            "version": __version__,
            "git_sha": settings.git_sha or "unknown",
        }

    app.include_router(router)
    return app


app = create_app()
