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
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from career_coach import __version__
from career_coach.api.limiter import limiter
from career_coach.api.routes import router
from career_coach.config import get_settings
from career_coach.db import close_pool, get_pool

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

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
    dsn = settings.supabase_db_url
    # Log the host (never the password) so misconfigured DSNs are obvious.
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(dsn)
        _host_hint = f"{_p.hostname}:{_p.port}"
    except Exception:
        _host_hint = "<unparseable DSN>"
    logger.info("Connecting to DB at %s …", _host_hint)
    try:
        await get_pool()
    except Exception as exc:
        logger.critical(
            "DB pool failed to initialise — check SUPABASE_DB_URL (target: %s). "
            "Error: %s",
            _host_hint,
            exc,
        )
        raise
    logger.info("DB pool ready.")

    yield  # application is live

    await close_pool()
    logger.info("DB pool closed.")


async def _handle_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a structured JSON 429 instead of slowapi's default plain-text."""
    retry_after = 3600
    with suppress(Exception):
        retry_after = exc.limit.limit.granularity.seconds
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit", "retry_after_seconds": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


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

    # ── Rate limiting ──────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)  # type: ignore[arg-type]

    # Wide-open CORS for the friend-test phase: no auth, no cookies, so there
    # is nothing sensitive to protect. Must be tightened to an explicit origin
    # allowlist before any authentication or session cookies are introduced.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
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
            "git_sha": settings.deployment_version,
        }

    app.include_router(router)

    # Static frontend — mounted LAST so API routes always take precedence.
    # html=True makes FastAPI serve index.html for / and any unknown path,
    # which is the standard SPA behaviour.
    # CORS is wide-open for the friend-test phase (no auth, no cookies).
    # Tighten to explicit origins before v2 / any auth is added.
    if _FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
