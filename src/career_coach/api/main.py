"""FastAPI application entry point.

For Task 1 this only exposes a ``GET /health`` liveness endpoint. Later tasks
register the real conversation routes from :mod:`career_coach.api.routes`.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from career_coach import __version__
from career_coach.config import get_settings

logger = logging.getLogger("career_coach.api")


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Using a factory keeps tests (and future ASGI workers) free to spin up
    isolated instances without relying on import-time side effects.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="Career Coach",
        version=__version__,
        description="v0 walking skeleton — see SPEC.md for the architectural contract.",
    )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        """Liveness probe. Returns a constant payload when the process is up."""
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
