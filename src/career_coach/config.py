"""Runtime settings loader.

Settings come from environment variables (``.env`` via ``python-dotenv``) and are
exposed through a single ``Settings`` object. Keep this module free of business
logic — it is imported everywhere and must be cheap and side-effect-free beyond
reading the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root, not the CWD, so uvicorn started
# from any directory still finds the file.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

LLMProvider = Literal["anthropic", "huggingface", "openai"]


class Settings(BaseSettings):
    """Application settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM providers
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    huggingface_api_token: str | None = Field(default=None, alias="HUGGINGFACE_API_TOKEN")

    # Web search providers
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    # Supabase / Postgres
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_KEY")
    supabase_db_url: str = Field(
        default="postgresql://coach:coach@localhost:5432/coach",
        alias="SUPABASE_DB_URL",
    )
    # Direct (non-pooler) Postgres URL for migrations and other DDL that
    # requires a full session or superuser access (e.g. CREATE EXTENSION).
    # In Supabase: Project Settings → Database → Connection String → URI.
    # Leave unset for local development; migrations fall back to supabase_db_url.
    supabase_direct_url: str | None = Field(
        default=None, alias="SUPABASE_DIRECT_URL"
    )

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    # LLM defaults
    llm_provider_default: LLMProvider = Field(default="huggingface", alias="LLM_PROVIDER_DEFAULT")

    # Runtime
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Deployment identity — used to stamp every turn row for transcript forensics.
    # GIT_SHA is baked into the Docker image at build time (see Dockerfile ARG).
    # RAILWAY_GIT_COMMIT_SHA is injected by Railway at runtime and used as a
    # fallback so un-rebuilt images still get the correct SHA.
    git_sha: str | None = Field(default=None, alias="GIT_SHA")
    railway_git_commit_sha: str | None = Field(
        default=None, alias="RAILWAY_GIT_COMMIT_SHA"
    )
    # Human-readable release tag, e.g. "v0.5.1". Optional — when set, the
    # deployment_version property combines it with the SHA.
    deployment_tag: str | None = Field(default=None, alias="DEPLOYMENT_TAG")

    @property
    def deployment_version(self) -> str:
        """Return a compact version string stamped onto every turn row.

        Priority / format:
        - Both tag and SHA set → ``"v0.5.1@a1b2c3d"``
        - SHA only             → ``"a1b2c3d"``
        - Neither              → ``"local"``

        SHA comes from ``GIT_SHA`` (baked at build time) or falls back to
        ``RAILWAY_GIT_COMMIT_SHA`` (injected by Railway at runtime).
        """
        sha = self.git_sha or self.railway_git_commit_sha
        sha_short = sha[:7] if sha else None
        if sha_short and self.deployment_tag:
            return f"{self.deployment_tag}@{sha_short}"
        if sha_short:
            return sha_short
        return "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Cached so that imports are cheap and tests can override via
    ``get_settings.cache_clear()`` when mutating the environment.
    """
    return Settings()
