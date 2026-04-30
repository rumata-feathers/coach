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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Cached so that imports are cheap and tests can override via
    ``get_settings.cache_clear()`` when mutating the environment.
    """
    return Settings()
