# ── Build-time arguments ──────────────────────────────────────────────────────
# GIT_SHA is baked at build time so every image carries the commit it was built
# from. Pass via: docker build --build-arg GIT_SHA=$(git rev-parse HEAD) ...
# Railway also injects RAILWAY_GIT_COMMIT_SHA at runtime as a fallback (see
# config.py Settings.deployment_version). Either is sufficient; baking it into
# the image makes it available even in containers with no Railway env.
ARG GIT_SHA=""

# ── Stage: runtime image ──────────────────────────────────────────────────────
FROM python:3.11-slim

# Install uv from the official distroless image.
# Pinning to a SHA is recommended for production; using :latest here for
# simplicity — pin once the deploy is stable.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Persist the build-time SHA as a runtime env var so the app can read it.
ENV GIT_SHA=${GIT_SHA}

# ── Dependency layer (cached unless pyproject.toml or uv.lock changes) ────────
COPY pyproject.toml uv.lock ./

# Sync runtime deps into a project-local venv but skip installing our own
# package yet (source not copied yet → keeps this layer cacheable).
RUN uv sync --frozen --no-dev --no-install-project

# ── Application source ────────────────────────────────────────────────────────
# Copy only what the running app needs. Tests, reports, and dev tooling are
# excluded via .dockerignore.
# README.md is required by hatchling (pyproject.toml: readme = "README.md").
COPY README.md  ./
COPY src/       ./src/
COPY config/    ./config/
COPY kb/        ./kb/
COPY migrations/ ./migrations/
COPY scripts/   ./scripts/
COPY frontend/  ./frontend/

# Install our package (career-coach) into the already-synced venv.
RUN uv sync --frozen --no-dev

# ── Runtime ───────────────────────────────────────────────────────────────────
# PORT is injected by Railway (and any other PaaS). Default to 8000 for
# local `docker run` without -e PORT=...
EXPOSE 8000

# sh -c so ${PORT:-8000} is evaluated by the shell at container start time.
CMD ["sh", "-c", "uv run uvicorn career_coach.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
