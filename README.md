# Career Coach

A longitudinal, personalized career coaching system for students (ages ~15–25).
Not a career encyclopedia — a multi-agent reasoning system with persistent
memory that coaches users over time, asking clarifying questions, running deep
research on demand, proposing challenges, and building an evolving model of who
they are.

## Status

**v0 — walking skeleton.** Single hardcoded test user, no frontend, no auth.
FastAPI backend + Postgres (Supabase-compatible) + multi-agent reasoning
pipeline (Understander → Orchestrator → Coach → Critic → Profiler).

The full architectural contract lives in [`SPEC.md`](SPEC.md). Instructions for
future Claude Code sessions live in [`CLAUDE.md`](CLAUDE.md). The ordered build
plan is in [`TASKS.md`](TASKS.md).

## Architecture at a glance

- **Three-tier memory** — structured facts (SQL rows), episodic logs (turns
  with pgvector embeddings), semantic hypotheses (append-only with weighted
  evidence). Never collapsed into one blob.
- **Agents as modules** — each agent is a Python class with typed Pydantic
  input and output. No raw strings at the interface boundary.
- **Critic gate** — on standard turns, every Coach response passes a structured
  quality check that rejects generic, ungrounded, or off-intent output. Up to
  two retries with critic feedback, then an honest escalation.
- **Provider-agnostic LLM layer** — one `LLMClient` protocol; Anthropic is the
  default, Hugging Face (via OpenAI-compatible Inference API) is the alternate.
  Swapping providers is a config-only change.
- **Everything logged** — every turn, every agent call, every token count,
  every Critic verdict lands in Postgres.

## Stack

- Python ≥ 3.11 (CI on 3.12)
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- FastAPI + Uvicorn
- Pydantic v2
- asyncpg + raw SQL migrations
- pgvector (via the `pgvector/pgvector` Postgres image locally; Supabase in
  deployment)
- Anthropic SDK for Claude; `openai` client pointed at the Hugging Face
  Inference API for the HF backend

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> career_coach && cd career_coach

# 2. Copy the env template and fill in keys
cp .env.example .env
# edit .env — at minimum set ANTHROPIC_API_KEY and SUPABASE_DB_URL

# 3. Start local Postgres (with pgvector)
docker compose up -d

# 4. Install dependencies into a uv-managed venv
uv sync --all-extras

# 5. (later tasks) run migrations and seed the test user
# uv run python scripts/run_migrations.py
# uv run python scripts/seed_test_user.py

# 6. Run the API
uv run uvicorn career_coach.api.main:app --reload
```

Check it is alive:

```bash
curl localhost:8000/health
# {"status":"ok","version":"0.0.1"}
```

Real conversation endpoints (`POST /chat`, `POST /users`,
`POST /admin/distill/{user_id}`) are wired up in Task 9.

## Tooling

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # strict type-check on src/career_coach
uv run pytest                # unit + async tests
uv run pre-commit install    # enable local git hooks
```

## Repo layout

See [`SPEC.md`](SPEC.md) §9 for the full tree. Every task in
[`TASKS.md`](TASKS.md) fleshes out another slice of it.

## Contributing

Build tasks are executed **in order, one commit per task**. Do not skip ahead.
Read [`CLAUDE.md`](CLAUDE.md) for the ground rules before touching anything.
