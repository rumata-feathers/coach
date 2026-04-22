# Career Coach

A longitudinal, personalized career coaching system for students (ages 15–25).
Not a career encyclopedia — a multi-agent reasoning system with persistent
memory that coaches users over time, asking clarifying questions, proposing
challenges, and building an evolving model of who they are.

## Status

**v0 complete.** Single hardcoded test user, no frontend, no auth.
FastAPI backend + Postgres (pgvector) + multi-agent pipeline.

The full architectural contract lives in [`SPEC.md`](SPEC.md).
Instructions for future sessions live in [`CLAUDE.md`](CLAUDE.md).
The ordered build plan is in [`TASKS.md`](TASKS.md).

---

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`brew install uv` or `pip install uv`)
- Docker (for local Postgres with pgvector)
- A HuggingFace account with a read token (free tier is enough for v0)

### 1. Clone and configure

```bash
git clone <repo-url> career_coach && cd career_coach
cp .env.example .env
```

Edit `.env` and set at minimum:

```
HUGGINGFACE_API_TOKEN=hf_...      # from huggingface.co/settings/tokens
SUPABASE_DB_URL=postgresql://coach:coach@localhost:5432/coach  # default local
```

### 2. Start Postgres

```bash
docker compose up -d
# Waits for healthy status, then:
```

### 3. Install dependencies

```bash
uv sync --all-extras
```

### 4. Apply migrations and seed test data

```bash
uv run python scripts/run_migrations.py
uv run python scripts/seed_test_user.py   # creates user "Alex" for testing
```

### 5. Start the API

```bash
uv run uvicorn career_coach.api.main:app --reload
```

Check it's alive:

```bash
curl localhost:8000/health
# {"status":"ok","version":"0.0.1"}
```

---

## Using the API

### Create a user

```bash
curl -s -X POST localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"display_name": "Alex"}' | jq .
# {"user_id": "..."}
```

### Send a chat turn

```bash
curl -s -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "<user_id>",
    "message": "I am trying to decide between economics and computer science."
  }' | jq .response
```

Pass `session_id` to continue an existing session:

```bash
curl -s -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user_id>", "session_id": "<session_id>", "message": "..."}' | jq .
```

### Trigger distillation

After a few turns, run the distillation job to build semantic hypotheses:

```bash
curl -s -X POST localhost:8000/admin/distill/<user_id> \
  -H "Content-Type: application/json" \
  -d '{}' | jq .
```

Or via CLI:

```bash
uv run python -m career_coach.jobs.distillation --user-id <uuid>
```

---

## Architecture overview

See [`docs/architecture.md`](docs/architecture.md) for diagrams and the key
design principles.

```
User message
    │
    ▼
Understander ──► IntentPacket
    │  ▲ clarification needed? → return question
    ▼
Orchestrator ──► flow A (fast) or B (critic gated)
    │
    ▼
Coach ──► CoachOutput
    │
    ▼  (flow B only)
Critic ──► pass? → return │  reject? → retry (max 2) → escalation
    │
    ▼
Persist turn → update session theory
    │
    ▼ (background)
Profiler ──► new facts + evidence → DB
                        │
                        ▼ (manual or cron)
                  Distillation job ──► hypotheses
```

### Three-tier memory

| Tier | Table(s) | Purpose |
|------|----------|---------|
| 1. Structured | `structured_facts` | Stable keyed attributes — age, location, etc. Updated by Profiler |
| 2. Episodic | `sessions`, `turns`, `turn_embeddings` | Full conversation log + pgvector similarity search |
| 3. Semantic | `hypotheses`, `hypothesis_evidence` | Evolving model of who the user is — built by distillation |

---

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # strict type-check on src/career_coach
uv run pytest                # unit + async tests (no DB needed)
uv run pytest tests/integration/  # integration tests (need local Postgres)
```

### Quality evaluation

```bash
uv run python scripts/run_quality_eval.py
# or save a report:
uv run python scripts/run_quality_eval.py --output reports/quality_$(date +%Y%m%d).md
```

Baseline: **19/20** (Critic: 10/10, Coach: 9/10). See [`reports/quality_baseline.md`](reports/quality_baseline.md).

---

## Repo layout

```
career_coach/
├── config/
│   ├── models.yaml          # per-agent model config (change here to swap models)
│   └── prompts/             # Jinja2 templates — all LLM prompts live here
├── migrations/              # plain SQL, applied by run_migrations.py
├── scripts/                 # run_migrations, seed_test_user, run_quality_eval
├── src/career_coach/
│   ├── agents/              # Understander, Orchestrator, Coach, Critic, Profiler
│   ├── api/                 # FastAPI app + routes
│   ├── jobs/                # distillation.py (CLI + async callable)
│   ├── llm/                 # LLMClient protocol, HF/Anthropic/Mock adapters, factory
│   ├── memory/              # structured, episodic, semantic repos
│   ├── models/              # Pydantic schemas for agent I/O and user model
│   ├── pipeline/            # turn.py — end-to-end turn orchestration
│   ├── config.py            # pydantic-settings
│   └── db.py                # asyncpg pool factory
└── tests/
    ├── agents/              # unit tests (mocked LLM)
    ├── integration/         # DB + API tests (need local Postgres)
    ├── pipeline/            # unit tests for TurnPipeline
    └── quality/             # Critic battery + run_quality_eval scenarios
```

---

## Contributing

Build tasks execute **in order, one commit per task**. Do not skip.
Read [`CLAUDE.md`](CLAUDE.md) before touching anything.
