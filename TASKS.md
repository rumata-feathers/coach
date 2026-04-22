# Build Tasks — Execute in Order

Each task is a single Claude Code session. Do NOT combine tasks. Commit after each.

---

## Task 1 — Project skeleton + tooling

**Goal:** Empty runnable project with tooling in place.

- [ ] Initialize repo with `pyproject.toml` (Poetry or uv — you pick, ask me)
- [ ] Dependencies: `fastapi`, `uvicorn`, `pydantic`, `asyncpg` (or Supabase client — ask), `anthropic`, `openai` (for HF compatibility), `jinja2`, `python-dotenv`, `pyyaml`, `pgvector`
- [ ] Dev dependencies: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `httpx` (for test client)
- [ ] `.env.example` with all required vars from SPEC §12
- [ ] `docker-compose.yml` with Postgres + pgvector for local dev
- [ ] Pre-commit hooks: `ruff`, `mypy`
- [ ] `README.md` with setup steps
- [ ] A trivial `GET /health` FastAPI endpoint to prove the server runs
- [ ] CI stub (github actions yaml, lint + test)

**Done when:** `uvicorn career_coach.api.main:app` runs and `curl localhost:8000/health` returns OK.

---

## Task 2 — Database schema + migrations

**Goal:** All tables from SPEC §5 exist.

- [ ] Write `migrations/001_initial.sql` with all tables from spec
- [ ] Choose migration tooling (plain SQL runner OR Alembic — ask me)
- [ ] Write a Python module `career_coach/db.py` that exposes an async connection pool
- [ ] Write a `scripts/seed_test_user.py` that creates a single test user with some structured facts (name, age=19, location=London, education_stage="undergrad")
- [ ] Integration test: connect to test Postgres, run migrations, seed user, query back

**Done when:** Migration applies cleanly. Seeded test user visible via SQL query.

---

## Task 3 — LLM adapter layer

**Goal:** Provider-agnostic LLM client. Anthropic works. HF stub works.

- [ ] `career_coach/llm/client.py` — `LLMClient` protocol + `LLMResponse` dataclass
- [ ] `career_coach/llm/anthropic.py` — implementation using the `anthropic` Python SDK
- [ ] `career_coach/llm/huggingface.py` — implementation using OpenAI-compatible client pointing at HF Inference API
- [ ] `career_coach/llm/factory.py` — reads `config/models.yaml`, returns configured client per agent name
- [ ] **Web-search first:** confirm current Claude model IDs before hardcoding any defaults
- [ ] Unit tests with a `MockLLMClient` for use by agent tests
- [ ] Integration test: real Anthropic call, assert response shape

**Done when:** `factory.get_client("coach").complete(...)` returns a real Claude response. HF variant returns a response from one open model (e.g. `meta-llama/Llama-3.1-8B-Instruct`).

---

## Task 4 — Pydantic models for agent I/O

**Goal:** Every schema from SPEC §6 exists as a typed Pydantic model.

- [ ] `career_coach/models/intent.py` — `IntentPacket`
- [ ] `career_coach/models/agent_io.py` — all agent input/output models
- [ ] `career_coach/models/user_model.py` — `FactUpdate`, `Hypothesis`, `EvidenceDraft`, `TurnSummary`, etc.
- [ ] Tests: round-trip serialization for every model

**Done when:** `mypy --strict` passes on `src/career_coach/models/`.

---

## Task 5 — Memory repositories

**Goal:** Clean async data-access layer for the three tiers.

- [ ] `career_coach/memory/structured.py` — `StructuredFactsRepo` with `get_all`, `upsert`, `bulk_upsert`
- [ ] `career_coach/memory/episodic.py` — `EpisodicRepo` with `create_session`, `save_turn`, `get_recent`, `search_similar` (pgvector)
- [ ] `career_coach/memory/semantic.py` — `SemanticRepo` with `get_active`, `create_hypothesis`, `append_evidence`, `update_confidence`
- [ ] `career_coach/memory/embeddings.py` — embedding generation helper (use an Anthropic/OpenAI/HF embedding endpoint, to decide)
- [ ] Integration tests against real test Postgres

**Done when:** All repos round-trip correctly against test DB.

---

## Task 6 — Agent base + Understander

**Goal:** First real agent. Sets the pattern for all others.

- [ ] `career_coach/agents/base.py` — `Agent` base class with `name`, `llm_client`, `prompt_template`, `run()`
- [ ] `config/prompts/understander.j2` — actual prompt template, iterated
- [ ] `career_coach/agents/understander.py` — the agent itself
- [ ] Prompt must enforce JSON output matching `IntentPacket` schema
- [ ] Unit tests with `MockLLMClient`: verify it parses correctly, fails gracefully on malformed JSON
- [ ] Integration test: real LLM, verify `IntentPacket` validates

**Done when:** Given a user message + empty context, Understander returns a valid `IntentPacket`.

---

## Task 7 — Orchestrator + Coach + Critic

**Goal:** The core reasoning chain works.

- [ ] `career_coach/agents/orchestrator.py` — pure Python, no LLM, just flow selection logic
- [ ] `config/prompts/coach.j2` — Coach prompt, must reference facts + hypotheses
- [ ] `career_coach/agents/coach.py`
- [ ] `config/prompts/critic.j2` — Critic rubric prompt, returns structured verdict JSON
- [ ] `career_coach/agents/critic.py`
- [ ] Critic test battery: hand-craft 10 Coach outputs (5 good, 5 slop across failure modes). Critic should correctly classify ≥ 8/10.
- [ ] Unit tests for Coach + Critic with mocked LLM

**Done when:** Critic test battery passes. Integration test: Coach produces response, Critic evaluates it, retry loop works.

---

## Task 8 — Profiler + turn pipeline

**Goal:** End-to-end turn processing works.

- [ ] `config/prompts/profiler.j2`
- [ ] `career_coach/agents/profiler.py`
- [ ] `career_coach/pipeline/turn.py` — implements the `process_turn` pseudocode from SPEC §10
- [ ] Profiler runs as a background task (`asyncio.create_task`) — not blocking the response
- [ ] Integration test: one turn, then verify DB has: turn row, agent_calls rows, at least one new structured fact written by Profiler

**Done when:** End-to-end turn from user message to persisted turn works.

---

## Task 9 — FastAPI surface

**Goal:** HTTP endpoints from SPEC §8.

- [ ] `POST /users` — creates user, returns ID
- [ ] `POST /chat` — processes a turn, returns response
- [ ] `POST /admin/distill/{user_id}` — triggers distillation script
- [ ] Error handling: any agent failure returns 500 with a useful error shape, never crashes the server
- [ ] API tests using `httpx.AsyncClient`

**Done when:** Full conversation flow works over HTTP with the seeded user.

---

## Task 10 — Distillation stub

**Goal:** Runnable distillation script, v0 behavior only.

- [ ] `career_coach/jobs/distillation.py` — CLI entry point: `python -m career_coach.jobs.distillation --user-id <uuid>`
- [ ] For v0: fetch queued `EvidenceDraft` entries, use an LLM call to match each to existing hypotheses or create new ones with low confidence
- [ ] Writes `hypothesis_evidence` rows and updates `hypotheses.confidence`
- [ ] Prints a summary of what was updated
- [ ] Integration test: seed a user with 5 turns, run distillation, assert at least 1 hypothesis exists afterward

**Done when:** Running the script against the seeded user after a few chat turns produces real hypothesis rows.

---

## Task 11 — Quality harness

**Goal:** Make it measurable.

- [ ] `tests/quality/` directory with a "golden set" of 20 input scenarios + expected behavior (not exact outputs — categorical assertions like "response references user age", "Critic rejects this, not that")
- [ ] `scripts/run_quality_eval.py` — runs the golden set, produces a report
- [ ] Baseline report checked in so we can compare over time

**Done when:** Running the quality eval produces a reproducible pass/fail summary.

---

## Task 12 — Documentation pass

**Goal:** Someone else can pick this up.

- [ ] `README.md` updated with full setup + run instructions
- [ ] `docs/architecture.md` — the diagrams and the key principles
- [ ] `docs/adding_an_agent.md` — step-by-step for adding the next agent (will matter for v1 when Researcher comes in)
- [ ] Every prompt template has a comment block explaining its design intent

**Done when:** A fresh developer can set up and run the whole system from docs alone.

---

## After v0

Do not start v1 without explicit go-ahead. v1 adds: Researcher, Devil's Advocate, Synthesizer, World KB, Supervisor, Evaluator, real distillation clustering, challenge scheduling.
