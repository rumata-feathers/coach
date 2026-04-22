# Career Coach — v0 Specification

## 1. Purpose

A longitudinal, personalized career coaching system for students (ages ~15–25) who are uninformed about careers, pay, exit options, and how to develop toward them. **Not** an encyclopedia. A multi-agent reasoning system with persistent memory that coaches the user over weeks and months — asking clarifying questions, running deep research on demand, proposing challenges, and building an evolving model of who they are.

**v1 persona focus:** students only. Pivoters and mid-career are deferred.

## 2. Non-Negotiable Architectural Principles

These are load-bearing. Do not simplify them away:

1. **Three-tier user memory** — structured facts, episodic logs, semantic hypotheses. Never collapsed.
2. **Agents are modules, not prompts** — each agent has a typed input/output contract. The Coach is a module, not a string.
3. **The Critic can reject output** — any response can be sent back to planning if it fails quality gates. This is not optional.
4. **Uncertainty is a valid output** — any agent may return "I don't know." Fabrication is a worse failure than silence.
5. **Hypotheses are append-only with evidence** — never overwrite, always accumulate.
6. **LLM layer is provider-agnostic** — swappable via config. Default Anthropic, HF as alternate.
7. **Everything logged** — every turn, every agent call, every token count, every Critic verdict. You cannot improve what you do not measure.

## 3. v0 Scope (Walking Skeleton)

### In scope
- FastAPI backend with conversation endpoints
- Supabase Postgres with full three-tier schema
- Agents: Understander, Orchestrator, Coach, Critic, Profiler
- Provider-agnostic LLM adapter (Anthropic default)
- Session-scoped working theory
- Basic logging of turns and agent calls
- Manual-trigger distillation script (stub, mostly pass-through in v0)

### Explicitly deferred (v1+)
- Researcher agent, Devil's Advocate, Synthesizer, Connector
- God-agent
- Supervisor (fast monitor) and Evaluator (slow meta-director)
- World Knowledge Base (careers, exits, pay data)
- Human practitioner network
- n8n orchestration, cron jobs
- Frontend
- Auth (Supabase auth deferred — single hardcoded test user for v0)
- Challenge scheduling and delivery
- Interaction preferences tracker (4th micro-store)

## 4. System Architecture (v0)

```
User (HTTP)
   │
   ▼
FastAPI endpoint: POST /chat
   │
   ▼
Understander ──loop──► User (clarifying Q if needed)
   │
   │ (intent packet)
   ▼
Orchestrator ──► Flow router (A/B/C — v0 only uses A and B)
   │
   ▼
Coach ──► Critic ──► response
   │         │
   │         └── reject ──► re-plan (max 2 retries)
   │
   ▼
Profiler (async, writes to structured + episodic)
```

Flow A = single-turn response (vent, quick fact)
Flow B = standard coaching chain (Coach + Critic)
Flow C = full deliberation (**deferred to v1** — just route to B for now)

## 5. Data Model (Postgres / Supabase)

### Structured Facts
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    -- v0: hardcoded test user, auth later
    display_name TEXT
);

CREATE TABLE structured_facts (
    fact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    key TEXT NOT NULL,           -- e.g. "age", "location", "education_stage"
    value JSONB NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT,                 -- "user_stated" | "inferred" | "system"
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, key)
);

CREATE INDEX idx_facts_user ON structured_facts(user_id);
```

### Episodic Memory
```sql
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    session_theory TEXT           -- Understander's working theory for the session
);

CREATE TABLE turns (
    turn_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id),
    user_id UUID REFERENCES users(user_id),
    turn_index INT NOT NULL,
    user_message TEXT,
    assistant_message TEXT,
    intent_packet JSONB,          -- output of Understander
    flow_used TEXT,               -- "A" | "B" | "C"
    critic_verdicts JSONB,        -- array of verdicts across retries
    tokens_used JSONB,            -- {"understander": 123, "coach": 456, ...}
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_turns_session ON turns(session_id);
CREATE INDEX idx_turns_user ON turns(user_id, created_at DESC);

-- pgvector extension for episodic search
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE turn_embeddings (
    turn_id UUID PRIMARY KEY REFERENCES turns(turn_id) ON DELETE CASCADE,
    embedding vector(1536)        -- dimension depends on embedding model; adjust in code
);

CREATE INDEX idx_turn_embeddings ON turn_embeddings
    USING ivfflat (embedding vector_cosine_ops);
```

### Semantic / Hypotheses
```sql
CREATE TABLE hypotheses (
    hypothesis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    statement TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL DEFAULT 'active',  -- active | dormant | retired | superseded
    open_questions JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_updated TIMESTAMPTZ DEFAULT now(),
    last_reviewed TIMESTAMPTZ
);

CREATE TABLE hypothesis_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id UUID REFERENCES hypotheses(hypothesis_id),
    turn_id UUID REFERENCES turns(turn_id),
    source_type TEXT,             -- "user_statement" | "reflection" | "challenge_outcome" | "forced_choice"
    excerpt TEXT,
    weight REAL NOT NULL,         -- positive supports, negative contradicts
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_hyp_user ON hypotheses(user_id, status);
CREATE INDEX idx_evidence_hyp ON hypothesis_evidence(hypothesis_id);
```

### Agent Call Log (for observability)
```sql
CREATE TABLE agent_calls (
    call_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID REFERENCES turns(turn_id),
    agent_name TEXT NOT NULL,
    model_used TEXT,
    input_payload JSONB,
    output_payload JSONB,
    latency_ms INT,
    tokens_in INT,
    tokens_out INT,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

## 6. Agent Contracts

Every agent is a Python class with a typed `run()` method. Inputs and outputs are Pydantic models. No raw strings at the interface boundary.

### 6.1 Understander

**Purpose:** Compress user input + session context into a structured intent packet. Loops with user if unclear. Tracks session-scoped working theory.

**Model:** Cheap/fast (Claude Haiku default).

**Input:**
```python
class UnderstanderInput(BaseModel):
    user_message: str
    session_id: UUID
    session_theory: str | None        # current working theory
    recent_turns: list[TurnSummary]   # last 3-5 turns
    user_facts: dict                  # structured facts dump
```

**Output:**
```python
class IntentPacket(BaseModel):
    session_theory: str               # updated working theory
    turn_intent: Literal["explore", "decide", "reflect", "research", "vent"]
    specific_ask: str                 # distilled concrete ask
    emotional_tenor: str
    clarity_score: float              # 0-1
    needs_clarification: bool
    clarification_question: str | None
    inferred_constraints: list[str]
    budget_hint: Literal["quick", "standard", "deep"]
```

**Key rule:** `needs_clarification=True` only if `clarity_score < 0.5` AND `budget_hint != "quick"`. Don't block cheap turns with clarifying questions.

### 6.2 Orchestrator

**Purpose:** Given intent packet + user model state, decide which flow and which agents run. Owns the execution graph.

**Model:** No LLM in v0 — pure Python logic based on intent packet fields. (v1 may upgrade to LLM-based routing.)

**Logic (v0):**
```
if turn_intent == "vent" OR budget_hint == "quick":
    flow = "A"     # Coach only, no Critic retries
else:
    flow = "B"     # Coach + Critic with retry loop
```

### 6.3 Coach

**Purpose:** Generate the user-facing response. The only agent whose output (post-Critic) reaches the user.

**Model:** Strong (Claude Sonnet default).

**Input:**
```python
class CoachInput(BaseModel):
    intent_packet: IntentPacket
    user_facts: dict
    active_hypotheses: list[Hypothesis]
    recent_turns: list[TurnSummary]
    critic_feedback: str | None      # present on retries
```

**Output:**
```python
class CoachOutput(BaseModel):
    response_text: str
    referenced_facts: list[str]       # which user-specific facts this response grounds on
    referenced_hypotheses: list[UUID]
    proposed_challenge: Challenge | None
    uncertainty_flags: list[str]      # things the coach is explicitly unsure about
```

**Prompt principles (encoded in template):**
- MUST reference at least 2 items from `user_facts` or `active_hypotheses` by content
- If the user model is too sparse to ground the response, return a response that asks for more context rather than guessing
- Uncertainty is allowed and preferred over fabrication

### 6.4 Critic

**Purpose:** Gatekeeper. Rejects slop. Runs after Coach on Flow B.

**Model:** Cheap/fast (Claude Haiku default). Runs with a structured rubric.

**Input:**
```python
class CriticInput(BaseModel):
    coach_output: CoachOutput
    user_facts: dict
    active_hypotheses: list[Hypothesis]
    intent_packet: IntentPacket
```

**Output:**
```python
class CriticVerdict(BaseModel):
    verdict: Literal["pass", "reject"]
    failure_modes: list[Literal["generic", "ungrounded", "false_confidence", "off_intent"]]
    specific_complaints: list[str]
    suggested_fix: str | None
```

**Checks performed:**
1. **Generic test:** Would this response apply equally to any other 22-year-old? → `generic`
2. **Grounding test:** Does it reference at least 2 user-specific items from `referenced_facts` / `referenced_hypotheses`? → `ungrounded`
3. **Confidence test:** Does it assert specific claims (numbers, "most people", etc.) without sources or hedging? → `false_confidence`
4. **Intent test:** Does it address `intent_packet.specific_ask`? → `off_intent`

Max 2 retries. After 2 rejections, Coach gets an "escalation prompt" that allows returning an honest "I need more context to help here well" response.

### 6.5 Profiler

**Purpose:** Extract new facts and hypothesis-relevant signal from each turn. Writes to structured store + queues evidence for hypotheses.

**Model:** Cheap/fast (Claude Haiku default).

**Input:**
```python
class ProfilerInput(BaseModel):
    turn_id: UUID
    user_message: str
    assistant_message: str
    existing_facts: dict
```

**Output:**
```python
class ProfilerOutput(BaseModel):
    new_facts: list[FactUpdate]               # with key, value, confidence, source
    fact_updates: list[FactUpdate]            # updates to existing
    hypothesis_evidence: list[EvidenceDraft]  # to be matched to hypotheses in distillation
```

**Runs async** — not on the critical path of the response. Queue on turn write, process in background.

### 6.6 Distillation job (stub in v0)

A Python script you run manually:

```bash
python -m career_coach.jobs.distillation --user-id <uuid>
```

v0 behavior:
- Collect all `hypothesis_evidence` drafts queued since last run
- Match to existing hypotheses (simple LLM prompt, no clustering yet)
- Update confidence scores, append evidence
- Create new hypotheses for unmatched strong signals
- Write run log

v1+ will add clustering, conflict detection, coherence review, dormancy logic.

## 7. LLM Provider Adapter

All LLM calls go through `LLMClient`. Never call providers directly from agents.

```python
# career_coach/llm/client.py
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Literal["text", "json"] = "text",
    ) -> LLMResponse: ...

# Implementations:
# AnthropicClient (default)
# HuggingFaceClient (alternate, uses HF Inference API, OpenAI-compatible endpoint)
# OpenAIClient (future)
```

Model selection is **per agent** via config:
```yaml
# config/models.yaml
understander:
  provider: anthropic
  model: claude-haiku-4-5-20251001
coach:
  provider: anthropic
  model: claude-sonnet-4-6
critic:
  provider: anthropic
  model: claude-haiku-4-5-20251001
profiler:
  provider: anthropic
  model: claude-haiku-4-5-20251001
```

**Important:** Before finalizing which Claude models to use, Claude Code MUST verify the currently available model IDs by searching Anthropic's docs. Do not trust model strings baked into this spec.

Swapping to HF requires changing `provider: huggingface` and `model:` to the HF endpoint URL/model name. No agent code changes.

## 8. HTTP API (v0)

FastAPI, three endpoints only:

```
POST /users                    — create a user (v0: just returns id, no auth)
POST /chat                     — { user_id, session_id?, message } → { response, session_id, turn_id }
POST /admin/distill/{user_id}  — trigger distillation manually
```

`/chat` is synchronous for v0. No streaming yet. Request-response. Keep it simple.

## 9. Repo Layout

```
career_coach/
├── pyproject.toml
├── .env.example
├── README.md
├── docker-compose.yml          # local Postgres + pgvector for dev (Supabase used in deployment)
├── config/
│   ├── models.yaml
│   └── prompts/                # agent prompt templates (Jinja2)
│       ├── understander.j2
│       ├── coach.j2
│       ├── critic.j2
│       └── profiler.j2
├── migrations/
│   └── 001_initial.sql
├── src/
│   └── career_coach/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py         # FastAPI app
│       │   └── routes.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py         # Agent protocol, base class
│       │   ├── understander.py
│       │   ├── orchestrator.py
│       │   ├── coach.py
│       │   ├── critic.py
│       │   └── profiler.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py       # LLMClient protocol
│       │   ├── anthropic.py
│       │   └── huggingface.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── structured.py   # structured facts repo
│       │   ├── episodic.py     # turns + embeddings repo
│       │   └── semantic.py     # hypotheses repo
│       ├── models/
│       │   ├── __init__.py
│       │   ├── intent.py       # IntentPacket, etc
│       │   ├── user_model.py
│       │   └── agent_io.py     # all agent input/output schemas
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── turn.py         # the core turn loop
│       ├── jobs/
│       │   ├── __init__.py
│       │   └── distillation.py
│       ├── db.py               # Supabase/asyncpg connection
│       └── config.py           # settings loader
└── tests/
    ├── agents/
    ├── pipeline/
    └── integration/
```

## 10. The Turn Loop (core pseudocode)

```python
async def process_turn(user_id: UUID, session_id: UUID | None, user_message: str) -> TurnResult:
    # 1. Session
    session = await get_or_create_session(user_id, session_id)

    # 2. Load context
    user_facts = await structured_repo.get_all(user_id)
    active_hypotheses = await semantic_repo.get_active(user_id)
    recent_turns = await episodic_repo.get_recent(session.session_id, limit=5)

    # 3. Understander loop
    intent_packet = await understander.run(UnderstanderInput(
        user_message=user_message,
        session_id=session.session_id,
        session_theory=session.session_theory,
        recent_turns=recent_turns,
        user_facts=user_facts,
    ))

    if intent_packet.needs_clarification:
        # Return clarifying question directly; no Coach invocation
        return TurnResult(response=intent_packet.clarification_question, ...)

    # 4. Orchestrator decides flow
    flow = orchestrator.decide(intent_packet)  # "A" or "B"

    # 5. Coach + Critic loop
    critic_feedback = None
    for attempt in range(3):  # 1 try + 2 retries
        coach_out = await coach.run(CoachInput(
            intent_packet=intent_packet,
            user_facts=user_facts,
            active_hypotheses=active_hypotheses,
            recent_turns=recent_turns,
            critic_feedback=critic_feedback,
        ))
        if flow == "A":
            break  # no critic on Flow A
        verdict = await critic.run(CriticInput(
            coach_output=coach_out,
            user_facts=user_facts,
            active_hypotheses=active_hypotheses,
            intent_packet=intent_packet,
        ))
        if verdict.verdict == "pass":
            break
        critic_feedback = verdict.suggested_fix or "; ".join(verdict.specific_complaints)
    else:
        # Exhausted retries — force honest response
        coach_out = await coach.run_escalation(...)

    # 6. Persist turn
    turn = await episodic_repo.save_turn(...)

    # 7. Update session theory
    await update_session_theory(session.session_id, intent_packet.session_theory)

    # 8. Queue profiler (async, don't await response)
    asyncio.create_task(profiler.run_and_save(turn.turn_id, ...))

    return TurnResult(response=coach_out.response_text, turn_id=turn.turn_id, ...)
```

## 11. Testing Requirements

At minimum for v0:

- **Unit tests** for each agent, mocking the LLM client. Verify contract compliance (output shapes, retry behavior).
- **Integration test** for `process_turn` with a real LLM against a test user with a seeded user model. Assert at least: response produced, turn persisted, profiler ran.
- **Critic test battery:** hand-crafted Coach outputs (some good, some generic slop, some ungrounded). Assert Critic catches the bad ones at ≥ 80%. This is the quality canary.

## 12. Environment & Secrets

`.env.example` must include:
```
ANTHROPIC_API_KEY=
HUGGINGFACE_API_TOKEN=
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_DB_URL=   # direct postgres connection for migrations
EMBEDDING_MODEL=text-embedding-3-small   # or HF equivalent
LLM_PROVIDER_DEFAULT=anthropic
```

## 13. What "v0 done" looks like

A developer can:

1. Clone repo, copy `.env.example` to `.env`, fill in keys
2. Run `docker compose up` for local Postgres OR point at Supabase
3. Run migrations
4. Run `uvicorn career_coach.api.main:app`
5. POST to `/chat` with a user message
6. Receive a response that **references the seeded user's facts**
7. Query the DB and see: a turn row, agent_calls rows, structured facts updates from Profiler
8. Run `python -m career_coach.jobs.distillation --user-id <uuid>` and see hypothesis tables populate

If all eight work, v0 is done.

## 14. Explicit anti-goals for v0

Do not:
- Build a frontend
- Implement Supabase auth (single test user only)
- Add Researcher, Devil's Advocate, Synthesizer, Connector, Supervisor, Evaluator, God-agent
- Build the World Knowledge Base
- Scrape or ingest any external data sources
- Add a challenge scheduler or notifications
- Optimize for latency or cost beyond reasonable defaults
- Write a clever prompt-optimization framework — just hand-write prompts and iterate

Every hour spent on any of the above in v0 delays the moment you can *feel whether the architecture works*.
