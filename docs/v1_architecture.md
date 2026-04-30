# Coach v1 — Architecture Reference

> **Audience:** Engineers onboarding to the codebase or returning after a break.
> Read this before touching `pipeline/turn.py` or any agent.

---

## 1. What the system does

Coach is a longitudinal AI career coach for UK students and early-career
professionals. It remembers everything the user has ever told it, builds a
structured model of who they are, and produces advice calibrated to that model.

The differentiating bet: **memory + deliberation over facts, not chat**.
A generic LLM wrapper gives the same answer to "should I do a PhD?" to
everyone. Coach gives a different answer to a 22-year-old Physics graduate in
London who has expressed risk aversion and a 23-year-old CS student in
Edinburgh who wants to stay in academia.

---

## 2. High-level flow

Every user message passes through a **LangGraph StateGraph**:

```
load_context
    │
understander
    │
    ├─ needs_clarification? ──► clarification_reply ──► END
    │
orchestrator
    │
    ├─ onboarding ──────────────────────────────────────────┐
    │                                                       │
    ├─ Flow A/B ──► coach ──┬─ Flow A ────────────────────►│
    │                       └─ Flow B ──► critic ──────────►│
    │                                  (loop ≤3)            │
    │                                                       │
    └─ Flow C ──► researcher ─┐                             │
                  coach ──────┴──► devils_advocate          │
                                       │                    │
                                   synthesizer              │
                                       │                    │
                                    critic ────────────────►│
                                  (loop ≤2)                 │
                                                            ▼
                                                       supervisor
                                                            │
                                                         persist
                                                            │
                                                     session_update
                                                            │
                                                   profiler_dispatch
                                                            │
                                                           END
```

The **Orchestrator** selects the flow based on three signals:

| Condition | Flow |
|-----------|------|
| New user (no structured facts) | onboarding |
| `budget_hint == "quick"` or `turn_intent == "vent"` | A (Coach only) |
| (`budget_hint == "deep"` or `turn_intent == "decide"`) AND `fact_count ≥ 5` | C (deliberation) |
| Everything else | B (Coach + Critic) |

---

## 3. Agents

Every agent is a class in `src/career_coach/agents/`, extends `Agent` (base
class), takes a typed Pydantic input, returns a typed Pydantic output, and
logs every call to `agent_calls` via `log_call()`. Prompts live in
`config/prompts/*.j2` — never hardcoded in Python.

### 3.1 Understander
**Input:** raw user message + session context + recent turns + user facts  
**Output:** `IntentPacket` — `turn_intent`, `specific_ask`, `budget_hint`,
`needs_clarification`, `session_theory`, `emotional_tenor`

Runs on every turn. Determines whether to ask the user a clarifying question
instead of producing advice. When `needs_clarification=True`, the pipeline
short-circuits — no other agents run.

**Model:** Qwen3-32B (temperature 0.2)

---

### 3.2 Orchestrator
**Input:** `IntentPacket` + `is_new_user` flag + `fact_count`  
**Output:** `Flow` literal — `"A" | "B" | "C" | "onboarding"`

Pure logic, no LLM. Implements the routing table above.

---

### 3.3 Coach
**Input:** `CoachInput` — intent, user facts, active hypotheses, recent turns,
optional `critic_feedback`  
**Output:** `CoachOutput` — `response_text`, `referenced_facts`,
`referenced_hypotheses`, `uncertainty_flags`

The primary voice of the system. On Flow A it's the only agent. On Flow B it
runs in a retry loop driven by Critic feedback. On Flow C it runs in parallel
with Researcher (LangGraph fan-out) and its output flows into DA → Synthesizer.

**Model:** Qwen3-235B-A22B (temperature 0.7)

---

### 3.4 Researcher *(Flow C only)*
**Input:** `ResearcherInput` — the user's specific question, user facts, depth
**Output:** `ResearchBrief` — `findings[]`, `research_plan`, `brief_id` (FK to
`research_briefs` table)

Two-phase:
1. **Planner** (Qwen3-32B, cheap): decides which search queries to issue.
2. **Synthesiser** (Qwen3-235B-A22B): turns raw Tavily search results + KB
   entries into typed `Finding` objects with `source_url`, `citation_index`,
   `claim`, `confidence`.

The Researcher initialises lazily — the Tavily API key is not required unless
Flow C actually fires. Safe to run in dev with `TAVILY_API_KEY` unset.

**Models:** Qwen3-32B (planner) + Qwen3-235B-A22B (synthesis)

---

### 3.5 Devil's Advocate *(Flow C only)*
**Input:** Coach output + Researcher brief + user facts + hypotheses  
**Output:** `DevilsAdvocateOutput` — `counter_points[]`, `agrees_with_coach`

Fan-in point: DA runs only after BOTH `researcher` and `coach` complete.
Provides structured challenges to the coach's position. If it agrees with the
coach (`agrees_with_coach=True`), the Synthesiser can produce an empty
`surfaced_tradeoffs` list without the Critic flagging `uncontested`.

**Model:** Qwen3-235B-A22B (temperature 0.3)

---

### 3.6 Synthesizer *(Flow C only)*
**Input:** Coach + DA + ResearchBrief + user facts + optional `critic_feedback`  
**Output:** `SynthesizedResponse` — `response_text` with inline `[N]`
citations, `referenced_facts[]`, `referenced_findings[]`, `surfaced_tradeoffs[]`,
`integrated_from[]`, optional `chart_specs[]`

Produces the user-facing response for Flow C. Must cite every claim with
`[N]` pointing at a finding in the research brief. Charts (≤1 per response
in v1) must have `source_citation_indices` pointing at their data source.

Runs in a retry loop with the Critic (max 2 attempts total).

**Model:** Qwen3-235B-A22B (temperature 0.5)

---

### 3.7 Critic
**Input:** `CriticInput` — coach/synthesizer output, user facts, hypotheses,
intent, `is_flow_c` flag  
**Output:** `CriticVerdict` — `verdict` ("pass"/"reject"),
`failure_modes[]`, `specific_complaints[]`, `suggested_fix`

Runs after every Coach call (Flow B) or every Synthesizer call (Flow C).
Eight possible `CriticFailureMode` values:

| Mode | Description |
|------|-------------|
| `generic` | Response is generic boilerplate |
| `ungrounded` | Claims not anchored to user facts |
| `false_confidence` | Overconfident about uncertain facts |
| `off_intent` | Misses the specific ask |
| `unintegrated` *(Flow C)* | Doesn't cite all available sources |
| `uncontested` *(Flow C)* | DA disagreed but no tradeoffs surfaced |
| `chart_uncited` *(Flow C)* | Chart has no source citation indices |
| `chart_data_invented` *(Flow C)* | Chart data not traceable to findings |

**Model:** Qwen3-32B (temperature 0.0 — deterministic)

---

### 3.8 Supervisor
**Input:** `SupervisorInput` — user message, response text, user facts,
specific ask, flow used  
**Output:** `SupervisorEvent` — `action` ("pass"/"warn"/"retry"/"block"),
`event_type`, `severity`, `details`, `scripted_override`

Runs on **every** flow as the final pre-response check. Three checks:

| Check | Triggers | Action |
|-------|----------|--------|
| `fact_contradiction` | Response contradicts a high-confidence fact | retry → warn |
| `off_topic` | Response clearly misses specific_ask | retry → warn |
| `unsafe` | Crisis keyword in user message OR response | block (scripted override) |

**Design principles:**
- **Fail-open:** parse errors return `pass` so a broken Supervisor never
  silences the user.
- **Early-return:** empty responses return `pass` immediately (latency budget).
- **One retry before block/warn:** on `retry`, the pipeline re-runs the
  appropriate agent (Coach for A/B, Synthesizer for C) with Supervisor feedback.
  If the retry still trips, falls through to `warn`.
- Crisis keywords maintained as a Jinja2 variable in `supervisor.j2` — easy
  to extend without touching Python code.

Non-pass events are persisted to `supervisor_events` AFTER the turn row is
saved (avoids FK violation).

**Model:** Qwen3-32B (temperature 0.0)

---

### 3.9 Profiler
**Input:** user message + assistant response + existing structured facts  
**Output:** `ProfilerOutput` — `fact_updates[]`, `hypothesis_evidence[]`

Runs as a **background task** — `asyncio.create_task()`, not awaited.
Extracts new facts and hypothesis evidence from the conversation, writes them
to `structured_facts` and the semantic hypothesis store.

**Model:** Qwen3-32B (temperature 0.1)

---

## 4. Memory tiers

Three tiers; never collapsed.

| Tier | Store | Description |
|------|-------|-------------|
| **Structured** | `structured_facts` (Postgres) | Key-value facts extracted by Profiler. High-confidence only. Used as "what we know" in every agent prompt. |
| **Episodic** | `turns` + `sessions` (Postgres) | Verbatim turn history. Last 5 turns loaded each call. |
| **Semantic** | `hypotheses` (Postgres + pgvector) | Active theories about the user — e.g. "gravitates toward analytical work". Coach references these; Profiler updates confidence. |

---

## 5. Database schema (v1)

Migration `004_v1_agents.sql` added:

```sql
-- Stores Researcher web+KB synthesis for later audit
CREATE TABLE research_briefs (
  brief_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES users(user_id),
  question        TEXT NOT NULL,
  findings_json   JSONB NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Supervisor audit log
CREATE TABLE supervisor_events (
  event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  turn_id         UUID NOT NULL REFERENCES turns(turn_id),
  event_type      TEXT NOT NULL,   -- fact_contradiction | off_topic | unsafe
  severity        TEXT NOT NULL,   -- low | med | high
  details         JSONB,
  action_taken    TEXT NOT NULL,   -- warn | retry | block
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Columns added to turns:
-- research_brief_id  UUID  FK → research_briefs
-- synthesizer_output JSONB Flow C SynthesizedResponse
-- devils_advocate_output JSONB
-- chart_specs        JSONB
```

---

## 6. LangGraph graph structure

`pipeline/turn.py` compiles a LangGraph `StateGraph(TurnState)` once at
`TurnPipeline.__init__()`. `process_turn()` is a thin wrapper:

```python
result_state = await self._graph.ainvoke(initial_state)
```

**`TurnState` TypedDict** carries all intermediate agent outputs. The
`critic_verdicts` field uses `Annotated[list[dict], operator.add]` so each
Critic node **appends** (not overwrites) across retry iterations.

**Key edges:**

```python
# Flow C parallel fan-out: returns a list → both run simultaneously
def _route_orchestrator(state):
    if state["flow"] == "C":
        return ["researcher", "coach"]   # LangGraph super-step
    ...

# Fan-in at devils_advocate: waits for both researcher and coach edges
g.add_edge("researcher", "devils_advocate")
# coach → devils_advocate (for Flow C) via conditional edge

# Critic retry cycles
# Flow B: coach ⇆ critic  (bounded by coach_attempt counter)
# Flow C: synthesizer ⇆ critic  (bounded by synth_attempt counter)
```

**No checkpointing, no HITL, no tool-calling routing** — per spec §9 anti-goals.

---

## 7. Key invariants

These must never be broken:

1. **Prompts in `config/prompts/*.j2`, never in Python.** Agent code only calls
   `self.render_prompt("name.j2", **kwargs)`.

2. **Every agent boundary is typed Pydantic.** No `dict[str, Any]` at agent
   I/O. `CriticInput`, `CoachInput`, `SupervisorInput`, etc. are the contract.

3. **Supervisor fails open.** A broken Supervisor returns `pass`. It never
   blocks a user response due to its own failure.

4. **Supervisor events persist after turns.** FK ordering: save turn → save
   supervisor event. Never before.

5. **Researcher is lazy.** `Researcher` is created on first Flow C call, not
   at pipeline startup. `TAVILY_API_KEY` not required at startup.

6. **Flow C has a 25-second budget.** `asyncio.timeout(25.0)` wraps the full
   C inner pipeline. Timeout falls back to plain Coach output.

7. **Profiler is fire-and-forget.** It runs after the turn is persisted.
   Never awaited. A Profiler failure never blocks the user response.

8. **Memory tiers are never collapsed.** Structured, episodic, and semantic
   stores remain three separate repositories. No merging.

---

## 8. Configuration

All tuneable knobs live in two files:

| File | Controls |
|------|----------|
| `config/models.yaml` | Per-agent model ID, temperature, max_tokens |
| `config/prompts/*.j2` | All LLM prompts (Jinja2 templates) |

Environment variables (set in `.env`):

| Variable | Required for |
|----------|-------------|
| `SUPABASE_DB_URL` | All turns (DB persistence) |
| `HUGGINGFACE_API_TOKEN` | All LLM calls |
| `TAVILY_API_KEY` | Flow C (Researcher web search) |
| `ANTHROPIC_API_KEY` | LLM-as-judge second grader only |

---

## 9. Observability

Every LLM call writes a row to `agent_calls`:

```sql
SELECT agent_name, model_used, latency_ms, tokens_in, tokens_out,
       error, retry_count, fallback_reason
FROM agent_calls
WHERE turn_id = $1
ORDER BY created_at;
```

Useful diagnostic queries:

```sql
-- Flow distribution
SELECT flow_used, count(*), round(avg(latency_ms)) AS avg_ms
FROM turns JOIN agent_calls USING (turn_id)
GROUP BY flow_used;

-- What Supervisor caught
SELECT event_type, action_taken, count(*)
FROM supervisor_events
GROUP BY 1, 2;

-- Chart emission rate (last 14 days)
SELECT
  count(*) FILTER (WHERE chart_specs != '[]') AS turns_with_charts,
  count(*) AS total
FROM turns
WHERE flow_used IN ('B','C') AND created_at > now() - interval '14 days';
```

---

## 10. Quality batteries

| Battery | File | Threshold | Requires |
|---------|------|-----------|---------|
| Critic unit battery (Flow B) | `tests/quality/test_critic_battery.py` | ≥8/10 | `ANTHROPIC_API_KEY` |
| Critic unit battery (Flow C) | same file | ≥4/5 | `ANTHROPIC_API_KEY` |
| Supervisor red-team | `tests/quality/test_supervisor_battery.py` | ≥8/15 triggers, 0/5 FP | `ANTHROPIC_API_KEY` |
| Onboarding battery | `tests/quality/test_onboarding_battery.py` | ≥8/10 | `HUGGINGFACE_API_TOKEN` |
| Flow C deep-decision eval | `scripts/run_flow_c_eval.py` | ≥17/20, ≥5 charts, 0 bad charts | all three keys |

---

## 11. Adding a new agent

1. Create `src/career_coach/agents/my_agent.py` extending `Agent`.
2. Add input/output types to `src/career_coach/models/agent_io.py`.
3. Write prompt to `config/prompts/my_agent.j2`.
4. Add model config block to `config/models.yaml`.
5. Wire into `pipeline/turn.py`: add node in `_build_graph()`, add edges,
   add node method `_node_my_agent()`.
6. Write unit tests in `tests/agents/test_my_agent.py`.
7. Run `mypy --strict src/career_coach/` — must pass.
8. Run `ruff check src/` — must pass.
