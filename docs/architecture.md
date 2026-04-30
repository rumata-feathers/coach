# Architecture

Career Coach is a **longitudinal multi-agent system** built around three core
ideas:

1. **Persistent memory** — the system remembers users across sessions and builds
   a structured model of who they are.
2. **Quality gate** — responses are never sent without passing a structured Critic
   check.
3. **Provider-agnostic LLM layer** — swapping models or providers is a config
   change, not a code change.

---

## Turn lifecycle

```
User message
    │
    ▼
┌──────────────────┐
│   Understander   │  Parses intent, updates session theory, detects clarity
└──────────────────┘
    │
    │ needs_clarification = True?
    ├──────────────────────────────► return clarifying question (no Coach)
    │
    ▼ IntentPacket
┌──────────────────┐
│   Orchestrator   │  Routes to flow A (no Critic) or B (Critic gated)
└──────────────────┘
    │
    ▼ flow decision
┌──────────────────┐
│      Coach       │  Generates grounded, personalised response
└──────────────────┘
    │
    │ flow B only ──────────────────────────────────────────────────────┐
    ▼                                                                   │
┌──────────────────┐   reject + feedback                               │
│      Critic      │ ─────────────────► retry Coach (max 2 times)      │
└──────────────────┘                                                   │
    │ pass                              exhausted? ────────────────────►│
    │                                                    escalation     │
    │◄──────────────────────────────────────────────────────────────────┘
    ▼
Persist turn (DB)
    │
    ▼ (background)
┌──────────────────┐
│     Profiler     │  Extracts new facts + hypothesis evidence from the turn
└──────────────────┘
                        │
                        ▼ (manual trigger or cron)
              ┌──────────────────┐
              │   Distillation   │  Matches evidence to hypotheses; creates new ones
              └──────────────────┘
```

---

## Memory tiers

### Tier 1 — Structured facts (`structured_facts`)

Stable, keyed attributes about the user. Examples: `age`, `location`,
`education_stage`, `interests`. Written by the Profiler, read by all agents.

Stored as JSONB so values can be any JSON-serialisable type (int, string, list).

### Tier 2 — Episodic memory (`sessions`, `turns`, `turn_embeddings`)

Every turn is logged in full. The `turn_embeddings` sidecar table stores a
384-dimension pgvector embedding of each turn (using HF `all-MiniLM-L6-v2`) for
similarity search across the user's history.

### Tier 3 — Semantic hypotheses (`hypotheses`, `hypothesis_evidence`)

An append-only model of the user's personality, values, and patterns. Hypotheses
are never overwritten — evidence is accumulated and confidence is computed from
all evidence. The distillation job drains the evidence queue and updates
confidence scores.

---

## Agent contracts

Every agent follows this pattern:

```python
class SomeAgent(Agent):
    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("some_agent", factory)   # name matches models.yaml key

    async def run(self, input_data: SomeInput) -> SomeOutput:
        prompt = self.render_prompt("some_agent.j2", **ctx)
        response = await self.complete(messages, response_format="json")
        output = _parse_output(response.text)
        await self.log_call(...)
        return output
```

Key rules:
- **Typed boundaries** — inputs and outputs are Pydantic models. No raw dicts.
- **Prompts in files** — every prompt is a Jinja2 template in `config/prompts/`.
  Hardcoding prompt text in Python is explicitly banned.
- **Observability** — every agent call writes a row to `agent_calls` via
  `self.log_call()`, including tokens in/out, latency, and any error.
- **Retry-once** — if parsing fails, each agent appends a corrective message and
  retries the LLM call once before falling back.  `retry_count` (0 or 1) and
  `fallback_reason` (`"retry_exhausted"` | `null`) are logged to `agent_calls`.
- **Fail safe** — agents that own the user-facing response (Coach, Understander)
  return a safe fallback on parse failure rather than crashing.

### `agent_calls` columns (migration 003)

| Column | Type | Notes |
|--------|------|-------|
| `retry_count` | `INT DEFAULT 0` | Number of retry attempts (0 = first call succeeded) |
| `fallback_reason` | `TEXT NULL` | `retry_exhausted` when both attempts failed; `null` on the happy path |

Query fallback rate:
```sql
SELECT agent_name,
       SUM(retry_count)                                   AS total_retries,
       COUNT(*) FILTER (WHERE fallback_reason IS NOT NULL) AS fallbacks,
       COUNT(*)                                            AS total_calls
FROM agent_calls
WHERE created_at > now() - interval '24 hours'
GROUP BY agent_name;
```

---

## LLM layer

```
LLMClient (Protocol)
    ├── AnthropicClient    — uses anthropic SDK
    ├── HuggingFaceClient  — uses openai SDK pointed at HF router
    └── MockLLMClient      — deterministic queue for unit tests
```

`LLMFactory` reads `config/models.yaml` and caches one client per provider.
Each agent only knows about `LLMClient` — never about a specific SDK.

### Switching providers

Change `provider:` and `model:` in `config/models.yaml`. No Python changes.

```yaml
# Before (Anthropic)
coach:
  provider: anthropic
  model: claude-sonnet-4-6

# After (HuggingFace)
coach:
  provider: huggingface
  model: Qwen/Qwen3-235B-A22B
```

---

## Orchestrator flows

| Flow | When | Agents | Critic | Latency budget |
|------|------|--------|--------|----------------|
| A | `budget_hint=quick` OR `turn_intent=vent` | Coach only | No | Fast |
| B | Everything else (standard) | Coach | Yes (≤2 retries) | ~5–10 s |
| C | `budget_hint=deep` OR `turn_intent=decide` AND ≥5 structured facts | Researcher ∥ Coach → DA → Synthesizer | Yes on Synthesizer (≤2 retries) | ≤25 s p95 |
| onboarding | New user (< 5 facts AND < 3 turns) | OnboardingPipeline | No | Fast |

---

## Quality gate (Critic)

The Critic checks every Coach response (Flow B) and every Synthesizer response
(Flow C) against a rubric of failure modes. On `reject` the Critic provides a
`suggested_fix` used on the next attempt. After 2 rejections, the pipeline
escalates to an honest "I need more context" response.

**Flows A and onboarding bypass the Critic** — they are on the fast path and
the Critic's latency cost is not justified there.

### Flow A/B failure modes

| Mode | Meaning |
|------|---------|
| `generic` | Could apply to any user — no personalisation |
| `ungrounded` | < 2 user-specific items from facts + hypotheses |
| `false_confidence` | Specific numbers/claims without hedging or sources |
| `off_intent` | Doesn't address what the user actually asked |

### Flow C additional failure modes

| Mode | Meaning |
|------|---------|
| `unintegrated` | Synthesizer's `integrated_from` doesn't include all available sources |
| `uncontested` | DA disagreed but no tradeoff surfaced in the response |
| `chart_uncited` | Chart present without `source_citation_indices` populated |
| `chart_data_invented` | Chart contains values absent from any cited finding |

---

## Database schema

See `migrations/` for the authoritative schema. Migrations are applied in
lexical order by `scripts/run_migrations.py`. Key indexes:

- `turn_embeddings` has an `ivfflat` cosine index for similarity search
- `structured_facts` has a `UNIQUE (user_id, key)` constraint for upsert
- `hypothesis_evidence` uses `hypothesis_id IS NULL` as the distillation queue

### Turns columns added in migration 004

| Column | Type | Notes |
|--------|------|-------|
| `research_brief_id` | `UUID NULL` | FK → `research_briefs.brief_id`. Set when Flow C Researcher ran. |
| `synthesizer_output` | `JSONB NULL` | Full `SynthesizedResponse` payload. Flow C only. |
| `devils_advocate_output` | `JSONB NULL` | Full `DevilsAdvocateOutput` payload. Flow C only. |
| `chart_specs` | `JSONB DEFAULT '[]'` | List of `ChartSpec` objects emitted by Coach (Flow B) or Synthesizer (Flow C). Empty array on Flows A/onboarding. |

### `research_briefs` (migration 004)

One row per Researcher invocation (every Flow C turn, including graceful
failures where `findings` is an empty array). Linked to `turns` via
`turns.research_brief_id`.

| Column | Type | Notes |
|--------|------|-------|
| `brief_id` | `UUID PK` | |
| `turn_id` | `UUID FK` → `turns` | |
| `question` | `TEXT` | The research question derived from the intent packet. |
| `findings` | `JSONB` | Array of `Finding` objects (each with claim, confidence, citations). |
| `caveats` | `JSONB` | Free-text caveats the Researcher returned (e.g. "research truncated"). |
| `next_questions` | `JSONB` | Follow-up questions suggested by the Researcher. |
| `used_kb_files` | `JSONB` | KB file paths that contributed to findings. |
| `web_searches_run` | `JSONB` | Search queries actually issued to the web search adapter. |
| `web_sources_consulted` | `JSONB` | URLs whose content fed into the findings synthesis. |

Query research activity:
```sql
SELECT rb.question, jsonb_array_length(rb.findings) AS finding_count,
       rb.web_searches_run, rb.created_at
FROM research_briefs rb
ORDER BY rb.created_at DESC
LIMIT 20;
```

### `supervisor_events` (migration 004)

One row per non-pass Supervisor event. `pass` verdicts are not persisted —
they add noise without signal.

| Column | Type | Notes |
|--------|------|-------|
| `event_id` | `UUID PK` | |
| `turn_id` | `UUID FK` → `turns` | |
| `event_type` | `TEXT` | `fact_contradiction` \| `off_topic` \| `unsafe` |
| `severity` | `TEXT` | `low` \| `med` \| `high` |
| `details` | `JSONB NULL` | Free-form evidence dict (e.g. which fact was contradicted). |
| `action_taken` | `TEXT` | `pass` \| `warn` \| `retry` \| `block` |

Query Supervisor catch rate:
```sql
SELECT event_type, action_taken, COUNT(*) AS events
FROM supervisor_events
WHERE created_at > now() - interval '14 days'
GROUP BY event_type, action_taken
ORDER BY events DESC;
```
