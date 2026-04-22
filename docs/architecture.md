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

| Flow | When | Critic | Latency |
|------|------|--------|---------|
| A | `budget_hint=quick` OR `turn_intent=vent` | No | Fast |
| B | Everything else | Yes (≤2 retries) | Slower, higher quality |

---

## Quality gate (Critic)

The Critic checks every Coach response (flow B) against four failure modes:

| Mode | Meaning |
|------|---------|
| `generic` | Could apply to any user — no personalisation |
| `ungrounded` | Claims facts not in the user model |
| `false_confidence` | Specific numbers/claims without hedging |
| `off_intent` | Doesn't address what the user actually asked |

On `reject`, the Critic provides a `suggested_fix` that Coach uses on the next
attempt. After 2 rejections, the pipeline escalates to an honest "I need more
context" response.

---

## Database schema

See [`migrations/001_initial.sql`](../migrations/001_initial.sql) for the
authoritative schema. Key indexes:

- `turn_embeddings` has an `ivfflat` cosine index for similarity search
- `structured_facts` has a `UNIQUE (user_id, key)` constraint for upsert
- `hypothesis_evidence` uses `hypothesis_id IS NULL` as the distillation queue
