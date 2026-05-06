# Career Coach — v0.5 Specification (Consolidation)

## Context

v0 shipped and works. Real conversations with the deployed system produced:
- 5 real structured facts extracted by the Profiler from a single session
- 5 evidence rows with meaningful hypothesis hints
- Coach responses that demonstrably grounded in extracted facts (turns 3-5 of the test conversation)

But the same deployment surfaced **four classes of defects** that must be fixed before adding any new agents in v1. v0.5 is pure consolidation — no new agents, no new product surface.

Read `SPEC.md` (v0) for the architectural contract. This document describes **additions and changes** for v0.5.

---

## 1. Non-Negotiable Additions

1. **Reliability is a first-class concern.** Empty LLM responses, JSON parse failures, and schema violations must not silently degrade the turn. The system must either retry or surface the failure — never fall through a fallback path without logging a structured warning that the evaluator can count.
2. **Cold-start is a distinct code path.** New users (fewer than 5 facts OR fewer than 3 turns in their history) route to an onboarding flow, not the standard Coach pipeline.
3. **Observability must be measurable.** Latency, retry counts, and fallback-path invocations all land in `agent_calls` as real numbers, not as zero or null.
4. **Distillation quality is measured, not assumed.** Same canary discipline as the Critic — ten hand-crafted multi-turn fixtures, graded pass/fail, part of CI.
5. **The memory loop is verified.** An end-to-end integration test starts a user, walks them through N turns across M sessions, runs distillation, and asserts the Coach uses a hypothesis on a later session.

---

## 2. What v0.5 does NOT do

Deferred to v1 (do not touch):
- Researcher, Devil's Advocate, Synthesizer agents
- God-agent
- Supervisor / Evaluator
- World Knowledge Base
- Frontend
- Auth
- LangGraph migration
- n8n job scheduling

---

## 3. Pillar 0 — Reliability fixes

### 3.1 Latency measurement (bug)

`Agent.now_ms` uses `int(time.monotonic() * 1000)`. Two calls within the same millisecond produce identical integers and subtraction gives `0`. Every row in `agent_calls.latency_ms` is currently 0.

**Fix:** Change to `time.perf_counter_ns() // 1_000_000` or equivalent. Write a unit test that asserts `log_call` receives non-zero latency for a deliberately slow mock LLM (insert an `asyncio.sleep(0.01)`).

### 3.2 Empty-response handling (high priority)

JSON parse errors like `Expecting value: line 1 column 1 (char 0)` appear across Profiler, Understander, Coach, and Critic in production `agent_calls` rows. Cause: Qwen3 sometimes returns only a `<think>...</think>` block, which `HuggingFaceClient` strips — leaving an empty string that `json.loads` rejects.

**Fix, in two parts:**

**Part A — Suppress thinking blocks for JSON agents.** `config/models.yaml` supports `extra_body`; use it:

```yaml
coach:
  provider: huggingface
  model: Qwen/Qwen3-235B-A22B
  temperature: 0.7
  max_tokens: 2500
  extra_body: { "thinking": false }
```

Do this for every agent that returns JSON: understander, coach, critic, profiler, and the distillation prompt.

**Part B — Retry on empty/invalid LLM output.** In each agent's `run()`, if parsing fails:
1. Log the failure explicitly to `agent_calls` with `error` populated (already happens).
2. **Retry once** with a stricter prompt: *"Your previous response was not valid JSON. Return ONLY the JSON object matching this schema: <schema>. No prose, no think-blocks."*
3. If the second attempt also fails, THEN use the existing fallback path.

Track retries: add a `retry_count` column to `agent_calls` (migration 003), default 0, incremented on the retry call. This makes "how often are we retrying?" a query, not a guess.

### 3.3 Structured fallback tracking

Every fallback path (understander `_fallback_packet`, coach `_error_output`, critic fail-open-with-pass, profiler empty-output) must log a tagged row. Add a `fallback_reason` text column to `agent_calls` (same migration 003). Populate with values like `empty_llm_response`, `schema_validation_failed`, `retry_exhausted`. Null means no fallback happened.

Result: a single SQL query can count fallbacks per agent per day — the actual quality canary.

### 3.4 Background-task lifecycle

In `pipeline/turn.py`:

```python
_task = asyncio.create_task(  # noqa: RUF006 — intentionally fire-and-forget
    self._profiler.run_and_save(...),
)
```

The `_task` reference is local and will be garbage-collected when `process_turn` returns. Python may — and under load, will — kill the task before it completes.

**Fix:** Keep a reference on the pipeline instance: `self._background_tasks: set[asyncio.Task] = set()`. On create: `self._background_tasks.add(task); task.add_done_callback(self._background_tasks.discard)`. Remove the `noqa`.

Integration test: fire 50 concurrent `process_turn` calls against a fixture pipeline with a slow Profiler mock (300 ms). Assert that `agent_calls` has 50 profiler rows after all complete.

---

## 4. Pillar 1 — Cold-Start / Onboarding

### 4.1 Detection

A user is "new" for routing purposes if EITHER:
- `count(structured_facts WHERE user_id = $1) < 5`, OR
- `count(turns WHERE user_id = $1) < 3`.

Implement as `async def is_new_user(user_id) -> bool` on a new `OnboardingPolicy` class (not the Orchestrator — keep the Orchestrator's logic LLM-free and stateless).

### 4.2 New flow type

The `Orchestrator.decide` signature changes:

```python
Flow = Literal["A", "B", "onboarding"]

def decide(self, intent: IntentPacket, is_new_user: bool) -> Flow:
    if is_new_user:
        return "onboarding"
    if intent.turn_intent == "vent" or intent.budget_hint == "quick":
        return "A"
    return "B"
```

The turn pipeline calls `OnboardingPolicy.is_new_user(user_id)` before invoking the Orchestrator, and passes the result in.

### 4.3 Onboarding flow mechanics

New module: `career_coach/pipeline/onboarding.py`.

**Exposed API:**

```python
class OnboardingPipeline:
    def __init__(self, factory: LLMFactory) -> None: ...

    async def process_turn(
        self,
        user_id: UUID,
        session_id: UUID,
        user_message: str,
        intent_packet: IntentPacket,
        user_facts: dict,
        recent_turns: list[TurnSummary],
    ) -> CoachOutput: ...
```

**Behavior:** Uses a dedicated prompt template `config/prompts/onboarder.j2` that instructs the Coach-tier model (Qwen3-235B) to:

1. Acknowledge the user's question warmly.
2. Explain briefly that to help well, it needs a little context.
3. Ask ONE targeted question tailored to what the user just asked.
4. Output as `CoachOutput` — same schema as Coach, so downstream persistence works unchanged.

**The dedicated question bank.** Onboarding uses three fixed question archetypes, cycled:

- **Turn 1** — Context probe: *"Before I can answer well, tell me about where you're at — what are you studying / working on, and what prompted this question today?"*
- **Turn 2** — Values probe (forced choice): *"Here's a scenario: two careers, same pay. One has predictable hours and a clear ladder; the other has chaotic hours, novel problems, and an unclear path. Which pulls you more, and why?"*
- **Turn 3** — Commitment probe: *"Based on what you've shared, here's how I'd start thinking about this <brief grounded thought>. Does that framing feel right, or does it miss something important?"*

The prompt template selects the probe based on the user's turn index within their first 3. After 3 onboarding turns OR after the user's fact count reaches 8+, the user graduates from onboarding on the NEXT turn.

Onboarding responses still flow through Profiler (no change).

### 4.4 Critic behavior during onboarding

Flow "onboarding" bypasses the Critic. Rationale: the onboarding response is a structured question, not a grounded coaching response — applying the "must reference ≥2 user facts" rule guarantees rejection on turn 1 by design.

Record this in `turns.flow_used = "onboarding"` so it shows up in logs.

### 4.5 Quality canary — onboarding eval

Five scripted cold-start fixtures, each a fresh user asking a different opening question ("Should I study economics?", "I'm thinking about switching majors", "Quant finance, thoughts?", "I don't know what I want to do", "What careers pay well in London?"). Run the full onboarding pipeline 3 turns deep for each. Assertions per fixture:

- Turn 1 response acknowledges the question (keyword present) AND asks a probe question (ends in `?` OR contains a question phrase).
- Turn 2 response includes a forced-choice or values-oriented probe.
- Turn 3 response references at least 2 facts that the Profiler extracted from turns 1-2.
- Over the 3 turns, at least 4 facts end up in `structured_facts`.

Report format same as `reports/quality_baseline.md`. Call it `reports/onboarding_baseline.md`.

---

## 5. Pillar 2 — Session continuity

### 5.1 Observation

Every turn in the v0 test conversation was a **new session** (the user didn't pass `session_id` back). This means `recent_turns` was always empty and the within-session memory path is untested.

### 5.2 No code changes required for the fix itself

The architecture already supports session continuity. What's missing is:

1. An integration test that **exercises** it.
2. Documentation in `README.md` showing the correct way to continue a session.
3. A "demo script" (`scripts/demo_conversation.py`) that runs a scripted 5-turn conversation against a live backend, preserving `session_id`, and dumps the final state.

### 5.3 Integration test

Add `tests/integration/test_session_continuity.py`:

- Seed a user.
- Run 4 sequential `process_turn` calls with the same `session_id`.
- Assert `turns.turn_index` goes 0, 1, 2, 3.
- Assert the 4th turn's `recent_turns` parameter into the Understander contained the first 3 messages. (Expose this via test-visible logging or a pipeline hook — don't introspect the LLM prompt.)
- Assert `sessions.session_theory` is non-null and has been updated at least once since turn 1.

### 5.4 Demo script

`scripts/demo_conversation.py`:

```
$ uv run python scripts/demo_conversation.py

[Creating user "Demo"…]
user_id: abc-123
session_id: def-456

[Turn 1] > "I'm 19, studying CS in London and considering a PhD."
[flow=onboarding]
Response: <…>

[Turn 2] > "I've done two internships — one at a bank, one at a research lab."
[flow=onboarding]
Response: <…>

…

[After 5 turns, running distillation…]
Created 2 hypotheses, matched 3 evidence rows to existing.

[Final user model:]
Facts (7): age, location, education_stage, major, internship_history_banking, internship_history_research, considering_phd
Hypotheses (2):
  [0.42] "User may prefer research environments over commercial ones."
  [0.35] "User values intellectual depth over career velocity."

[Done.]
```

Non-negotiable: this script is how you and future-you prove the system still works. Make it satisfying.

---

## 6. Pillar 3 — Distillation quality

### 6.1 Problem

The distillation code works (integration test passes) but is **semantically untested**. The integration test uses a mock LLM that returns a scripted response; we've never verified that distillation produces sensible hypotheses from a real Qwen3 call on real evidence.

### 6.2 Ten multi-turn conversation fixtures

Create `tests/quality/distillation_fixtures/`, each fixture a short YAML file:

```yaml
# fixtures/01_analytical_student.yaml
name: analytical_student
persona: "19yo undergrad in London, math major, gravitates to theory over applied work"
seed_facts:
  age: 19
  location: London
  education_stage: undergrad
  major: Mathematics
turns:
  - role: user
    text: "I find proofs much more interesting than programming projects."
  - role: assistant
    text: "<short coach response>"
  - role: user
    text: "When I had to do a group software project I hated every moment."
  # …5-8 turns total
expected_hypotheses:
  - theme: "analytical / abstract preference"
    min_confidence: 0.4
  - theme: "prefers solo / deep work over collaboration"
    min_confidence: 0.3
forbidden_hypotheses:
  - "user is sociable and enjoys teamwork"
```

Ten fixtures across persona types: analytical, social, indecisive, pivoter-curious, prestige-motivated, impact-motivated, stability-seeking, chaos-comfortable, burned-out, over-committed.

### 6.3 LLM-as-judge evaluation

`scripts/run_distillation_eval.py`:

1. For each fixture: create user, seed facts, run each turn through `process_turn` against a real LLM, then run distillation.
2. Read back the produced hypotheses.
3. For each expected theme, ask a judge LLM: *"Does this set of hypotheses include one that expresses the theme <X>? Yes/no and why."* Use `coach` model tier for the judge — Qwen3-235B.
4. For each forbidden theme, same check — a match is a failure.
5. Score: matched expected - matched forbidden. Pass if ≥ 0.7 average across fixtures.

Write `reports/distillation_baseline.md` with per-fixture pass/fail. Commit it.

### 6.4 Cross-run deduplication

Currently `distillation.run_distillation` deduplicates new hypotheses within a single run (the `new_hyp_cache` dict) but not across runs. After 5 distillation passes on the same user, you can get 5 near-duplicate hypotheses.

**Fix:** before creating a new hypothesis, fetch all active hypotheses and do a cheap LLM similarity check: *"Is this new proposed hypothesis substantively the same as any of these existing ones? If yes, return the existing ID. If no, return 'new'."* Use the profiler-tier model. One extra LLM call per new hypothesis proposal — acceptable.

Measure the effect: re-run the distillation eval after the dedupe change, assert average hypothesis count per user stays ≤ 4 after 3 distillation passes.

---

## 7. Pillar 4 — Memory loop closure

### 7.1 The verification

End-to-end integration test `tests/integration/test_memory_loop.py`:

1. Create user, seed minimal facts (age, location, stage).
2. Run a 6-turn conversation in session A about "considering CS PhD vs industry", through onboarding and normal flow.
3. Run distillation. Assert ≥ 1 hypothesis created with confidence ≥ 0.3.
4. Open a NEW session B for the same user.
5. Run 1 new turn asking: "what should I be considering about this decision?"
6. Fetch the resulting turn's `intent_packet`, coach output, and `agent_calls` payload.
7. **Assert** that either:
   - `coach_output.referenced_hypotheses` contains at least one UUID from the hypotheses created in step 3, OR
   - the Coach prompt (from `agent_calls.input_payload`) demonstrably included the hypothesis statements.

### 7.2 Coach prompt tweak to force the loop closure

In `config/prompts/coach.j2`, under GROUNDING, add:

> If any `active_hypotheses` have confidence > 0.5, AT LEAST ONE of your `referenced_hypotheses` must be a hypothesis UUID (not just a fact key). If the active hypotheses don't apply to this turn, explicitly note that in `uncertainty_flags` as `"active_hypotheses_did_not_apply"`.

Re-run the Critic battery after this change. If Critic pass rate drops below 8/10, revert and try a softer rule.

---

## 8. Schema changes (migration 003)

```sql
-- 003_observability.sql
ALTER TABLE agent_calls
    ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS fallback_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_agent_calls_fallback
    ON agent_calls(agent_name, fallback_reason)
    WHERE fallback_reason IS NOT NULL;
```

No data loss, no renames, existing rows get `retry_count=0` and `fallback_reason=NULL`.

---

## 9. Definition of Done

v0.5 is done when ALL of these pass:

1. The existing `critic_battery` still passes ≥ 8/10 (no regression).
2. The new `onboarding_baseline` passes all 5 fixtures.
3. The new `distillation_baseline` averages ≥ 0.7 across 10 fixtures.
4. The new `memory_loop` integration test passes.
5. A full week of demo-script usage (you, daily, 5 minutes) produces:
   - `count(agent_calls WHERE fallback_reason IS NOT NULL) < 5% of count(agent_calls)`
   - `avg(latency_ms) > 200` for coach-tier calls (i.e. we're actually measuring latency)
6. The single-user quality eval (`run_quality_eval.py`) still scores ≥ 19/20.

When all 6 hit: cut a v0.5 tag, then plan v1.
