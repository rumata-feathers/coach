# Career Coach — v2 Specification (Cognition / Production Split)

## Context

v0 shipped a walking skeleton. v0.5 added onboarding, session continuity, distillation, memory-loop closure. v1 added Researcher, Devil's Advocate, Synthesizer, Supervisor, Flow C deliberation, web search, charts, and the LangGraph migration.

v1 alpha shipped to a small audience. Real conversations surfaced four core failure modes:

1. **Committee voice.** Synthesizer responses read as transcripts ("the Coach suggests, the Devil's Advocate raises") instead of one integrated view.
2. **Premature advice.** The system jumped to recommendations before understanding what the user was actually wrestling with — surface keywords ("quant" → quant tradeoffs) drove generic advice patterns instead of user-specific reasoning.
3. **Vague action items.** Recommendations like "consider experimenting with side projects" or "explore informational interviews" without named programs, companies, courses, or people.
4. **Generic feel.** Conversations lacked the depth signal of "this thing actually understands me." Insufficient elicitation before production.

The root cause is structural, not stylistic: a single agent (Coach or Synthesizer) is asked to decide what the turn should accomplish *and* execute it. Under that pressure, fluency wins — "give advice" is the most fluent move, so the model pattern-matches there regardless of fit.

**v2 is the cognition / production split.** A typed `Directive` separates *what this turn should do* from *how to do it well*. The intelligence migrates upstream into a multi-agent cognition layer; production becomes focused execution against a clear contract.

> **Scope discipline.** v2 is a load-bearing rebuild. Do not start without (a) a week of real v1 alpha conversations to validate the failure modes are what we think they are, (b) a written list of which v1 components are kept versus removed, and (c) confirmed appetite for ~3 weeks of refactoring. The 9-commit v1 plan does not survive this; v2 has its own sequence.

Read `SPEC.md`, `SPEC_v0_5.md`, and `SPEC_v1.md` for foundation. v2 keeps every non-negotiable principle from v0 and most components from v1; what changes is the routing structure and the addition of cognition.

---

## 1. Non-Negotiable Principles

Preserved from v0/v0.5/v1:

1. Three-tier user memory (structured / episodic / semantic) — never collapsed.
2. Agents are modules with typed Pydantic contracts at every boundary.
3. The Critic can reject; uncertainty is a valid output.
4. Hypotheses are append-only with evidence.
5. Provider-agnostic LLM adapter.
6. Everything logged: every turn, every agent call, every token, every verdict.
7. Prompts in Jinja2 templates, never hardcoded.

v2 adds:

8. **Cognition produces no prose.** Cognition agents output typed objects only. The user-facing voice is exclusively a Production concern.
9. **Production picks no moves.** Production agents execute against a Directive. They do not infer what the turn should do; if the Directive is wrong, that's a Cognition bug or a Critic catch, not a Production override.
10. **Storage is unconditional.** Every turn persists every cognition output, every directive, every agent trace, full UserView snapshot. Cost optimization deferred until ≥10 active users (per HANDOFF_v2.md gating decision).

---

## 2. The Cognition / Production Split

```
                  ┌────────────────── COGNITION ──────────────────┐
                  │                                                │
                  │   Understander → SessionTheorist               │
   incoming turn ─┤                       ↓                        │
                  │              UserModeler → WorldModeler        │
                  │                       ↓                        │
                  │                   Navigator                    │
                  │                       ↓                        │
                  │              ┌── DIRECTIVE ──┐                 │
                  └──────────────┼───────────────┼─────────────────┘
                                 │               │
                  ┌──────────────┼───────────────┼─────────────────┐
                  │              ↓                                  │
                  │       Producer (one of six)                    │
                  │              ↓                                  │
                  │          Critic loop                           │
                  │              ↓                                  │
                  │         Supervisor                              │
                  │              ↓                                  │
                  │          response                  PRODUCTION   │
                  └─────────────────────────────────────────────────┘
                                 ↓ (after response, async)
                Profiler → UserModeler reactive cards → KB Curator
```

Cognition runs **sequentially** (Understander → SessionTheorist → UserModeler → WorldModeler → Navigator). Each cognition agent emits a status event over SSE while running, surfacing live progress to the user. Latency is dominated by Researcher and Coach calls anyway; sequential cognition adds ~3–5s of ordered latency that the SSE stream makes visible rather than dead time.

The interface is the Directive: strict, typed, replayable from logs.

---

## 3. The Directive

```python
class VoiceConstraints(BaseModel):
    target_length: Literal["short", "medium", "long"]
    breadth_first: bool                # forbid niche-narrowing
    name_gaps: bool                    # response must explicitly state what it doesn't know
    ban_advice_verbs: bool             # forbid "consider/explore/look into" without named entity
    must_carry_questions: int          # questions to append after main content (0–2)

class DirectiveScope(BaseModel):
    primary_question: str              # the actual ask, or the dimension being probed
    dimension_to_probe: Literal[
        "values_articulated",
        "constraints_surfaced",
        "current_path_belief",
        "alternatives_considered",
        "todays_driver",
    ] | None                           # for elicit moves
    hypothesis_to_test: UUID | None    # for re-examination
    landscape_required: bool           # set when answer requires landscape context

class Directive(BaseModel):
    mode: Literal["coach", "researcher"]   # session-level user preference
    move: Literal[
        "elicit", "landscape_map", "answer_grounded",
        "deliberate", "calibrated_partial", "checkpoint",
    ]
    scope: DirectiveScope
    user_context_slice: ContextSlice
    world_context_slice: WorldContextSlice
    voice_constraints: VoiceConstraints
    open_questions_to_carry: list[str]
    rationale: str                     # Navigator's reasoning, for dev panel + eval
    confidence: float                  # Navigator's confidence in this move
```

### 3.1 Mode

Two persistent interaction modes the user toggles in the UI, stored on `sessions.mode`:

- **Coach mode** (default) — warm, reflective, asks questions, shorter outputs. Navigator biases toward `elicit`, `answer_grounded`, `checkpoint`. "Just tell me" → `calibrated_partial`.
- **Researcher mode** — structured, sourced, charts where applicable, longer outputs. Navigator biases toward `landscape_map`, `deliberate`. Citations and charts are mandatory where data permits.

Mode is a Directive constraint, not a separate flow. Every move respects mode in voice; Navigator weights move selection by mode. The user can switch mid-session.

### 3.2 Moves and producers

| Move | Producer | When it fires |
|------|----------|---------------|
| `elicit` | Elicitor | Probe a missing dimension; one reflection + one question, no advice |
| `landscape_map` | Researcher (breadth) + Synthesizer (map renderer) | User needs to see the broader space before narrowing |
| `answer_grounded` | Coach | User has a specific question; landscape awareness present or not required |
| `deliberate` | Flow C | Deep decision; user has landscape awareness; multiple sources warranted |
| `calibrated_partial` | Coach (gap-naming mode) | User demands an answer; system isn't ready; produce defensible answer + named gaps + one carried question |
| `checkpoint` | Checkpointer | Confusion signal fired or 10-turn fallback |

### 3.3 Hard preconditions

Preconditions live in `Navigator.decide()` as Python checks, not in prompts:

- `deliberate` requires either an existing **valid** `landscape_map` for this user (see §3.4) **or** a SessionTheorist signal that the user has demonstrated landscape awareness in conversation. Otherwise Navigator must emit `landscape_map` first.
- `answer_grounded` does not require landscape (a salary-chart question doesn't need a landscape map first; this is the routing fix for the failure observed in v1 alpha).
- `checkpoint` requires confusion-signal-fired OR `turns_since_checkpoint >= 10`. Minimum gap of 5 turns between checkpoints.

This is the breadth-before-depth invariant made structural.

### 3.4 Landscape map staleness

A `landscape_map` is **valid** for the deliberate-precondition until any of the following invalidate it:

- `education_stage` changes
- `location` changes
- `financial_constraints` changes
- `time_horizon` changes
- A hypothesis crosses `confidence >= 0.8` for the first time
- A contradiction is resolved against an existing landscape category

Anything else, the map persists. Invalidation re-routes the next `decide` intent through `landscape_map` again.

---

## 4. Cognition Layer

### 4.1 Understander (mostly unchanged)

Reads the incoming user turn against recent conversation history. Outputs the existing `IntentPacket`. No changes from v1 except that `IntentPacket` is no longer the routing decision — Navigator subsumes that role.

### 4.2 Session Theorist

Promoted from v0.5's `session_theory` text field to a proper agent.

```python
class SessionTheory(BaseModel):
    what_user_came_in_for: str             # the surface ask
    what_seems_actually_at_stake: str      # the deeper concern
    confidence: float                      # how confident we are in the read
    convo_arc: Literal["opening", "exploring", "narrowing", "stuck", "wrapping"]
    user_demonstrated_landscape_awareness: bool  # gates `deliberate` precondition
    confusion_signals: list[str]           # signs we don't understand the user
    last_updated_turn: int
```

Reads recent_turns (last 10), current intent, prior session_theory. Updates the theory.

`confusion_signals` is the input to the checkpoint trigger. Examples:
- "User has corrected my read in the last 2 turns"
- "Recent intents don't form a coherent thread"
- "User-stated values appear to contradict observed choices"
- "I cannot articulate what this conversation is about"

If `len(confusion_signals) >= 2` or `confidence < 0.4`, the confusion flag fires.

**Model:** Coach-tier (genuine reasoning, not classification). Temperature 0.4.

### 4.3 User Modeler

Owns the user's structured facts, hypotheses, and a new **Completeness Vector**.

```python
class Coverage(BaseModel):
    state: Literal["not_probed", "partial", "sufficient", "contradictory"]
    last_evidence_turn: int | None
    notes: str

class CompletenessVector(BaseModel):
    values_articulated: Coverage      # what user says they care about
    constraints_surfaced: Coverage    # money, geography, family, time, health
    current_path_belief: Coverage     # what user thinks they're doing / about to do
    alternatives_considered: Coverage # paths user has weighed and dropped
    todays_driver: Coverage           # why this specific question now

class UserView(BaseModel):
    facts: dict[str, Any]
    hypotheses: list[Hypothesis]
    completeness: CompletenessVector
    open_questions_queue: list[OpenQuestion]    # ranked
    recent_corrections: list[Correction]        # user corrections of system reads
    contradictions: list[Contradiction]         # unresolved tensions
```

UserView is computed per turn and persisted as a full snapshot in `user_views` (per principle 10).

**Two maintenance modes (different SLAs, different triggers):**

**Reactive maintenance** runs synchronously when the Profiler (post-turn, async) emits a `ContradictionSignal`. The User Modeler:
1. Identifies which hypothesis or fact is contradicted.
2. Generates a `ReExaminationCard` describing the contradiction and the question to ask.
3. Pushes the card to the front of `open_questions_queue`.

The card is available to the *next* turn's Navigator. Interrupt-driven; SLA: ready before next turn arrives (typing time provides slack).

**Periodic maintenance** runs at session start and every K=10 turns (matching the checkpoint cadence). The User Modeler:
1. Audits the completeness vector for `not_probed` dimensions.
2. Audits hypotheses for low-evidence cases (≤1 evidence record, conf < 0.5).
3. Audits facts for staleness (not confirmed in last 20 turns).
4. Outputs `AuditFindings` with prioritized questions.

Audit findings populate `open_questions_queue` from the back. Sweep-driven; SLA: complete within ~30s.

**Model:** Coach-tier for reactive (judgment-heavy), Haiku-tier for periodic (mostly structural).

### 4.4 World Modeler

Owns the WorldView and KB tiered trust.

```python
class BreadthSignal(BaseModel):
    user_named_categories: set[str]    # categories user has actually mentioned
    user_compared_alternatives: bool   # has user weighed at least 2 alternatives?
    judged_aware: bool                 # composite signal

class WorldView(BaseModel):
    relevant_landscape_categories: list[str]  # canonical categories for this user's profile
    available_kb_coverage: dict[str, Tier]    # canonical / draft / none
    knowledge_gaps: list[str]                 # categories World Modeler thinks are relevant but unknown
    breadth_signal: BreadthSignal
    last_landscape_map_turn: int | None
    last_landscape_map_invalidated_at: datetime | None
```

The World Modeler is the structural enforcer of breadth-first. Given a user's facts and the running conversation, it produces a list of categories *relevant to this user* — not the universal career space, but the slice the user could realistically be considering. For a Polytechnique student, this might be 6–8 categories: quant trading, quant research, IB / DCM, pure software, AI/ML research, AI startups, deep tech, applied math academic. For a humanities undergrad it would be entirely different.

This list gates `deliberate` and feeds `landscape_map`. World Modeler also enforces the staleness rule from §3.4 — when a significant fact updates, it marks the latest landscape_map invalid and the next `decide` intent re-routes.

**KB tiered trust:**
- `kb/` (canonical) — hand-authored, every numeric field has reviewed `source_notes`. The current 15 entries.
- `kb_draft/` — Curator-generated, awaiting human review. Findings tagged `tier=draft`.

When the World Modeler queries KB and finds insufficient coverage for a relevant category, it flags it in `knowledge_gaps`. The KB Curator (background worker) picks up flagged gaps, drafts entries from research, and stages them in `kb_draft/`. Synthesizer surfaces a "draft, unreviewed" caveat when leaning on draft entries.

Promotion `kb_draft/` → `kb/` is manual.

**Model:** Coach-tier. Temperature 0.3.

### 4.5 Navigator

Small, deterministic-leaning, but informed by rich input. Takes IntentPacket + SessionTheory + UserView + WorldView + session mode and emits a Directive.

```python
class NavigatorInput(BaseModel):
    intent: IntentPacket
    session_theory: SessionTheory
    user_view: UserView
    world_view: WorldView
    turns_since_checkpoint: int
    session_mode: Literal["coach", "researcher"]

class Navigator:
    def decide(self, input: NavigatorInput) -> Directive: ...
```

Decision logic, in priority order:

1. **Checkpoint trigger:** if `session_theory.confusion_flag` or `turns_since_checkpoint >= 10` (and ≥5 turns since last checkpoint) → `checkpoint`.
2. **Reactive card:** if `user_view.open_questions_queue` has a reactive card → `elicit` against that card.
3. **Onboarding-equivalent:** if completeness has ≥2 `not_probed` core dimensions → `elicit` against the highest-priority dimension.
4. **Demand-but-not-ready:** intent is `decide` or `answer`, but `world_view.breadth_signal.judged_aware` is False AND the question requires breadth → `landscape_map` first, OR `calibrated_partial` if user has explicitly waved off elicitation.
5. **Deep decision, ready:** intent is `decide` AND landscape awareness present AND ≥5 facts → `deliberate`.
6. **Direct factual question:** intent is `quick` or simple-fact → `answer_grounded`.
7. **Default:** `answer_grounded` with breadth-first voice constraint.

Mode is applied as a *re-weighting* on top of these rules:

- **Researcher mode** boosts steps 4 and 5 — the user opted into deliverable output, so landscape and deliberation fire more readily. Voice constraints set `target_length="long"`, mandate citations on every claim, and emit charts where data permits.
- **Coach mode** boosts steps 1, 2, 3, and 6 — reflective, conversational. Voice constraints set `target_length="short"` to "medium", warmer tone, mandatory grounding without obligatory citations on every claim.

Steps 1–3 are hard rules (deterministic). Steps 4–7 use a small LLM call for tie-breaking and to generate the rationale string.

**User-demands-execution behavior** is critical. In Coach mode, "stop asking, just tell me" → `calibrated_partial`. In Researcher mode this is the default posture, so the system is already in delivery mode.

**Model:** Haiku-tier. Temperature 0.0 for hard-rule path; 0.3 for tie-breaking path.

---

## 5. Production Layer

Every producer reads its slice of context plus the Directive. None of them re-decide the move.

### 5.1 Elicitor

One reflection + one question. No advice. Length: 2–4 short paragraphs maximum.

The reflection must demonstrate the system *heard* the user's last message specifically — paraphrasing actual content, surfacing the implicit tension the user didn't name, or naming what's underneath the question. The question targets `directive.scope.dimension_to_probe`.

Critic enforces:
- Exactly one question mark in the response (`multiple_questions` failure mode).
- No advice verbs (`premature_advice`).
- Reflection references at least one specific thing the user said in their last message (`unanchored_reflection`).

**First-turn expectation-setting:** on `turn_index == 0`, the Elicitor prefixes with calibrated framing — *"Before I give you serious advice I want to actually understand you. That'll take a few turns. If you want quick answers right now I can do that too — just tell me."* Implemented in the template, gated by turn index.

**Blended elicitation:** the Elicitor may include a brief grounded acknowledgment or partial answer before pivoting to the question. Voice constraint: the answer portion is ≤2 sentences and explicitly framed as partial. This kills the "interrogation" failure mode.

**Model:** Coach-tier. Temperature 0.6 (warmest voice).

### 5.2 Landscape Mapper

A landscape category is *a cluster of specific roles that share entry-paths, day-to-day rhythm, and exit spaces enough that they make sense to consider together*. The point is mental scaffolding for the user, not enumeration.

For your transcript user (Polytechnique math/CS, IB/DCM offer in hand), the relevant categories are roughly: quantitative finance (trading + research + structuring collapsed because they share entry paths), investment banking (M&A + DCM + ECM + LevFin collapsed for the same reason), software engineering (product + infra + ML eng), AI/ML research (academic + industry labs), deep tech / hardware, pure academia (math/CS PhD), strategy consulting (MBB + tech boutiques), entrepreneurship. Eight. Each gets a fit rating against this *specific* user, not in the abstract.

The 4–10 bound: below 4, it's not a landscape — it's a comparison, which is what `deliberate` is for. Above 10, the user can't hold it in their head, *and* it's a structural tell that the World Modeler is showing the universal space rather than user-filtered, which is exactly the failure mode the move exists to prevent.

Two-step production:

1. **Researcher in breadth mode** — queries explicitly enumerate the space, not narrow into it. Retrieval plan optimizes for category coverage, not depth per category.
2. **Synthesizer in map renderer mode** — produces a `LandscapeMap` (structured output, plus a prose summary for the chat surface).

```python
class SubPath(BaseModel):
    name: str
    description: str
    fit_signals_for_user: list[str]

class LandscapeCategory(BaseModel):
    name: str
    summary: str                                        # 1–2 neutral sentences
    sub_paths: list[SubPath]
    fit_overall: Literal["strong", "possible", "weak", "unclear"]
    fit_reasoning: str                                  # tied to user_facts
    next_questions: list[str]                           # if user wants to explore this category
    citations: list[Citation]
    kb_tier: Literal["canonical", "draft", "none"]

class ExcludedCategory(BaseModel):
    name: str
    reason: str                                         # tied to user_facts

class LandscapeMap(BaseModel):
    user_id: UUID
    generated_at: datetime
    triggering_question: str
    categories: list[LandscapeCategory]                 # ≥4, ≤10
    excluded_categories: list[ExcludedCategory]         # NOT shown + reason
    caveats: list[str]
```

`excluded_categories` is structurally important: it shows the system considered breadth and explicitly ruled some out, making the breadth claim falsifiable. Example: `{name: "Medicine", reason: "User did not take pre-clinical pathway; switching now costs 6+ years"}`.

Critic on `landscape_map` enforces:
- `narrow_landscape`: fewer than 4 categories included → reject.
- `unfounded_exclusion`: excluded entries with empty reasons → reject.
- `niche_drift`: a category is itself a niche of another in the list (e.g., "Quant Trading" and "HFT Market Making" as siblings).
- `committee_voice`: prose summary mentions internal agents.

Frontend renders categories as cards with fit badges; v3 will add interactive drill-down. v2 ships the structured object plus the prose summary; data backs both.

**Model:** Coach-tier (Synthesizer). Researcher uses existing two-phase setup with breadth-mode prompt variant.

### 5.3 Coach (answer_grounded)

The existing Flow B Coach. Receives `user_context_slice` with full UserView (minus low-confidence hypotheses), the question, optional research findings.

New voice constraints from Directive:
- `ban_advice_verbs`: forbid "consider/explore/look into/experiment with" without a named entity in the same sentence (the `vague_action` Critic check, made structural).
- `name_gaps`: response must explicitly name what it doesn't know.

Otherwise unchanged from v1 Flow B.

### 5.4 Flow C deliberation (deliberate)

Existing Flow C: parallel Researcher + Coach → Devil's Advocate → Synthesizer → Critic.

Two changes in v2:

1. **Synthesizer prompt rewrite.** Eliminates committee voice. Coach/DA/Researcher outputs are passed as raw context labeled neutrally ("Considerations", "Counter-considerations", "Research findings"), with explicit anti-instructions and few-shot bad/good pairs. The Synthesizer is *one coach's voice*, not a transcript.
2. **`committee_voice` Critic check.** Cheap regex pre-check before any LLM call: strings like "the coach", "devil's advocate", "the synthesizer", "the researcher", paired with action verbs, fail immediately.

### 5.5 Calibrated Partial Coach (calibrated_partial)

When the user demands an answer the system isn't ready to give well, this is the producer.

Required structure:
1. **What I can defensibly say:** answer based on what we know, explicitly bounded.
2. **What I don't know:** named gaps in the user model that affect answer quality. This is `name_gaps` made structural — not an apology, a specific list.
3. **One question that would change my answer the most:** the highest-priority question from `open_questions_queue`.

This converts "I don't know enough about you yet" from a refusal into a useful response. The user gets a real answer, knows its limits, and sees what would sharpen it.

**Model:** Coach-tier.

### 5.6 Checkpointer (checkpoint)

Renders the user model in the user's voice.

```
Here's how I'm seeing you so far:
- I'm fairly sure: <high-conf facts and stable hypotheses>
- I'm guessing: <medium-conf hypotheses with their evidence>
- I'm confused about: <contradictions + gaps>
- What I think this conversation is about: <session_theory.what_seems_actually_at_stake>

Does this land, or am I getting something wrong?
```

The user's response to a checkpoint is a high-signal correction event. The Profiler weights checkpoint-response evidence at 1.5x normal weight. Hypotheses confirmed in a checkpoint reach `confirmed=true` faster.

Checkpoints persist in their own table for replay and analysis.

**Model:** Coach-tier. Temperature 0.4.

### 5.7 Two-stage misroute recovery

When Critic catches `move_mismatch` or another unrecoverable failure mode, the recovery is **not** a within-turn retry. It's a turn-spanning self-correction:

**Stage 1 (this turn):**
- Producer outputs the `calibrated_partial` response with whatever data is available.
- Response includes a transparent failure note: *"I tried to answer this as X, but the response didn't pass quality checks because Y. Here's what I can defensibly say, and I'm queueing Z to do better next turn."* — phrased warmly, not technically.

**Stage 2 (between turns, async):**
- A `RecoveryProbe` is queued: deeper UserModeler audit, Researcher sub-query, or KB Curator gap-fill, depending on the failure mode.
- Output lands in `open_questions_queue` (for elicitation) or `world_view.knowledge_gaps` (for research).

**Stage 3 (next turn):**
- Navigator picks up the recovery output naturally during decision logic. The next turn benefits without an explicit retry handshake.

This is more honest than silent retries and uses turn-to-turn context properly. It also gives the user agency — they can react to the failure note, redirect, or ask about what specifically went wrong.

---

## 6. Critic and Supervisor (extended)

Critic gains move-aware checks. Per move:

| Move | New failure modes |
|------|-------------------|
| `elicit` | `premature_advice`, `multiple_questions`, `unanchored_reflection` |
| `landscape_map` | `narrow_landscape`, `unfounded_exclusion`, `niche_drift` |
| `answer_grounded` | `vague_action`, `committee_voice` |
| `deliberate` | `committee_voice` + existing v1 set (`unintegrated`, `uncontested`, `chart_uncited`, `chart_data_invented`) |
| `calibrated_partial` | `gaps_unnamed`, `no_carried_question`, `vague_action` |
| `checkpoint` | `not_falsifiable` (no clear "did I get this right" prompt), `dump_format` (just listing facts without integration) |

`committee_voice` is implemented as a cheap regex pre-check before any LLM call: strings like `the coach`, `devil's advocate`, `the researcher`, `the synthesizer`, paired with action verbs, fail immediately.

Supervisor unchanged structurally — runs after Critic on every flow. v2 adds one new check: `move_mismatch` (response content doesn't fit the chosen move — e.g., elicit response contains advice). Failures route to the two-stage recovery flow (§5.7), not within-turn retry.

---

## 7. Memory and State

### 7.1 New tables (migration 005)

```sql
CREATE TABLE user_views (
    user_view_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    turn_id UUID REFERENCES turns(turn_id),
    completeness JSONB NOT NULL,
    confidence_summary JSONB,
    open_questions_queue JSONB,
    contradictions JSONB,
    facts_snapshot JSONB,                 -- full snapshot per principle 10
    hypotheses_snapshot JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE world_views (
    world_view_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    turn_id UUID REFERENCES turns(turn_id),
    relevant_categories JSONB,
    knowledge_gaps JSONB,
    breadth_signal JSONB,
    landscape_map_valid BOOL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE landscape_maps (
    map_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    triggering_turn_id UUID REFERENCES turns(turn_id),
    map JSONB NOT NULL,                   -- full LandscapeMap
    invalidated_at TIMESTAMPTZ,
    invalidation_reason TEXT,
    generated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE directives (
    directive_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID REFERENCES turns(turn_id),
    mode TEXT NOT NULL,
    move TEXT NOT NULL,
    scope JSONB,
    voice_constraints JSONB,
    rationale TEXT,
    confidence REAL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE re_examination_cards (
    card_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    triggered_by_turn UUID REFERENCES turns(turn_id),
    contradiction_summary TEXT,
    target_hypothesis_id UUID,
    target_fact_key TEXT,
    suggested_question TEXT NOT NULL,
    consumed_by_turn UUID REFERENCES turns(turn_id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE audit_findings (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    audit_run_id UUID NOT NULL,
    dimension TEXT,
    priority INT,
    suggested_question TEXT,
    consumed_by_turn UUID REFERENCES turns(turn_id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    turn_id UUID REFERENCES turns(turn_id),
    user_model_snapshot JSONB,
    user_response TEXT,
    corrections_extracted JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE recovery_probes (
    probe_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    triggered_by_turn UUID REFERENCES turns(turn_id),
    failure_mode TEXT,
    probe_kind TEXT,                       -- "user_modeler_audit" | "research" | "kb_gap_fill"
    payload JSONB,
    consumed_by_turn UUID REFERENCES turns(turn_id),
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 7.2 New columns on `turns`

```sql
ALTER TABLE turns ADD COLUMN directive_id UUID REFERENCES directives(directive_id);
ALTER TABLE turns ADD COLUMN move_used TEXT;
ALTER TABLE turns ADD COLUMN producer_used TEXT;
ALTER TABLE turns ADD COLUMN turns_since_checkpoint INT;
ALTER TABLE turns ADD COLUMN cognition_tokens_in INT DEFAULT 0;
ALTER TABLE turns ADD COLUMN cognition_tokens_out INT DEFAULT 0;
ALTER TABLE turns ADD COLUMN production_tokens_in INT DEFAULT 0;
ALTER TABLE turns ADD COLUMN production_tokens_out INT DEFAULT 0;
ALTER TABLE turns ADD COLUMN background_tokens_in INT DEFAULT 0;
ALTER TABLE turns ADD COLUMN background_tokens_out INT DEFAULT 0;
ALTER TABLE turns ADD COLUMN total_cost_usd_estimate REAL DEFAULT 0;
```

### 7.3 New column on `sessions`

```sql
ALTER TABLE sessions ADD COLUMN mode TEXT DEFAULT 'coach';
```

Migration sequence: v0 = 001–002, v0.5 = 003, v1 = 004, v2 = 005. Backward-compatible — all v1 columns and tables remain.

### 7.4 Storage policy

Per principle 10, snapshots are full, not deltas. UserView, WorldView, Directive, all agent traces persist on every turn. A 100-turn conversation may produce a few MB of JSONB; this is fine on Supabase free tier well past 100 users. Revisit when data volume becomes a real cost (≥10 active users with sustained engagement).

---

## 8. User-Context Slicing

A single helper governs what each agent sees of the user model:

```python
def context_slice_for(
    consumer: Consumer,
    user_view: UserView,
    intent: IntentPacket | None = None,
) -> ContextSlice: ...
```

Per-consumer relevance map:

| Consumer | Includes |
|----------|----------|
| Critic | facts (full), recent corrections, recent critic complaints; **no hypotheses** |
| Researcher_planner | facts: location, education_stage, financial_constraints, time_horizon |
| Coach | facts (full), hypotheses (conf ≥ 0.5), recent corrections |
| Synthesizer | facts (full), hypotheses (conf ≥ 0.5), recent corrections, contradictions |
| Devil's Advocate | facts (full), hypotheses (all, including low-conf), contradictions |
| Elicitor | completeness vector, open_questions_queue (top 3), recent corrections |
| Navigator | full UserView |
| Profiler | facts (existing only), no hypotheses |
| World Modeler | facts: education, profile basics, stated interests; no hypotheses |
| Session Theorist | facts (full), recent corrections, contradictions; no hypotheses |
| Checkpointer | full UserView |

Centralized policy. Lives in `src/career_coach/cognition/context_slicing.py`. One place to maintain.

---

## 9. Background Workers

### 9.1 Profiler (existing, extended)

After each turn, extracts facts and hypothesis evidence as before. **New:** detects contradictions against existing high-confidence hypotheses and emits a `ContradictionSignal` event consumed by reactive User Modeler.

### 9.2 User Modeler reactive

Triggered by `ContradictionSignal`. Generates a `ReExaminationCard` and writes to `re_examination_cards`. SLA: complete before next turn arrives (typically <5s; user typing time provides slack).

### 9.3 User Modeler periodic

Triggered at session start and every 10 turns. Audits completeness, hypothesis evidence depth, fact staleness. Writes `audit_findings`. SLA: ~30s.

### 9.4 KB Curator

Triggered by World Modeler `knowledge_gap` signals. Drafts a KB entry from research findings, with citations on every numeric field. Stages in `kb_draft/`. Logs a notification flagging entries pending review. SLA: nightly batch, plus on-demand for gaps blocking current turns.

### 9.5 Recovery Probe runner

Triggered by Supervisor `move_mismatch` or unrecoverable Critic failures. Runs the probe specified in `recovery_probes.probe_kind` and writes results back to the appropriate queue (`open_questions_queue` or `knowledge_gaps`). SLA: between turns, typically 5–30s depending on probe kind.

---

## 10. First-Session Policy

Not mode-locked. Navigator can blend within a single turn.

**Turn 1.** Elicitor prefixes with the expectation-setting line. Acknowledges the surface ask with 1–2 sentence partial reflection. Asks one targeted question against the highest-priority `not_probed` dimension. Default priority: `todays_driver` > `current_path_belief` > `values_articulated` > `constraints_surfaced` > `alternatives_considered`.

**Turns 2–4.** Navigator typically continues `elicit` until ≥3 dimensions reach `partial` or better, but blends in `calibrated_partial` if the user pushes for an answer ("just tell me X"). Each elicit reflects on the prior turn's specific answer, not generically.

**Turn 5+.** Once completeness reaches `partial` on ≥3 dimensions and `sufficient` on ≥1, `landscape_map` becomes available. After landscape_map, `deliberate` becomes available.

The user can fast-forward by saying "stop asking, just tell me" (Coach mode) or by toggling Researcher mode, in which case full delivery is the default posture.

---

## 11. Confusion Signal and Checkpoint Trigger

The checkpoint move fires when:

1. `SessionTheory.confidence < 0.4`, OR
2. `len(SessionTheory.confusion_signals) >= 2`, OR
3. `UserView.contradictions` has ≥1 unresolved contradiction older than 3 turns, OR
4. `turns_since_checkpoint >= 10` (fallback).

Conditions 1–3 are the "model is confused" path; condition 4 is the periodic safety net. Condition 4 alone (with no 1–3 firing) produces a lighter-weight checkpoint — more "here's what I think, am I tracking" than "I'm lost, help me catch up."

After a checkpoint, `turns_since_checkpoint` resets to 0. Hypotheses confirmed in the checkpoint response receive a 1.5x evidence weight bonus.

A minimum gap of 5 turns between checkpoints is enforced to prevent thrashing in unstable conversations.

---

## 12. Out of Scope for v2

Deferred to v3+:

- **Research memory / cache.** Caching findings with TTL and freshness checks. Useful but not blocking.
- **Initiator agent / proactive check-ins.** No system-initiated turns in v2; user still drives.
- **Connector / human practitioner network.** Same as v1.
- **God-agent / Evaluator.** Cross-session strategic oversight. The cognition layer is the foundation this would build on, but v2 stops at within-session cognition.
- **Interactive frontend rendering of LandscapeMap.** v2 emits the structured object and a prose summary; v3 renders categories as drillable cards.
- **Promotion-to-permanent logic for hypothesis lifecycle.**
- **EVoI scoring as a numerical metric.** Navigator uses heuristic priority ordering; v3 can quantify.
- **Cost optimization.** Cheap models everywhere for now; revisit when ≥10 active users.

---

## 13. Open Questions

Resolved during spec review (Q1–Q5 incorporated above):

- Q1 (cognition execution model) → sequential, with SSE progress events
- Q2 (misroute recovery) → two-stage, §5.7
- Q3 (storage policy) → full snapshots, principle 10
- Q4 (landscape staleness) → significant-fact-update list, §3.4
- Q5 (user fast-forward) → Coach/Researcher mode toggle

**Q6 (cost ceiling) remains deferred.** Cognition runs all four agents on every turn (~3 LLM calls minimum). Acceptable for v2 with cheap models. Revisit when ≥10 active users provide cost data.

---

## 14. User-facing Product Surface

Everything in this section is what the end user sees or interacts with. Distinct from §15 (dev/admin observability), which the user does not see.

### 14.1 Cognition trace stream (live)

Server-Sent Events from the backend during turn processing. Each cognition agent emits a status event when it starts:

- `understanding_message` — Understander running
- `updating_my_model_of_you` — UserModeler running
- `mapping_landscape` — WorldModeler running
- `choosing_approach` — Navigator running
- `researching_X` — Researcher in progress (with the actual query)
- `composing_response` — Producer running

Frontend renders as a thin "thinking" indicator with the current phase visible. Sub-second transitions for cognition agents; longer for Researcher/Coach calls. The point: 8–25s waits become legible work, not dead time.

### 14.2 Chart rendering

v1 emits `ChartSpec`; v2 renders. Recharts integration in the React frontend. Charts inline below prose. Researcher mode emits charts liberally; Coach mode only when comparison/trend data warrants.

### 14.3 Clickable citations

Two fixes:

1. Frontend renders `[N]` numerals in response text as clickable anchors against `citations[]`.
2. KB citations get clickability via a new `/kb/career/{name}` endpoint that renders YAML entries as readable HTML; KB citations resolve to those URLs in the response payload. No more dead `kb_path` strings.

Both web and KB citations open in a new tab; URL is shown on hover.

### 14.4 Coach / Researcher mode toggle

UI toggle in the chat header (or settings). Stored as `sessions.mode`. Default `coach`. User can switch mid-session; the next turn's Directive picks up the new mode. UI adjusts: in Researcher mode, citations are visually emphasized, charts are larger, layout is denser.

A small inline tooltip explains the modes when first encountered.

---

## 15. Dev / Admin Observability Surface

**Not user-facing.** Auth-gated route, accessible only to project maintainers. Lives at `/admin/*` or behind a feature flag. The end-user UI never references this.

### 15.1 Per-turn dev panel

For any turn, an expandable view showing:

- The full Directive (mode, move, scope, voice_constraints, rationale, confidence)
- The UserView snapshot at the time of the turn
- The WorldView snapshot
- The cognition trace (every agent's input/output, latency, tokens)
- The producer output
- All Critic verdicts (including retries)
- Supervisor result
- Citations resolved (with clickable URLs)
- Token breakdown by phase (cognition / production / background)

This makes every conversation self-documenting. Essential for debugging cognition decisions, identifying patterns in misroutes, and validating fixes after they ship.

### 15.2 Cross-turn token rollup

Aggregate views:
- Per-session cost
- Per-user cost (across sessions)
- Cost split: cognition / production / background
- Top-N most expensive turns, ranked
- Cost-per-turn histogram

Implementation: SQL views over the new `turns` token columns. No new tables.

### 15.3 Failure mode dashboard

For each Critic failure mode and Supervisor flag, count over time. Per move. Lets you see whether `committee_voice` is actually getting fixed by the Synthesizer rewrite, whether `vague_action` is decreasing, etc. Same SQL-view approach.

### 15.4 Recovery probe inspector

List of recovery probes with status (queued / running / consumed). Lets you see how often two-stage recovery fires and whether downstream turns benefit.

---

## 16. Summary: What Changes from v1

**Added agents:** Session Theorist (promoted from field), User Modeler, World Modeler, Navigator, Elicitor, Checkpointer, KB Curator (background), Recovery Probe runner (background).

**Modified agents:** Synthesizer (prompt rewrite for committee-voice, plus map renderer mode), Coach (voice constraints from Directive, plus calibrated_partial mode), Critic (move-aware failure modes), Supervisor (move_mismatch check), Profiler (ContradictionSignal emission).

**Removed/reduced:** Orchestrator's role as the single routing decision point — subsumed by Navigator. Orchestrator class can stay as a thin compatibility shim or be deprecated outright.

**New data:** user_views, world_views, landscape_maps, directives, re_examination_cards, audit_findings, checkpoints, recovery_probes; new columns on turns and sessions.

**New invariants:** Cognition produces no prose. Production picks no moves. Storage is unconditional. Breadth before depth, structurally enforced.

**New surfaces:** Cognition trace SSE stream, chart rendering, clickable citations, Coach/Researcher mode toggle (user-facing). Per-turn dev panel, token rollup, failure dashboard, recovery probe inspector (auth-gated, dev-only).

The result: "the model didn't understand me" becomes a Cognition bug with a clear locus, not a vibes problem in a 4000-token prompt.
