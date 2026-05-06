# TASKS — Career Coach v2

15 commits across 7 phases. Each task = one commit. Read `SPEC_v2.md` first; this document is the build sequence, not the spec.

**Hard gate** between Phase 2 and Phase 3 (see HANDOFF_v2.md §7). Do not start producers until cognition produces reasonable Directives on logged v1 transcripts.

---

## Phase 1 — Foundation

### Task 1 — Migration 005 + Pydantic types + context slicing helper

**Goal:** Establish the data model and typed contracts. No agents yet, no logic, no LLM calls. Just the foundation that everything else builds on.

- [ ] Create `migrations/005_v2_cognition_split.sql` with:
  - All 8 new tables from SPEC_v2.md §7.1: `user_views`, `world_views`, `landscape_maps`, `directives`, `re_examination_cards`, `audit_findings`, `checkpoints`, `recovery_probes`.
  - All new columns on `turns` from §7.2 (directive_id, move_used, producer_used, turns_since_checkpoint, all six token columns, total_cost_usd_estimate).
  - New column on `sessions` from §7.3 (`mode` defaulting to `'coach'`).
  - Indexes on user_id, turn_id, and consumed_by_turn where appropriate.
- [ ] Add Pydantic models in `src/career_coach/models/cognition.py`:
  - `Coverage`, `CompletenessVector`, `UserView`
  - `BreadthSignal`, `WorldView`
  - `SessionTheory`
  - `OpenQuestion`, `Correction`, `Contradiction`
  - `ReExaminationCard`, `AuditFinding`, `RecoveryProbe`
- [ ] Add Pydantic models in `src/career_coach/models/directive.py`:
  - `VoiceConstraints`, `DirectiveScope`, `Directive`
  - `ContextSlice`, `WorldContextSlice`
- [ ] Add `LandscapeMap`, `LandscapeCategory`, `SubPath`, `ExcludedCategory` to `models/agent_io.py`.
- [ ] Create `src/career_coach/cognition/context_slicing.py`:
  - `Consumer` enum with all 11 values from §8.
  - `context_slice_for(consumer, user_view, intent=None) -> ContextSlice` implementing the relevance map exactly.
  - Unit tests for each consumer slice: assert correct fields included, asserted fields excluded.
- [ ] Add repos in `src/career_coach/db/`:
  - `UserViewRepo`, `WorldViewRepo`, `LandscapeMapRepo`, `DirectiveRepo`, `ReExaminationCardRepo`, `AuditFindingRepo`, `CheckpointRepo`, `RecoveryProbeRepo`.
  - Each with `insert`, `get_latest_for_user`, `list_for_user` minimum.
- [ ] Unit tests:
  - Round-trip serialize/deserialize every new Pydantic model.
  - Round-trip insert/select for every new repo.
  - `test_context_slice_critic_excludes_hypotheses`
  - `test_context_slice_researcher_planner_minimal_facts_only`
  - `test_context_slice_navigator_full_userview`

**Done when:** Migration runs cleanly on a fresh dev DB. All Pydantic models import without error. Every repo round-trips. Every consumer's context slice matches the map in §8. No agent logic exists yet.

**Commit message:** `feat(v2): foundation — migration 005, cognition types, directive types, context slicing`

---

## Phase 2 — Cognition

### Task 2 — Session Theorist agent

**Goal:** Promote `sessions.session_theory` from a text field to a typed agent that runs every turn.

- [ ] Add `SessionTheoristInput` and `SessionTheory` (already in cognition.py from Task 1) to `models/agent_io.py` if needed for re-export.
- [ ] Add `session_theorist` block to `config/models.yaml` (Coach-tier, temperature 0.4).
- [ ] Create `config/prompts/session_theorist.j2` per §4.2 principles. Must instruct the model to populate `confusion_signals` with specific examples; refuse vague signals.
- [ ] Create `src/career_coach/agents/session_theorist.py` with retry-on-parse-failure pattern.
- [ ] Persist output: extend `sessions` table or use a new `session_theories` table for per-turn snapshots — choose one and document.
- [ ] Unit tests with `MockLLMClient`:
  - `test_session_theorist_extracts_arc` — verify `convo_arc` parsing.
  - `test_session_theorist_flags_confusion_when_corrections_present` — input with recent corrections, verify confusion_signals populated.
  - `test_session_theorist_low_confidence_triggers_flag` — verify the `confidence < 0.4` path.
  - `test_session_theorist_landscape_awareness_detection` — user mentioned ≥2 alternative categories, verify `user_demonstrated_landscape_awareness=True`.
- [ ] Integration test: 5-turn conversation, verify `session_theory.confidence` evolves and `confusion_signals` flags appropriately on a synthetically confusing turn.

**Done when:** SessionTheorist runs on a real turn input and produces a valid `SessionTheory`. Confusion signal fires correctly on a contradiction fixture.

**Commit message:** `feat(v2): SessionTheorist agent — typed session theory with confusion signals`

---

### Task 3 — KB tier infrastructure + WorldModeler agent

**Goal:** Add `kb_draft/` directory tier and the WorldModeler agent that owns landscape category enumeration.

- [ ] Create `kb_draft/` directory at project root (next to `kb/`). Add `.gitkeep`.
- [ ] Extend `WorldKBRepo` in `src/career_coach/kb/repo.py`:
  - Load both `kb/` (canonical) and `kb_draft/` (draft) at construction.
  - Tag each loaded entry with `tier: Literal["canonical", "draft"]`.
  - Update query methods to optionally filter by tier or return tier in results.
- [ ] Add `world_modeler` block to `config/models.yaml` (Coach-tier, temperature 0.3).
- [ ] Create `config/prompts/world_modeler.j2`:
  - Takes user_facts + recent_turns + KB index summary.
  - Outputs `WorldView` with `relevant_landscape_categories` (4–10), `knowledge_gaps`, `breadth_signal`.
  - Strict instruction: only include categories the user could realistically be considering given their facts. No universal career space.
- [ ] Create `src/career_coach/agents/world_modeler.py`.
- [ ] Implement landscape map staleness invalidation per §3.4:
  - Add `_check_significant_fact_update(old_facts, new_facts) -> bool` helper.
  - When called from the turn pipeline (after fact updates land), update `world_views.landscape_map_valid` to FALSE for the relevant landscape map.
- [ ] Unit tests:
  - `test_world_modeler_polytechnique_user_returns_quant_ib_software_etc` — fixture user with math/CS at Polytechnique, assert categories include quant_finance, investment_banking, software_engineering.
  - `test_world_modeler_humanities_user_excludes_quant` — fixture user with literature degree, assert quant_finance NOT in categories.
  - `test_world_modeler_kb_coverage_marks_draft_tier` — KB has only 1 canonical career, 1 draft career, verify `available_kb_coverage` distinguishes.
  - `test_world_modeler_knowledge_gaps_flagged` — relevant category with no KB coverage, verify it's in `knowledge_gaps`.
  - `test_landscape_map_invalidated_on_education_stage_change`.
  - `test_landscape_map_invalidated_on_high_conf_hypothesis`.
- [ ] Integration test: WorldModeler against the real 15 KB careers + a fixture user, verify a sensible 6–8 category list comes back.

**Done when:** WorldModeler returns user-filtered category lists, KB tier infrastructure loads both directories, staleness invalidation fires on the right triggers.

**Commit message:** `feat(v2): WorldModeler agent + kb_draft tier infrastructure`

---

### Task 4 — UserModeler agent (reactive + periodic) + Profiler ContradictionSignal

**Goal:** Maintain the user model with two distinct maintenance modes.

- [ ] Extend `Profiler`:
  - After existing extraction, run a contradiction check against high-confidence existing hypotheses.
  - Emit a `ContradictionSignal` event (just a typed log entry + a row in a new `contradiction_signals` table or as an `audit_findings` entry — choose).
  - Unit test: fixture turn that contradicts an existing hypothesis fires the signal.
- [ ] Add `user_modeler_reactive` and `user_modeler_periodic` blocks to `config/models.yaml`. Reactive = Coach-tier, Periodic = Haiku-tier.
- [ ] Create `config/prompts/user_modeler_reactive.j2`:
  - Input: contradiction signal + user_view.
  - Output: `ReExaminationCard` with `suggested_question` targeting the contradiction.
- [ ] Create `config/prompts/user_modeler_periodic.j2`:
  - Input: full UserView.
  - Output: list of `AuditFinding` with priorities.
- [ ] Create `src/career_coach/agents/user_modeler.py` with two methods: `run_reactive(signal, user_view)` and `run_periodic(user_view)`.
- [ ] Implement `CompletenessVector` computation:
  - Function in `src/career_coach/cognition/completeness.py` that takes structured_facts and hypothesis_evidence and produces a `CompletenessVector`.
  - This is rule-based, not an LLM call — defines what "values_articulated=sufficient" means in terms of fact keys present.
  - Document the rules clearly in docstrings.
- [ ] Wire reactive maintenance:
  - Triggered by ContradictionSignal from Profiler post-turn-async.
  - Writes ReExaminationCard to `re_examination_cards`.
- [ ] Wire periodic maintenance:
  - Triggered at session start AND every 10 turns.
  - Writes AuditFindings to `audit_findings`.
- [ ] Unit tests:
  - `test_user_modeler_reactive_produces_card_for_contradiction`
  - `test_user_modeler_periodic_flags_not_probed_dimensions`
  - `test_user_modeler_periodic_flags_low_evidence_hypothesis`
  - `test_completeness_vector_values_articulated_partial_after_one_value_stated`
  - `test_completeness_vector_todays_driver_sufficient_after_explicit_question_about_change`
- [ ] Integration test: a 12-turn conversation that triggers periodic maintenance twice (turn 0, turn 10) and one reactive trigger from a contradiction in turn 7. Verify queue contents at each point.

**Done when:** Both maintenance modes run cleanly. Cards and findings land in the right tables. Completeness vector advances as facts accumulate. Profiler emits signals on contradictions.

**Commit message:** `feat(v2): UserModeler reactive + periodic maintenance, completeness vector, ContradictionSignal`

---

### Task 5 — Navigator + cognition pipeline integration

**Goal:** Wire all cognition agents into the turn pipeline. Output: a Directive on every turn.

- [ ] Add `navigator` block to `config/models.yaml` (Haiku-tier, temperature 0.0 for hard rules / 0.3 for tie-break).
- [ ] Create `config/prompts/navigator_tiebreak.j2`:
  - Used only when steps 4–7 of decision logic are reached.
  - Input: NavigatorInput.
  - Output: `{move, rationale, confidence}`.
- [ ] Create `src/career_coach/agents/navigator.py`:
  - `decide(input: NavigatorInput) -> Directive` per §4.5 logic.
  - Steps 1–3 are pure Python (no LLM).
  - Steps 4–7 invoke the tie-break LLM call.
  - Mode-based reweighting per §4.5 last paragraph.
  - Hard precondition checks for `deliberate` (landscape map valid + breadth awareness).
- [ ] Update `TurnPipeline.process_turn` to add cognition phase BEFORE production phase:
  - Sequential: Understander → SessionTheorist → UserModeler (read snapshot only, no maintenance) → WorldModeler → Navigator.
  - Persist UserView, WorldView, SessionTheory, Directive snapshots to their tables.
  - Pass the Directive to the producer (existing producers ignore it for now; Phase 3 will wire them).
- [ ] Note: cognition runs sequentially; latency adds ~3–5s ordered. Add structured timing logs.
- [ ] Unit tests for Navigator:
  - `test_navigator_checkpoint_fires_on_confusion`
  - `test_navigator_checkpoint_respects_5_turn_minimum`
  - `test_navigator_checkpoint_fires_on_10_turn_fallback`
  - `test_navigator_reactive_card_routes_to_elicit`
  - `test_navigator_not_probed_dimensions_route_to_elicit`
  - `test_navigator_decide_without_landscape_routes_to_landscape_map`
  - `test_navigator_decide_with_landscape_routes_to_deliberate`
  - `test_navigator_quick_intent_routes_to_answer_grounded`
  - `test_navigator_researcher_mode_boosts_landscape`
  - `test_navigator_coach_mode_biases_to_short_voice`
- [ ] Integration test: re-run 5 v1 transcripts through cognition. Save the Directives produced. Manually inspect — do they look reasonable? This is the Stage 2 → Stage 3 hard gate.

**Done when:** Every turn produces a Directive. Cognition trace visible in logs. Stage 2 → Stage 3 gate passes (Directives on v1 transcripts look right).

**Commit message:** `feat(v2): Navigator agent + cognition pipeline integration with sequential trace`

> **GATE: Stop here. Re-run logged v1 conversations through cognition. Inspect Directives manually. If they don't look reasonable, fix cognition before starting Phase 3.**

---

## Phase 3 — New Producers

### Task 6 — Elicitor + Checkpointer

**Goal:** Two new small producers covering `elicit` and `checkpoint` moves.

- [ ] Add `elicitor` and `checkpointer` blocks to `config/models.yaml` (both Coach-tier; Elicitor temp 0.6, Checkpointer temp 0.4).
- [ ] Create `config/prompts/elicitor.j2`:
  - One reflection + one question per §5.1.
  - Turn 0 expectation-setting line, gated by `turn_index == 0`.
  - Blended-elicitation allowance (≤2-sentence partial answer before pivoting to question).
  - Strict: exactly one `?` in output.
- [ ] Create `config/prompts/checkpointer.j2`:
  - Per §5.6 structure (fairly sure / guessing / confused about / what this is about / does this land?).
  - Must end with a falsifiable question.
- [ ] Create `src/career_coach/agents/elicitor.py` and `checkpointer.py` with retry-on-parse-failure.
- [ ] Wire both into the producer router (still gated by Directive.move; Phase 3 makes the router move-aware).
- [ ] Unit tests:
  - `test_elicitor_turn_0_includes_expectation_setting`
  - `test_elicitor_response_has_exactly_one_question_mark`
  - `test_elicitor_response_references_user_message_specifically`
  - `test_elicitor_blended_partial_answer_under_two_sentences`
  - `test_checkpointer_includes_all_four_sections`
  - `test_checkpointer_ends_with_falsifiable_question`
- [ ] Integration test: feed an Elicitor turn 0 a generic opener; verify expectation-setting prefix appears. Feed a Checkpointer a confused user_view; verify all four sections present.

**Done when:** Elicitor and Checkpointer run end-to-end against fixture inputs. Critic checks pass on outputs.

**Commit message:** `feat(v2): Elicitor + Checkpointer producers`

---

### Task 7 — Landscape Mapper (Researcher breadth + Synthesizer map renderer)

**Goal:** The `landscape_map` move. Researcher in breadth mode, Synthesizer producing a structured `LandscapeMap`.

- [ ] Add a `breadth` parameter to the Researcher's `RetrievalPlan` shaping logic. When `breadth=True`:
  - Planner generates queries that enumerate the space (e.g., "career paths for math/CS graduates UK") rather than narrowing.
  - Top-N fetch widens to cover more sources at lower depth per source.
- [ ] Create `config/prompts/researcher_planner.j2` variant or a `breadth_mode` conditional block in the existing template.
- [ ] Create `config/prompts/synthesizer_map_renderer.j2`:
  - Input: ResearchBrief (breadth findings) + WorldView.relevant_landscape_categories + user_facts.
  - Output: `LandscapeMap` JSON with ≥4, ≤10 categories, plus `excluded_categories` with reasons, plus a prose summary.
  - Strict instruction: each category gets a fit rating tied to user_facts.
- [ ] Create `src/career_coach/agents/landscape_mapper.py` orchestrating Researcher + Synthesizer.
- [ ] Persist output: write `LandscapeMap` to `landscape_maps` table. Update `world_views.landscape_map_valid = TRUE`.
- [ ] Unit tests:
  - `test_landscape_mapper_produces_4_to_10_categories`
  - `test_landscape_mapper_includes_excluded_categories_with_reasons`
  - `test_landscape_mapper_each_category_has_fit_overall_rating`
  - `test_landscape_mapper_categories_match_world_view_relevant_list` (consistency check)
  - `test_landscape_mapper_kb_tier_marked_per_category`
- [ ] Integration test: fixture user (Polytechnique math/CS), real KB, real Tavily call. Verify a reasonable 6–8 category landscape comes back, with `quant_finance`, `investment_banking`, `software_engineering` at minimum.

**Done when:** Landscape Mapper produces structured maps that pass the Critic checks for `landscape_map` (Phase 4 adds those; for now just validate the structure).

**Commit message:** `feat(v2): Landscape Mapper — Researcher breadth mode + Synthesizer map renderer`

---

### Task 8 — Calibrated Partial Coach mode + Coach voice constraints

**Goal:** The `calibrated_partial` move + general voice constraint enforcement on the Coach.

- [ ] Extend `config/prompts/coach.j2` to honor `voice_constraints` from the Directive:
  - `ban_advice_verbs`: forbid bare "consider/explore/look into/experiment with" — must be paired with a named entity.
  - `name_gaps`: response must include an explicit "what I don't know" section.
  - `target_length`: short / medium / long.
  - `must_carry_questions`: append N questions after main content.
- [ ] Create `config/prompts/coach_calibrated_partial.j2` (or a conditional block in `coach.j2`):
  - Required structure: defensible answer / what I don't know / one carried question.
  - The carried question is the highest-priority entry from `directive.scope.open_questions_to_carry`.
- [ ] Update `src/career_coach/agents/coach.py` to dispatch by `directive.move` (`answer_grounded` vs `calibrated_partial`).
- [ ] Unit tests:
  - `test_coach_answer_grounded_respects_ban_advice_verbs`
  - `test_coach_answer_grounded_respects_name_gaps`
  - `test_coach_calibrated_partial_includes_three_required_sections`
  - `test_coach_calibrated_partial_carries_highest_priority_question`
- [ ] Integration test: fixture turn where Navigator emits `calibrated_partial`, verify producer output structure.

**Done when:** Coach handles both `answer_grounded` and `calibrated_partial` modes correctly. Voice constraints enforced.

**Commit message:** `feat(v2): Coach voice constraints + calibrated_partial mode`

---

## Phase 4 — Quality

### Task 9 — Synthesizer rewrite + committee_voice + move-aware Critic checks

**Goal:** Kill the committee voice failure mode. Add all move-aware Critic failure modes.

- [ ] Rewrite `config/prompts/synthesizer.j2` per §5.4:
  - Coach/DA/Researcher outputs labeled neutrally ("Considerations", "Counter-considerations", "Research findings"). No agent names in the labels.
  - Explicit anti-instruction: "Do not narrate the reasoning process. Do not name or refer to internal agents. Speak in one voice."
  - Add few-shot bad/good pairs.
- [ ] Add `committee_voice` regex pre-check to `src/career_coach/agents/critic.py`:
  - Patterns: `r'\b(the )?(coach|devil\'?s advocate|synthesi[sz]er|researcher|critic)\b.*(suggest|argue|raise|point out|consider|emphasi[sz]e|note|flag|recommend)'`
  - And: `r'\baccording to the (coach|devil|researcher)'`
  - Tune via `tests/regex_false_positives.txt` (see HANDOFF risk note).
- [ ] Implement all move-aware failure modes from §6:
  - `elicit`: `premature_advice` (advice verbs without question), `multiple_questions` (>1 `?`), `unanchored_reflection` (no quote-or-paraphrase of last user message).
  - `landscape_map`: `narrow_landscape` (categories < 4), `unfounded_exclusion` (excluded_categories with empty reason), `niche_drift` (nested-niche detection).
  - `answer_grounded`: `vague_action` (advice verbs without named entity in same sentence), `committee_voice`.
  - `deliberate`: `committee_voice` plus existing v1 set.
  - `calibrated_partial`: `gaps_unnamed` (no "what I don't know" section), `no_carried_question`, `vague_action`.
  - `checkpoint`: `not_falsifiable`, `dump_format`.
- [ ] Update `config/prompts/critic.j2` to be move-aware: receives `directive.move` and applies the appropriate check set.
- [ ] Unit tests for each new failure mode (one fixture per mode):
  - 11 new test cases minimum, plus regression tests for existing v1 modes.
- [ ] Integration test: re-run all four `tests/v1_alpha_failures/` fixtures through v2. Verify each is now caught by the appropriate Critic check.
- [ ] Regression test: run all v1 Flow C canary fixtures, verify they still pass post-Synthesizer-rewrite.

**Done when:** All four v1 alpha failures are caught by Critic. v1 Flow C canaries still pass. Regex false-positive list is documented.

**Commit message:** `feat(v2): Synthesizer rewrite + committee_voice + move-aware Critic failure modes`

---

### Task 10 — Supervisor move_mismatch + two-stage recovery flow

**Goal:** Supervisor catches misroutes. Recovery is turn-spanning (§5.7), not within-turn retry.

- [ ] Add `move_mismatch` check to Supervisor:
  - Input: response text + directive.move.
  - Logic per move (e.g., elicit response containing advice verbs without a question; landscape_map response without a structured map; etc.).
- [ ] Implement `RecoveryProbe` queueing:
  - When Critic or Supervisor catches an unrecoverable failure, do NOT retry within the turn.
  - Producer outputs `calibrated_partial` with a transparent failure note (warmly phrased).
  - Queue a `RecoveryProbe` to `recovery_probes` table with `probe_kind` matching the failure mode.
- [ ] Implement Recovery Probe runner (background worker):
  - Polls `recovery_probes` for unconsumed probes.
  - Dispatches to: `user_modeler_audit`, `research`, or `kb_gap_fill` based on `probe_kind`.
  - Writes results to `open_questions_queue` or `world_view.knowledge_gaps`.
- [ ] Update Navigator to consume probe outputs naturally on subsequent turns (already does via existing queue reads; verify no special-case logic needed).
- [ ] Unit tests:
  - `test_supervisor_move_mismatch_detects_advice_in_elicit`
  - `test_supervisor_move_mismatch_detects_no_map_in_landscape`
  - `test_recovery_probe_queued_on_critic_failure`
  - `test_recovery_probe_consumed_by_next_turn_navigator`
- [ ] Integration test: 3-turn sequence where turn 1 produces a misroute, turn 2 picks up the recovery output, turn 3 confirms the issue is resolved.

**Done when:** Supervisor catches move mismatches. Two-stage recovery fires reliably. Next turn picks up recovery output.

**Commit message:** `feat(v2): Supervisor move_mismatch + two-stage recovery flow`

---

## Phase 5 — Background + Observability

### Task 11 — KB Curator background worker

**Goal:** Auto-draft KB entries from research findings on flagged knowledge gaps.

- [ ] Create `src/career_coach/workers/kb_curator.py`:
  - Polls `world_views` for unresolved `knowledge_gaps`.
  - For each gap, runs a focused research query.
  - Drafts a YAML in `kb_draft/` per the `CareerEntry` schema, with citations on every numeric field.
  - Logs a notification (structured log entry) flagging entries pending review.
- [ ] Add Curator schedule:
  - Nightly batch via cron / n8n / Supabase scheduled function (whatever the project already uses).
  - On-demand trigger from World Modeler when a gap blocks the current turn.
- [ ] Unit tests:
  - `test_kb_curator_drafts_valid_yaml` — fixture knowledge gap, verify generated YAML loads cleanly through `WorldKBRepo`.
  - `test_kb_curator_every_numeric_field_has_citation` — structural lint on output.
  - `test_kb_curator_skips_already_drafted` — idempotency.
- [ ] Integration test: fixture user with a knowledge gap not in canonical or draft KB. Run Curator. Verify a new file appears in `kb_draft/` with expected structure.

**Done when:** Curator drafts entries that pass schema validation and round-trip through `WorldKBRepo`. Manual review path documented.

**Commit message:** `feat(v2): KB Curator background worker for kb_draft tier`

---

### Task 12 — Token rollup + cognition trace SSE stream

**Goal:** Backend observability — per-turn token totals + live progress events to the frontend.

- [ ] Implement token rollup:
  - After each turn closes, compute totals across all `agent_calls` for the turn.
  - Categorize: cognition (Understander, SessionTheorist, UserModeler, WorldModeler, Navigator) / production (everything else in the producer chain) / background (Profiler, Curator, etc.).
  - Write to `turns.cognition_tokens_in/out`, `production_tokens_in/out`, `background_tokens_in/out`.
  - Compute `total_cost_usd_estimate` from a model price table in `config/model_prices.yaml`.
- [ ] Implement SSE stream:
  - New endpoint `GET /chat/{session_id}/stream`.
  - Each cognition agent emits a status event when it starts: `{phase, summary, started_at}`.
  - Producer emits start event too: `composing_response` (with optional sub-event `researching_X`).
  - Frontend consumes via `EventSource` (Phase 6 wires the UI; this task ships the backend).
- [ ] Add SSE event emitter to each cognition agent's `run()` method (small helper, not a refactor).
- [ ] Unit tests:
  - `test_token_rollup_categorizes_correctly` — fixture turn with known agent_calls, verify column totals.
  - `test_cost_estimate_computes_from_price_table`.
  - `test_sse_stream_emits_phase_events_in_order`.
- [ ] Integration test: full turn through the pipeline, capture SSE events, verify ordered sequence covers all cognition + production phases.

**Done when:** `turns` row contains correct token totals after every turn. SSE endpoint emits live phase events that match the cognition trace.

**Commit message:** `feat(v2): per-turn token rollup + cognition trace SSE stream`

---

## Phase 6 — Frontend + UX

### Task 13 — Chart rendering + clickable citations + mode toggle

**Goal:** User-facing UI improvements: charts render, citations are real links, mode toggle visible.

- [ ] Recharts integration in the React frontend:
  - Component that takes a `ChartSpec` and renders the appropriate chart type (line, bar, scatter for v2).
  - Inline below prose, sized for readability (responsive).
- [ ] Citation rendering:
  - Frontend renders `[N]` numerals in response text as clickable anchors against `citations[N-1]`.
  - For `source_type=web`, link to URL in new tab.
  - For `source_type=kb`, link to `/kb/career/{name}` (next bullet).
- [ ] Implement `/kb/career/{name}` endpoint:
  - Backend route that loads the YAML (canonical or draft) and renders a readable HTML view.
  - Show `tier=draft` warning banner if applicable.
- [ ] Mode toggle UI:
  - Toggle in chat header: "Coach" / "Researcher".
  - Stored in `sessions.mode`; backend reads on next turn.
  - Inline tooltip explains modes on first encounter.
  - Visual differences: Researcher mode emphasizes citations and charts; Coach mode is denser prose.
- [ ] Frontend SSE client:
  - Consume the SSE stream from Task 12.
  - Render a thin "thinking" indicator showing the current phase.
  - Hide once response arrives.
- [ ] Unit tests (frontend):
  - Chart component renders LineChart from a fixture ChartSpec.
  - Citation `[1]` becomes a clickable anchor.
  - Mode toggle updates `sessions.mode` via API call.
- [ ] Manual QA pass: walk through a 5-turn conversation in each mode, verify charts render, citations click correctly, mode toggle works mid-session.

**Done when:** All four user-facing UX additions work end-to-end on the live frontend.

**Commit message:** `feat(v2): chart rendering + clickable citations + mode toggle + SSE indicator`

---

### Task 14 — Admin under-the-hood panel (auth-gated)

**Goal:** Dev observability surface. NOT user-facing.

- [ ] Auth gate:
  - New route `/admin/*` requires authentication.
  - Use whatever auth mechanism the project uses (Supabase auth, basic auth, env-flag, etc.).
  - Document the choice clearly; default to most secure available.
- [ ] Per-turn dev panel:
  - URL: `/admin/turns/{turn_id}`.
  - Shows: full Directive (mode, move, scope, voice_constraints, rationale, confidence).
  - UserView snapshot at the time of the turn.
  - WorldView snapshot.
  - Cognition trace (every agent's input/output, latency, tokens) — read from `agent_calls`.
  - Producer output (raw, before frontend rendering).
  - All Critic verdicts including retries.
  - Supervisor result.
  - Citations resolved with clickable URLs.
  - Token breakdown by phase.
- [ ] Cross-turn token rollup view:
  - URL: `/admin/usage`.
  - Per-session cost, per-user cost, cost split (cognition / production / background).
  - Top-N most expensive turns ranked.
  - Cost-per-turn histogram.
  - SQL views, no new tables.
- [ ] Failure mode dashboard:
  - URL: `/admin/quality`.
  - Count per Critic failure mode and Supervisor flag, over time, per move.
  - Lets you see if `committee_voice` is decreasing after the Synthesizer rewrite, etc.
- [ ] Recovery probe inspector:
  - URL: `/admin/recovery`.
  - List of probes with status (queued / running / consumed). Filter by failure_mode and probe_kind.
- [ ] Manual QA: walk through 10 turns, click into the dev panel for each, verify all sections render with real data.

**Done when:** Admin panel renders correctly for any logged turn. Cross-turn rollups load. Failure mode dashboard shows real counts. Auth gate is verified.

**Commit message:** `feat(v2): admin dev panel — directive trace, token rollup, quality dashboard, recovery inspector`

---

## Phase 7 — Validation

### Task 15 — Integration tests + canary fixtures + v1 regression suite

**Goal:** End-to-end validation. v2 must fix v1 failures without regressing v1 wins.

- [ ] v1 alpha failure regression tests:
  - For each of the four failure modes, fixture from `tests/v1_alpha_failures/` is run end-to-end.
  - Assert: the response now passes Critic where v1's response would have failed (or actively did fail).
  - Assert: no `move_mismatch` Supervisor flag.
- [ ] v1 Flow C canary regression suite:
  - All v1 Flow C canary fixtures run end-to-end against v2.
  - Assert: each still passes (no regression from Synthesizer rewrite).
- [ ] First-session canary:
  - Fresh user, 5-turn conversation.
  - Assert turn 0 includes expectation-setting prefix.
  - Assert turn 0–3 are mostly `elicit` moves.
  - Assert turn 4+ has `landscape_map` or `answer_grounded` available.
  - Assert completeness vector advances at least 2 dimensions to `partial` or better.
- [ ] Two-mode canary:
  - Same conversation in Coach mode and Researcher mode.
  - Assert mode-appropriate move bias (Researcher mode produces more landscape_map and deliberate; Coach more elicit).
  - Assert citations and charts present in Researcher responses where data permits.
- [ ] Recovery flow canary:
  - Synthetic turn that triggers a `move_mismatch`.
  - Verify calibrated_partial response is produced with transparent failure note.
  - Verify recovery probe is queued.
  - Verify next turn benefits from probe output.
- [ ] Confusion checkpoint canary:
  - 3-turn sequence that produces 2 confusion signals.
  - Verify checkpoint fires automatically on turn 4.
  - Verify minimum 5-turn gap is respected if a second confusion sequence triggers.
- [ ] 10-turn end-to-end canary:
  - Full conversation against the live system.
  - No `move_mismatch` Supervisor flags.
  - Token rollup populated on all turns.
  - SSE events captured for each turn.
  - Admin panel renders all 10 turns correctly.
- [ ] Add a `reports/v2_baseline.md` documenting:
  - Pass/fail per canary.
  - Token cost summary across the full canary suite.
  - Latency p50/p95 per move.
  - Critic fail-rate per failure mode (should be near zero on canaries).

**Done when:** All canaries pass. `reports/v2_baseline.md` exists with measurable results. v2 alpha-ship gate per HANDOFF_v2.md §10 is met.

**Commit message:** `test(v2): integration canaries + v1 regression suite + baseline report`

---

## Summary

| Phase | Tasks | What ships |
|-------|-------|-----------|
| 1. Foundation | 1 | Migration, types, context slicing |
| 2. Cognition | 2–5 | SessionTheorist, WorldModeler, UserModeler, Navigator |
| 3. Producers | 6–8 | Elicitor, Checkpointer, Landscape Mapper, Calibrated Partial |
| 4. Quality | 9–10 | Synthesizer rewrite, move-aware Critic, Supervisor + recovery |
| 5. Background + observability | 11–12 | KB Curator, token rollup, SSE stream |
| 6. Frontend + UX | 13–14 | Charts, citations, mode toggle, admin panel |
| 7. Validation | 15 | Canaries + regression + baseline report |

15 commits. Hard gate after Task 5 (cognition produces reasonable Directives on v1 transcripts). Stage 6 can start any time after Stage 1 if you want to parallelize.
