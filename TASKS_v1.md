# v1 Build Tasks — Execute in Order

Eleven tasks, one commit each, do NOT combine. Read `SPEC_v1.md` fully before Task 0.

The ordering: bug fixes from v0.5 review (Task 0) before piling new agents on shaky foundations. Schema and infrastructure adapters first (Tasks 1-3) because they have no code dependencies. Researcher (Task 4) once both web and KB are available. The two new reasoning agents in isolation (Tasks 5-6). Flow C wiring composes them (Task 7). Supervisor (Task 8) is cross-cutting, benefits from having Flow C to test against. LangGraph migration (Task 9) last, when graph is fully populated. Quality eval (Task 10) measures the whole thing.

Before starting Task 0: confirm the v0.5 definition-of-done still passes on a clean DB AND you have at least one week of real usage data on v0.5. Do not start v1 without this.

---

## Task 0 — v0.5.1 patch (bugs found in v0.5 review)

**Goal:** Fix four issues identified in the v0.5 spec review before adding any new code on top.

- [ ] Fix `extra_body` shape for Qwen3 thinking suppression. Verify against current HF Inference API docs that the OpenAI-compatible endpoint accepts `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}` for Qwen3 models, and update `config/models.yaml` accordingly. Run a manual curl smoke test confirming a Qwen3 response no longer contains `<think>...</think>` blocks.
- [ ] Add an explicit definition for `onboarding_turn_index` in `config/prompts/onboarder.j2` and the OnboardingPipeline: `count of prior turns for this user where flow_used='onboarding'`. Write a unit test asserting that a user starting their second session at onboarding_turn_index=1 (not 0) gets the values probe, not the context probe.
- [ ] Resolve the Coach/Critic rule conflict on hypotheses: update `config/prompts/coach.j2` and the Critic prompt to make explicit that a referenced hypothesis UUID counts toward the Critic's ≥2 grounding requirement. Re-run Critic battery, assert ≥8/10 still passes.
- [ ] Fix the filename references in `HANDOFF_v0_5.md` (currently says `SPEC_v0.5.md` but actual file is `SPEC_v0_5.md`).

**Done when:** All four fixes committed. Critic battery still ≥8/10. Quick manual smoke test through the demo script confirms no `<think>` blocks leak into agent_calls payloads.

**Commit message:** `fix: v0.5.1 patches — qwen3 thinking suppression, onboarding turn index, coach/critic hypothesis rule, handoff filename`

---

## Task 1 — Migration 004 (schema additions)

**Goal:** Add the v1 schema (research_briefs, supervisor_events, turns columns including chart_specs) before any code depends on it.

- [ ] Write `migrations/004_v1_agents.sql` per spec §12.
- [ ] Apply migration against local and CI Postgres instances.
- [ ] Verify: insert dummy rows into `research_briefs` and `supervisor_events` via psql, confirm FKs to `turns` work, confirm `chart_specs` column accepts JSON arrays.
- [ ] Integration test `tests/integration/test_migration_004.py`: apply all migrations fresh, insert minimal fixtures, read back.
- [ ] Document the new tables and columns in `docs/architecture.md`.

**Done when:** Migration applies cleanly. New tables queryable. Existing tests unchanged.

**Commit message:** `feat: migration 004 — research_briefs, supervisor_events, flow C and chart columns`

---

## Task 2 — Web Search Adapter

**Goal:** Provider-agnostic web search infrastructure. Tavily working end-to-end.

- [ ] Create `src/career_coach/web/client.py` — `WebSearchClient` protocol + `SearchResult` Pydantic model per spec §4.2.
- [ ] Create `src/career_coach/web/tavily.py` — `TavilyClient` implementation. Use Tavily's REST API directly via httpx, not their Python SDK (keeps deps lean).
- [ ] Create `src/career_coach/web/factory.py` — reads `config/web_search.yaml`, returns configured client per agent.
- [ ] Create `src/career_coach/web/cache.py` — in-process LRU cache, 24h TTL, keyed per spec §4.4.
- [ ] Create `src/career_coach/web/quota.py` — daily quota tracking with graceful degradation per spec §4.5.
- [ ] Create `MockWebSearchClient` in `tests/fixtures/` for use by agent tests — same shape as `MockLLMClient`.
- [ ] Add `TAVILY_API_KEY` to `.env.example` with comment about free tier.
- [ ] Add `web_search.yaml` to `config/`.
- [ ] Unit tests:
  - `test_tavily_search` against a recorded fixture (use vcrpy or hand-recorded JSON).
  - `test_quota_exhaustion_returns_empty_with_flag`.
  - `test_cache_hits_dont_call_api` (mock the underlying httpx client).
  - `test_fetch_extracts_main_content` (test fixture: a sample HTML page, assert clean text out).
- [ ] Integration test (skipped if no `TAVILY_API_KEY` in env): live search for "quant finance careers UK 2026", assert ≥3 results with valid URLs, fetch top result, assert non-empty content.
- [ ] Update `agent_calls` logging to include `web_search` calls per spec §4.6 — add a small wrapper or adapt `Agent.log_call` so non-LLM agents can log too.

**Done when:** `factory.get_client("researcher").search("test query")` returns real results from Tavily. Mock client works in unit tests. Quota exhaustion returns empty + flag, doesn't error.

**Commit message:** `feat: web search adapter with Tavily — cache, quota, observability`

---

## Task 3 — World Knowledge Base v0

**Goal:** 15 career YAMLs + repo, zero agent wiring yet (Researcher in Task 4 consumes this).

- [ ] Create `kb/careers/` with the 15 files per spec §13. Every file has `source_notes` on every numeric field. NO fake precision.
- [ ] Create `kb/pay_bands_uk.yaml` and `kb/exits.yaml`.
- [ ] Write `src/career_coach/kb/models.py` — Pydantic `CareerEntry`, `PayBand`, `ExitPath`.
- [ ] Write `src/career_coach/kb/repo.py` — `WorldKBRepo` with `get_career(name)`, `search_by_tags(tags)`, `list_all_careers()`.
- [ ] Unit tests: load all 15 files, round-trip serialize, assert required fields present, assert every numeric field has a source_notes sibling (this is a structural lint).
- [ ] Integration test `test_kb_search_by_tags` — given tag `math_heavy`, returns quant_finance + ML research + academia_stem.

**Note on content authoring:** Of the 15 files, the user (Georgy) writes the first 5 (quant_finance, investment_banking, software_engineering, academia_stem_phd_track, medicine_uk). Claude Code drafts the remaining 10. EVERY `source_notes` field is reviewed by the user before commit — these can't be hand-waved.

**Done when:** `WorldKBRepo(Path("kb")).list_all_careers()` returns 15 entries, all validate, source_notes lint passes.

**Commit message:** `feat: world KB v0 — 15 UK career entries with source notes`

---

## Task 4 — Researcher agent

**Goal:** ResearchBrief production using both web search and KB, with citation hygiene.

- [ ] Add `ResearcherInput`, `Citation`, `Finding`, `ResearchBrief`, `RetrievalPlan` to `models/agent_io.py`.
- [ ] Add `researcher` and `researcher_planner` blocks to `config/models.yaml` (Coach-tier and Haiku-tier respectively).
- [ ] Create `config/prompts/researcher_planner.j2` — takes question + user_facts, returns RetrievalPlan JSON (search queries + KB tags).
- [ ] Create `config/prompts/researcher.j2` — takes question + retrieved web content + KB files + user_facts, returns ResearchBrief JSON. Encode citation hygiene rules (§5.5) directly in the prompt.
- [ ] Create `src/career_coach/agents/researcher.py`:
  - Two-phase mechanics per spec §5.4.
  - Web search and KB lookup run in parallel (`asyncio.gather`).
  - Top-2 fetch per query, also parallel.
  - Latency budget enforcement (8s shallow, 15s deep) — partial results returned with truncation caveat.
  - Same retry-on-parse-failure pattern from v0.5 Task 1.
  - Persist to `research_briefs` table.
- [ ] Unit tests with `MockLLMClient` + `MockWebSearchClient`:
  - `test_researcher_planner_outputs_valid_plan` — verify plan structure.
  - `test_researcher_handles_empty_web_results` — degrades to KB-only with caveat.
  - `test_researcher_handles_empty_kb_match` — uses web alone, no error.
  - `test_researcher_handles_both_empty` — returns empty findings + caveats, no error.
  - `test_researcher_truncates_on_timeout` — slow mock fetches, partial findings + truncation caveat.
- [ ] Integration test with real Tavily and real LLM: feed "Should I go into quant finance?" + user_facts `{age: 20, major: Mathematics}`, assert ≥2 findings, ≥1 caveat, every finding has ≥1 citation, web_sources_consulted non-empty.
- [ ] Quality battery: 20 deep-decision fixtures in `tests/quality/researcher_fixtures/`. Run via `scripts/run_researcher_eval.py`. Asserts §14.2 thresholds.

**Done when:** Researcher returns valid ResearchBrief in all scenarios (full input, empty web, empty KB, both empty, timeout). Quality battery passes thresholds.

**Commit message:** `feat: Researcher agent — web + KB with citation hygiene`

---

## Task 5 — Devil's Advocate agent

**Goal:** Structured contrarian pressure on Coach output, with honest-agreement path.

- [ ] Add `DevilsAdvocateInput`, `CounterPoint`, `Risk`, `DevilsAdvocateOutput` to `models/agent_io.py`.
- [ ] Add `devils_advocate` block to `config/models.yaml` (Coach-tier).
- [ ] Create `config/prompts/devils_advocate.j2` per spec §8 principles.
- [ ] Create `src/career_coach/agents/devils_advocate.py` with retry-on-parse-failure.
- [ ] Unit tests:
  - `test_da_honest_agreement` — mock returns `agrees_with_coach=true` with empty counter_points, asserts no error and verifies output.
  - `test_da_grounds_in_user_profile` — verify rendered prompt includes user_facts.
  - `test_da_severity_required` — mock omits severity, parse fails, retry succeeds.
- [ ] Integration test with hand-crafted Coach output + seeded user facts: assert ≥2 counter_points with severity populated, assert at least one source_type=="user_profile".

**Done when:** DA produces valid output across both scenarios (disagreement and honest agreement). No tests rely on severity defaulting to "high".

**Commit message:** `feat: Devil's Advocate agent — structured counter-points with honest agreement`

---

## Task 6 — Synthesizer agent (with charts)

**Goal:** Integrate Coach + DA + Researcher into one user-facing response, optionally including a chart.

- [ ] Add `SynthesizerInput`, `SynthesizedResponse`, `ChartSpec`, `AxisSpec` to `models/agent_io.py`.
- [ ] Extend `CoachOutput` to include optional `chart_specs: list[ChartSpec]` (max 1 in v1).
- [ ] Add `synthesizer` block to `config/models.yaml` (Coach-tier).
- [ ] Create `config/prompts/synthesizer.j2` per spec §9 principles, including:
  - Chart emission gate: only when §6.4 conditions hold.
  - Inline citation numerals `[1]`, `[2]`.
- [ ] Update `config/prompts/coach.j2` to support optional chart emission on Flow B (max 1, same gates).
- [ ] Create `src/career_coach/agents/synthesizer.py` with retry-on-parse-failure.
- [ ] Unit tests:
  - `test_synthesizer_integrates_all` — all three inputs non-empty, assert `integrated_from` contains all three.
  - `test_synthesizer_empty_research` — empty research brief, `integrated_from` contains only `coach` and `devils_advocate`.
  - `test_synthesizer_da_agrees_allows_empty_tradeoffs` — DA `agrees_with_coach=true`, empty `surfaced_tradeoffs` valid.
  - `test_synthesizer_chart_emission_gated` — input lacks ≥3 comparable items, mock includes a chart anyway, assert validation/Critic catches it (use `chart_data_invented` failure mode).
  - `test_synthesizer_chart_with_valid_data` — input has comparable findings, valid chart with `source_citation_indices` populated.
- [ ] Integration test: hand-crafted inputs → response with inline citations, ≥1 tradeoff word, optional chart referencing real data.

**Done when:** Synthesizer produces valid SynthesizedResponse across all scenarios. Charts are gated correctly (no invented data, citations required).

**Commit message:** `feat: Synthesizer agent — integrates Coach+DA+Researcher with optional charts`

---

## Task 7 — Flow C wiring

**Goal:** Orchestrator routes to Flow C; pipeline runs Researcher + Coach in parallel, then DA, then Synthesizer, then Critic with retries.

- [ ] Extend `Orchestrator.decide(intent, is_new_user, fact_count) -> Flow` per spec §7.1. Returns `"C"` under stated conditions.
- [ ] Update `TurnPipeline.process_turn` (still hand-rolled asyncio; LangGraph migration is Task 9):
  - Load `fact_count` once, pass into `Orchestrator.decide`.
  - On flow `"C"`: `asyncio.gather(researcher.run(...), coach.run(...))`, then `await devils_advocate.run(...)`, then Synthesizer loop with Critic retry (max 2).
  - Persist `research_brief_id`, `devils_advocate_output`, `synthesizer_output`, `chart_specs` on the turn row.
  - Critic retries Synthesizer only, not Researcher or DA.
  - Total Flow C latency budget 25 s; on timeout, log `fallback_reason="flow_c_timeout"` and return Coach output as fallback.
- [ ] Extend `CriticInput` and `config/prompts/critic.j2` to support Flow C checks per spec §7.3 (`unintegrated`, `uncontested`, `chart_uncited`, `chart_data_invented` failure modes).
- [ ] Extend Critic battery with 5 Flow C fixtures — 4 for each new failure mode + 1 clean reference.
- [ ] Integration test `tests/integration/test_flow_c_end_to_end.py`:
  - Seed user with ≥5 facts including math/research signals.
  - Send: "I'm deciding between a Math PhD and quant finance — help me think about it."
  - Assert `turns.flow_used == "C"`, `synthesizer_output` non-null, `research_brief_id` non-null.
  - Assert response contains ≥2 user-fact references, ≥1 finding-derived claim, ≥1 inline citation `[N]`, ≥1 tradeoff word.

**Done when:** Flow C integration test passes end-to-end with real LLM and real Tavily. Critic catches ≥4/5 Flow C failure fixtures. v0.5 tests still green.

**Commit message:** `feat: Flow C deliberation — parallel Researcher+Coach → DA → Synthesizer → Critic`

---

## Task 8 — Supervisor agent

**Goal:** Cross-cutting safety/correctness monitor on every flow.

- [ ] Add `SupervisorInput`, `SupervisorEvent` to `models/agent_io.py`.
- [ ] Add `supervisor` block to `config/models.yaml` (Haiku-tier).
- [ ] Create `config/prompts/supervisor.j2`:
  - Three checks only: fact_contradiction, off_topic, unsafe.
  - Output a single action: pass | warn | retry | block.
  - Scripted override text only populated on block.
  - Crisis-keyword list maintained as a Jinja2 variable for easy iteration.
- [ ] Create `src/career_coach/agents/supervisor.py`:
  - Early-return `pass` for empty responses (latency budget discipline).
  - Persist every non-`pass` event to `supervisor_events`.
- [ ] Wire into `TurnPipeline.process_turn` post-Critic (or post-Coach on Flow A):
  - `pass` → return response as-is.
  - `warn` → append a brief caveat.
  - `retry` → re-run Coach/Synthesizer once with supervisor feedback; if retry also trips, fall through to `warn`.
  - `block` → return `scripted_override` verbatim.
- [ ] Create `tests/quality/supervisor_redteam/` with 20 fixtures per spec §10.5.
- [ ] Write `tests/quality/test_supervisor_battery.py` — assert ≥8/15 intended triggers caught AND 0/5 false positives.
- [ ] Integration test verifying Supervisor runs on Flow A, Flow B, Flow C, and onboarding.

**Done when:** Supervisor red-team thresholds pass. Full pipeline integration test confirms Supervisor runs on every flow.

**Commit message:** `feat: Supervisor agent — runtime safety and correctness monitor`

---

## Task 9 — LangGraph migration

**Goal:** Internal refactor of `pipeline/turn.py` to a LangGraph StateGraph. Zero behavior change.

- [ ] Add `langgraph` to dependencies. Pin to current stable.
- [ ] Define `TurnState` TypedDict carrying all intermediate agent outputs.
- [ ] Define nodes: `understander`, `orchestrator`, `coach`, `researcher`, `devils_advocate`, `synthesizer`, `critic`, `supervisor`, `onboarding`, `profiler_dispatch`.
- [ ] Define conditional edges matching flow routing per spec §7.2.
- [ ] Critic-driven retry stays explicit (reads verdict from state and loops back to Synthesizer node).
- [ ] Keep `TurnPipeline.process_turn` signature unchanged — becomes a thin wrapper invoking `graph.ainvoke(initial_state)`.
- [ ] NO LangGraph checkpointing, NO HITL, NO tool-calling routing.
- [ ] **All v0.5 and v1 tests pass unmodified.** If any test needs adapting, STOP and diagnose. Either LangGraph is wrong here or the refactor is leaking abstractions.

**Done when:** Full test suite — v0.5 + v1 — green without modifying any existing test. `pipeline/turn.py` is the LangGraph entrypoint, not hand-rolled async.

**Commit message:** `refactor: migrate TurnPipeline to LangGraph, preserve all behavior`

---

## Task 10 — Flow C quality eval

**Goal:** Measure Flow C deliberation quality on a 20-fixture deep-decision battery.

- [ ] Create `tests/quality/deep_decision_fixtures/` — 20 YAML fixtures, each a persona + seed facts + a deep-decision question. Cover:
  - PhD vs industry (3)
  - Finance vs tech vs consulting (3)
  - Pivot decisions mid-degree (3)
  - Early-career firm selection (3)
  - Geographic tradeoffs (2)
  - Staying vs leaving (2)
  - Prestige vs fit (2)
  - Intentional contrarian fixtures where KB+web have genuine gaps (2) — to test graceful degradation.
- [ ] Create `scripts/run_flow_c_eval.py`:
  - For each fixture: create user, seed facts, send question as a Flow C turn.
  - Assert: `flow_used=="C"`, `synthesizer_output` non-null, response contains ≥2 user-fact references, ≥1 finding reference, ≥1 inline citation, ≥1 tradeoff word.
  - Critic verdict = pass/fail.
  - Optional second grader: LLM-as-judge comparing the Flow C response to a Flow B response on the same fixture (use Anthropic for this to avoid same-model self-grading).
  - Output `reports/flow_c_baseline.md` with per-fixture pass/fail + Critic failure modes.
- [ ] Assert: ≥17/20 pass.
- [ ] Assert: ≥5 fixtures emitted a chart, 0 emitted invented-data charts.
- [ ] Track total cost; assert <£25.
- [ ] Commit the baseline report.

**Done when:** Flow C battery scores ≥17/20. Chart emission threshold met. Cost ceiling held. Report checked in.

**Commit message:** `feat: Flow C quality eval — 20 deep-decision fixtures, baseline report`

---

## After v1

Cut a `v1` tag. Use the system for AT LEAST two weeks before planning v1.5. v1 introduces the biggest behavior change to date (Flow C completely rewrites what "a deep question" produces, charts add a new output dimension, web search introduces external dependencies); evaluation needs real user input, not just fixtures.

Collect:

- `SELECT flow_used, count(*), avg(latency_ms) FROM turns JOIN agent_calls USING (turn_id) GROUP BY 1` — flow distribution and latency per flow.
- `SELECT event_type, action_taken, count(*) FROM supervisor_events GROUP BY 1, 2` — what Supervisor catches vs blocks.
- `SELECT count(*) FILTER (WHERE chart_specs != '[]') AS turns_with_charts, count(*) AS total_turns FROM turns WHERE flow_used IN ('B','C') AND created_at > now() - interval '14 days'` — chart emission rate.
- `SELECT agent_name, count(*) FROM agent_calls WHERE agent_name='web_search' AND created_at > now() - interval '14 days'` — web search volume / cost.
- Save and read at least 10 Flow C transcripts end-to-end.

Then plan v1.5 (Active Coach: challenges as first-class objects, Initiator agent, check-in admin endpoint, Profiler updates for follow-ups, engagement_state micro-store).

If two-week data shows Flow C fires < 5% of turns, OR users don't engage with charts, OR Researcher's web findings are systematically distrusted — revise v1 before v1.5, don't paper over.
