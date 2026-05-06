# v0.5 Build Tasks — Execute in Order

Six tasks, one commit each, do NOT combine. Read `SPEC_v0.5.md` fully before Task 1.

The ordering is deliberate: reliability fixes (Task 1) unlock trustworthy measurement for everything after. Schema changes (Task 2) must land before code uses them. Onboarding (Task 3) is the biggest user-facing gap and should be proven next. Session continuity (Task 4) is mostly verification work. Distillation quality (Task 5) is the biggest intellectual gap. Memory loop closure (Task 6) ties it all together.

---

## Task 1 — Reliability fixes (Pillar 0)

**Goal:** Stop silent failures. Every LLM call either succeeds, retries and succeeds, or logs a structured failure row.

- [ ] Fix `Agent.now_ms` latency bug — switch to `time.perf_counter_ns() // 1_000_000`.
- [ ] Add unit test: mock LLM with `await asyncio.sleep(0.02)`, assert `agent_calls.latency_ms >= 20`.
- [ ] Update `config/models.yaml` to include `extra_body: { "thinking": false }` for all four agents.
- [ ] Add retry-once-on-parse-failure logic to:
  - `understander.run` — on `ValidationError` or `JSONDecodeError`, append a strict-format user message and retry ONCE with the same mock-capable `complete()` call. If second attempt fails, use existing fallback.
  - `coach.run` — same pattern.
  - `critic.run` — same pattern (still fail-open to "pass" only after retry fails).
  - `profiler.run` — same pattern (still fall back to empty output only after retry fails).
- [ ] Verify each retry path is exercised by a MockLLMClient test that queues `["", <valid_json>]` — first call returns empty, retry returns valid, assertion passes that both were consumed.
- [ ] Fix the background-task lifecycle bug in `pipeline/turn.py`:
  - Add `self._background_tasks: set[asyncio.Task]` on `TurnPipeline.__init__`.
  - Track every `asyncio.create_task` result; remove the `noqa: RUF006`.
- [ ] Integration test: fire 20 concurrent `process_turn` calls against a pipeline with a mocked slow Profiler (50 ms sleep). Assert `count(agent_calls WHERE agent_name='profiler') == 20` after `await asyncio.gather(…)` returns.

**Done when:**
- `mypy --strict` and `ruff check .` pass.
- New unit tests pass.
- Full existing test suite still passes unmodified.

**Commit message:** `fix: eliminate silent LLM failures — latency measurement, parse retries, background task lifecycle`

---

## Task 2 — Migration 003 (schema additions)

**Goal:** Add `retry_count` and `fallback_reason` columns to `agent_calls` and wire them through `Agent.log_call`.

- [ ] Write `migrations/003_observability.sql` per spec §8.
- [ ] Extend `Agent.log_call` signature:
  ```python
  async def log_call(
      self,
      *,
      turn_id, input_payload, output_payload,
      latency_ms, tokens_in, tokens_out,
      error=None,
      retry_count: int = 0,
      fallback_reason: str | None = None,
  ) -> None: ...
  ```
- [ ] Update each agent to pass `retry_count` and `fallback_reason` when relevant:
  - Retries from Task 1 → `retry_count=1` on the final `log_call`.
  - Existing fallback paths → `fallback_reason="empty_llm_response" | "schema_validation_failed" | "retry_exhausted"`.
- [ ] Integration test `tests/integration/test_observability.py`: run a pipeline turn with a mock LLM that returns an empty string twice, verify the resulting `agent_calls` row has `retry_count=1` AND `fallback_reason="retry_exhausted"`.
- [ ] Document the new columns in `docs/architecture.md`.

**Done when:** Migration applies cleanly. New columns queryable. Existing tests unchanged (they don't assert on these columns).

**Commit message:** `feat: add retry_count and fallback_reason to agent_calls`

---

## Task 3 — Onboarding flow (Pillar 1)

**Goal:** New users get a scripted 3-turn onboarding before the standard coaching pipeline takes over.

- [ ] Create `career_coach/pipeline/onboarding.py`:
  - `OnboardingPolicy.is_new_user(user_id) -> bool` per spec §4.1.
  - `OnboardingPipeline.process_turn(...)` per spec §4.3.
- [ ] Extend `Orchestrator.decide(intent, is_new_user)` to return `"onboarding"` per spec §4.2.
- [ ] Create `config/prompts/onboarder.j2` with the three-probe archetype (spec §4.3). Use Jinja2 to select the probe by `onboarding_turn_index` (0, 1, or 2+).
- [ ] Wire into `TurnPipeline.process_turn`:
  - Call `OnboardingPolicy.is_new_user(user_id)` before `Orchestrator.decide`.
  - If flow == "onboarding", skip Critic loop, pass through `OnboardingPipeline`.
  - Persist turn with `flow_used="onboarding"`.
- [ ] Unit tests:
  - `test_onboarding_policy` — verify threshold logic (5 facts OR 3 turns).
  - `test_onboarding_pipeline_turn_index_selects_correct_probe` with MockLLMClient.
  - `test_orchestrator_routes_new_users_to_onboarding`.
- [ ] Write `tests/quality/test_onboarding_battery.py` with the 5 cold-start fixtures (spec §4.5). Generate `reports/onboarding_baseline.md`.

**Done when:** Onboarding battery passes 5/5. Existing Critic battery unchanged. New user on the live backend gets a welcoming probe question instead of "I need more context."

**Commit message:** `feat: cold-start onboarding flow for new users`

---

## Task 4 — Session continuity verification (Pillar 2)

**Goal:** Prove the within-session memory path works and give the maintainer a repeatable demo.

- [ ] Write `tests/integration/test_session_continuity.py` per spec §5.3. Requires real HF token; skip if absent.
- [ ] Write `scripts/demo_conversation.py` per spec §5.4. Make it useful:
  - Creates user with a chosen `display_name` (CLI arg).
  - Loops through a scripted list of 5 messages (can be read from a text file via `--script` arg).
  - Preserves `session_id` across turns.
  - At the end: runs distillation, pretty-prints final state (facts, hypotheses, evidence count).
  - Writes a transcript to `reports/demo_<timestamp>.md`.
- [ ] Update `README.md`:
  - Add a "Running a real conversation" section showing how to pass `session_id` back.
  - Document the demo script.
- [ ] Manual verification: run the demo script, attach the generated report as an artifact in the commit message.

**Done when:** The integration test passes against a live backend with HF credentials. Demo script runs end-to-end and produces a readable transcript.

**Commit message:** `feat: session continuity test and demo_conversation script`

---

## Task 5 — Distillation quality harness (Pillar 3)

**Goal:** Measure, don't assume, that distillation produces sensible hypotheses from real evidence.

- [ ] Create `tests/quality/distillation_fixtures/` with 10 YAML fixtures per spec §6.2. Each fixture has:
  - `name`, `persona`, `seed_facts`, `turns` (5-8 messages, alternating user/assistant), `expected_hypotheses`, `forbidden_hypotheses`.
- [ ] Create `scripts/run_distillation_eval.py`:
  - Walks all fixtures, creates a fresh user per fixture, seeds facts, replays turns through `process_turn`.
  - Runs distillation at end.
  - For each expected/forbidden theme, calls a judge-tier LLM with a strict prompt.
  - Aggregates score = (matched_expected - matched_forbidden), averages across fixtures.
  - Outputs `reports/distillation_baseline.md`.
- [ ] Add cross-run deduplication to `jobs/distillation.py` per spec §6.4:
  - Before creating a new hypothesis, compare against ALL active hypotheses via a small LLM call.
  - If judged similar, treat as `match_existing` to the returned ID.
- [ ] Integration test `test_distillation_dedup` — seed user, run distillation twice on similar evidence, assert active hypothesis count stays ≤ 2.
- [ ] Run the eval; assert average score ≥ 0.7. If below, STOP and surface findings — don't ship a weak distillation baseline.

**Done when:** Distillation eval produces a baseline report with ≥ 0.7 average. Dedupe test passes.

**Commit message:** `feat: distillation quality eval with 10 fixtures and dedup`

---

## Task 6 — Memory loop closure + Coach prompt tweak (Pillar 4)

**Goal:** Prove the architectural punchline — hypotheses formed in one session get used by the Coach in a later session.

- [ ] Modify `config/prompts/coach.j2` per spec §7.2: the "if active_hypotheses with confidence > 0.5, at least one referenced_hypothesis must be a UUID" rule.
- [ ] Re-run `run_quality_eval.py` — assert Critic battery still ≥ 8/10. If it drops, soften the rule (e.g., "prefer hypotheses over facts when both are available") and iterate.
- [ ] Write `tests/integration/test_memory_loop.py` per spec §7.1:
  - 6-turn conversation in session A (onboarding + standard flow).
  - Distillation. Assert ≥ 1 hypothesis with confidence ≥ 0.3.
  - Brand new session B for same user. One turn asking for deliberation on the earlier topic.
  - Assert the Coach output OR the prompt input references at least one hypothesis from session A.
- [ ] Extend `scripts/demo_conversation.py` with a `--two-session-mode` flag:
  - Session A: 5 turns + distillation.
  - Prints intermediate state.
  - Session B: 1 follow-up turn.
  - Asserts hypothesis reference in the final response (prints ✅ / ❌).

**Done when:**
- Critic battery ≥ 8/10.
- Memory-loop integration test passes.
- Demo script two-session mode prints ✅ for hypothesis reference.
- Full `run_quality_eval.py` still scores ≥ 19/20.

**Commit message:** `feat: close the memory loop — hypotheses reused across sessions`

---

## After v0.5

Cut a `v0.5` tag. Use the system for a week. Collect:
- `SELECT agent_name, fallback_reason, count(*) FROM agent_calls WHERE created_at > now() - interval '7 days' GROUP BY 1, 2`
- `SELECT agent_name, percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms, percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms FROM agent_calls GROUP BY 1`
- The demo-script outputs from the week.

THEN plan v1 with that data in hand. Do not start v1 without running the full v0.5 definition-of-done one more time on a clean DB.

v1 adds, roughly: Researcher, Devil's Advocate, Supervisor, Evaluator, World KB. But v1 scope will be re-decided based on what v0.5 surfaces — do not pre-commit to the full agent expansion.
