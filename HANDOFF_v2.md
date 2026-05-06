# HANDOFF — Career Coach v2

This document hands off `SPEC_v2.md` and `TASKS_v2.md` to the implementer (Claude Code). It frames the work, calls out the validation gates that must pass before starting, names the risks, and specifies the build order.

---

## 1. What v2 is

A load-bearing rebuild of the routing and reasoning layer. v2 introduces a Cognition / Production split: a multi-agent cognition layer produces a typed `Directive`, and the production layer (Coach, Synthesizer, Critic, Supervisor, plus new producers) executes against it. The intelligence about *what to do this turn* migrates upstream into Cognition; Production becomes execution against a clear contract.

This is not an incremental patch. It rewires the turn pipeline.

## 2. Why v2 is a confident decision (not a hypothesis)

v1 was framed as a hypothesis pending real usage data. v2 is different: the four failure modes that motivate it are validated by a real user transcript with the v1 alpha system (committee voice, premature advice, vague action items, generic feel). Each maps to a structural cause that v2 addresses by name.

What remains uncertain is not *whether* the failure modes are real, but *whether the cognition layer's value scales* — i.e., whether the completeness vector and session theorist actually become richer over real conversations rather than producing token-cost without quality lift.

**Validation hypothesis for v2:** within 10 real-user conversations on the v2 alpha, the average response receives at least one structural Critic-pass that v1 would have failed (committee_voice, premature_advice, vague_action, or move_mismatch). If this hypothesis fails, the cognition layer is paying for itself in cost but not in quality, and Stage 6 frontend work should pause until prompts and slicing are tuned.

## 3. Pre-flight gates (DO NOT START until all four pass)

1. **v1 alpha is shipped and instrumented.** Logs from at least 5 real (non-test) conversations exist in `agent_calls` and `turns`. If you don't have this, you don't have a baseline to measure v2 against.
2. **The four failure modes have at least one logged example each.** Pull the transcripts. Save them to `tests/v1_alpha_failures/` as fixtures. v2 must fix all four; you need the fixtures to verify.
3. **Migration 004 (v1) has run cleanly on the production DB.** v2's migration 005 builds on it. If 004 is partial, fix that before starting 005.
4. **A week has passed since v1 alpha shipped.** This is the discipline gate from `HANDOFF_v0_5.md`, applied again. Architectural rebuilds without observed user behavior are how scope creeps and ships nothing useful.

## 4. What's preserved from v1

Keep, do not touch:

- The provider-agnostic LLM adapter (`career_coach.llm.factory`) and its config files.
- The web search adapter and Tavily client.
- The World Knowledge Base file format (15 canonical YAMLs, `source_notes` invariant).
- The Researcher agent's two-phase planner/executor pattern (extended in §5.2 for breadth mode but the core stays).
- The Coach agent's prompt scaffolding (extended for `ban_advice_verbs` and `name_gaps`).
- Devil's Advocate (unchanged).
- LangGraph as the pipeline substrate.
- Migration 004 schema.
- All v1 logging and observability columns.

## 5. What's removed or reduced

- Orchestrator's role as the single routing decision point. The class can stay as a thin compatibility shim or be deleted; Navigator subsumes its job.
- The "decide → Flow C automatically" routing pattern. v2 routing is move-based and gated by preconditions, not flow-based.

## 6. What's new (high-level)

Cognition agents: Session Theorist (promoted from a text field), User Modeler, World Modeler, Navigator. Plus an Elicitor and Checkpointer in Production, a Calibrated Partial Coach mode, a Landscape Mapper move that uses Researcher in breadth mode + Synthesizer as a structured map renderer. KB Curator and Recovery Probe runner as new background workers. New tables for user_views, world_views, landscape_maps, directives, re_examination_cards, audit_findings, checkpoints, recovery_probes.

User-facing additions: cognition trace SSE stream, chart rendering, clickable citations (web + KB), Coach/Researcher mode toggle.

Dev-only additions: per-turn admin panel showing the full Directive, cognition trace, token breakdown, Critic verdicts, recovery probes. Auth-gated, NOT user-facing.

## 7. Build order — read TASKS_v2.md for detail

The 15 tasks fall into seven phases. Stage gating is real:

- **Stage 1 (Foundation, Task 1).** Migration + types + context slicing helper. Nothing runs yet. Validates the schema makes sense.
- **Stage 2 (Cognition, Tasks 2–5).** Each cognition agent in dependency order. Navigator is last because it consumes all the others. **Stop after this stage and validate cognition runs cleanly on logged v1 turns** (re-run v1 transcripts through cognition, verify Directives look reasonable). Do not proceed to producers until cognition is sane.
- **Stage 3 (Producers, Tasks 6–8).** New producers. These can be parallelized if you have multiple sessions of work; otherwise sequential.
- **Stage 4 (Quality, Tasks 9–10).** Synthesizer rewrite, Critic move-aware checks, Supervisor recovery flow. **This is where the four v1 failure modes get fixed.** Run the `tests/v1_alpha_failures/` fixtures after each Critic check is added; verify they now reject what v1 accepted.
- **Stage 5 (Background + observability, Tasks 11–12).** KB Curator, token rollup, SSE stream. Backend completeness.
- **Stage 6 (Frontend, Tasks 13–14).** Chart rendering, clickable citations, mode toggle, admin panel. Can start any time after Stage 1 (admin panel reads from existing tables) but is most valuable after Stage 5 when there's something to display.
- **Stage 7 (Validation, Task 15).** Full integration tests, canary fixtures, regression suite against v1 transcripts.

**Hard gate between Stages 2 and 3.** If cognition isn't producing reasonable Directives on logged v1 conversations, no amount of producer work will save it. Don't paper over a confused Navigator by tuning Elicitor prompts.

## 8. Risks

**Cognition latency compounds.** Sequential cognition adds ~3–5s of ordered overhead before any production work starts. The SSE stream makes this legible, but if a user is on Researcher mode with `deliberate` move, total turn latency could hit 30s+. Track median and p95 from day one. If p95 exceeds 25s on simple turns, something is wrong with cognition (probably WorldModeler over-fetching).

**Synthesizer rewrite could regress Flow C quality.** The committee-voice fix changes the Synthesizer prompt substantially. Existing Flow C cases that worked could degrade. Mitigation: keep all v1 Flow C canary fixtures and run them after the Synthesizer rewrite. If quality drops, the new prompt needs more few-shot examples, not a different architecture.

**Researcher in breadth mode is a new prompt.** It's the mechanism for landscape_map. Easy to mis-prompt into "list every job ever" or "narrow to one path." Test against at least 5 distinct user profiles before declaring it works.

**Storage policy is permissive on purpose.** Full UserView snapshots per turn will accumulate. This is a deliberate choice (principle 10) for the first 10 active users. After that, revisit. Don't optimize prematurely; don't ignore it forever either. Set a calendar reminder.

**The Critic regex pre-checks (committee_voice, vague_action) will have false positives.** Plan for them. If a regex flags a legitimate response, it's better than missing the failure mode in production, but tune the regex before users complain. Keep a `tests/regex_false_positives.txt` file of legitimate strings that must NOT match.

**Mode toggle adds product complexity.** Most users won't read the explanation tooltip. Default to Coach mode. Make Researcher mode discoverable but not pushy.

**Admin panel is a security surface.** Auth-gate it. Do not ship to a public URL without authentication. UserView snapshots contain personal information about real users.

## 9. Day 1 checklist

- [ ] Verify pre-flight gates 1–4 (§3) all pass.
- [ ] Pull at least one transcript from v1 alpha that demonstrates each of the four failure modes. Save as fixtures.
- [ ] Read SPEC_v2.md end to end. Note any disagreement before starting code.
- [ ] Read TASKS_v2.md end to end. Confirm the dependency order makes sense to you.
- [ ] Confirm migration 004 ran cleanly and the v1 schema is intact in the dev DB.
- [ ] Snapshot the dev DB before starting migration 005.
- [ ] Branch off main: `v2-rebuild`. Plan to merge in stages, not all at once.

## 10. Done when

v2 is shipped to alpha when:

- All 15 tasks in TASKS_v2.md are merged.
- Migration 005 has run on production.
- The four v1 failure-mode fixtures all produce structurally different (and qualitatively better) responses on v2.
- The v1 Flow C canary fixtures still pass.
- The cognition trace SSE stream is live in the frontend.
- The admin panel renders directive + trace for at least the last 10 turns.
- A 10-turn end-to-end conversation against the live system completes with no `move_mismatch` Supervisor flags.

After alpha ships, observe at least 10 real conversations before declaring v2 stable. Use the failure mode dashboard (§15.3 of SPEC_v2.md) to track Critic-fail-rate per move. v2 is "stable" when committee_voice, premature_advice, and vague_action each register at <5% of relevant turns over a 7-day window.

---

## 11. After v2

v3 directions, in approximate priority:

1. **Research memory / cache.** Caching findings by semantic similarity. Now that cognition state is rich, this becomes the highest-leverage cost cut.
2. **Initiator agent.** Proactive check-ins between sessions. Requires session-state tracking that v2's cognition layer makes possible.
3. **Interactive LandscapeMap rendering.** v2 ships structured + prose; v3 makes it drillable.
4. **EVoI scoring.** Replace Navigator's heuristic priority ordering with computed expected-value-of-information.
5. **God-agent / Evaluator.** Cross-session strategic oversight. The cognition layer is the substrate for this.

None of these are blocked on v2 finishing perfectly. They're blocked on v2 *shipping* and producing data.
