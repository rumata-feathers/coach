# v0.5 Handoff — How to resume with Claude Code

## Context you need to paste first

Open Claude Code in the existing `career_coach` repo. Paste these files:

- `SPEC_v0.5.md`  (the new architectural additions)
- `TASKS_v0.5.md` (the ordered 6-task plan)
- `SPEC.md`       (unchanged v0 contract, still the foundation)
- `CLAUDE.md`     (unchanged ground rules)

Then paste this as the first message:

---

> v0 is done and deployed. Real conversations with the backend surfaced four classes of defects I need to fix before any v1 work. The full findings and the consolidation plan are in `SPEC_v0.5.md`. The ordered build plan is in `TASKS_v0.5.md`. The original v0 architectural contract is still in `SPEC.md` — v0.5 is additions and fixes, not a rewrite.
>
> Before you start:
>
> 1. Read `SPEC_v0.5.md` fully, then `TASKS_v0.5.md`. Confirm back in 5-8 bullets:
>    - The four defect classes we're fixing (§1 of v0.5 spec).
>    - Why Task 1 (reliability) has to come first.
>    - How onboarding differs from the standard flow (including when it's bypassed).
>    - What "done" means for v0.5 (the 6 criteria in §9).
> 2. List anything you want me to resolve before starting Task 1. Candidates include:
>    - Unit-test strategy for the retry-once pattern (how do we simulate an empty LLM response cleanly without overfitting the test to the implementation?)
>    - Whether migration 003 should be wrapped in a transaction with a rollback script.
>    - Whether the onboarding probe questions should be hardcoded in the template or held in a small YAML side-file for easier iteration.
>    - Whether the judge LLM in the distillation eval should be a DIFFERENT model tier from the distillation LLM itself (to avoid self-grading).
> 3. Only after I confirm, begin Task 1 ("Reliability fixes") per `TASKS_v0.5.md`.
>
> Commit after every task with the exact commit message suggested. Do not combine tasks. Do not skip ahead.

---

## What to expect from Claude Code

Claude Code will come back with the summary and its questions. The questions above are the ones I *expect* it to raise. Answer them directly — don't defer. In particular my strong opinions:

- **Onboarding probes in the Jinja template.** Keep them there. Moving them to YAML adds a layer of indirection we don't need yet. If we want A/B testing, we'll add it in v1.
- **Judge LLM for distillation eval: same model tier is fine.** Self-grading is a legitimate concern but Qwen3-235B is strong enough that we're more worried about false rejections (perfectionism) than false passes. Revisit if the scores come in suspiciously high.
- **Migration transactions.** SPEC_v0.5 §8 migration is additive-only (`ADD COLUMN IF NOT EXISTS`) so rollback is optional. Skip it for v0.5.
- **Retry test strategy.** Use `MockLLMClient.queue("", '<valid json>')` — queue returns empty first, valid second. Assert `len(mock.calls) == 2` after a single agent run. That's the whole test.

## Task cadence

- Task 1: reliability (one session, should be fast — no new features).
- Task 2: migration (30 min — trivial).
- Task 3: onboarding (biggest user-visible change — spend time on the probe prompt).
- Task 4: session continuity (one session — mostly tests and a demo script).
- Task 5: distillation quality (longest — 10 fixtures + eval script + dedupe).
- Task 6: memory loop closure (one session — ties it together).

Budget: roughly a week of focused time across Tasks 1-6 if you're doing one task per sitting.

## When to stop and call me back

Same cues as v0, plus:

- **Task 1:** if after the retry-once fix, the fallback rate in a quick test is still > 15% — something deeper is wrong, probably model or config. Stop.
- **Task 3:** if the onboarding battery scores under 4/5 after two prompt iterations — the probe archetypes may be wrong for your users. Stop and rethink.
- **Task 5:** if the distillation eval averages under 0.5 (well below the 0.7 bar) — distillation quality is fundamentally weaker than we assumed. Stop; this is a scope-changing finding.
- **Task 6:** if the Critic battery drops below 8/10 after the Coach prompt tweak, AND you can't recover it by softening the rule — back out the change, keep v0.5 without it, proceed to the v1 planning. Don't spend more than an afternoon fighting it.

## A note on how v0 data informs v0.5

The Profiler investigation before v0.5 was valuable — it proved extraction works. But the *same* investigation style is valuable during v0.5. Before Task 5, actually go have a 10-turn conversation and inspect `hypothesis_evidence` to check what hints the Profiler generates on a real arc. You may find that your fixture evidence (hand-crafted) differs from real-evidence (model-generated) in ways that bias the eval. Sanity-check before running the 10-fixture eval.

## After v0.5

Read the bottom of `TASKS_v0.5.md`. The short version: don't start v1 without a week of real usage on the v0.5 tag first. The v1 scope will depend on what the week surfaces.
