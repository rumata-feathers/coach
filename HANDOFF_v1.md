# v1 Handoff — How to resume with Claude Code

## Precondition checklist

Before any pasting, confirm:

1. ☐ `v0.5` tag is cut and all six v0.5 DoD criteria verified ON A CLEAN DB.
2. ☐ You have used the deployed v0.5 system for at least one full week.
3. ☐ You have run the two debrief queries from `TASKS_v0_5.md` §"After v0.5" and read them.
4. ☐ You have re-read `SPEC_v1.md` §1 and §2 with that data in hand and confirmed scope still matches.
5. ☐ You have a Tavily API key (free tier is fine for v1) — sign up at tavily.com.

If any of these isn't true, stop and complete it first.

---

## Step-by-step: what to put in Claude Code


### Step 2 — Open Claude Code in the repo

```bash
cd /path/to/career_coach
claude
```

### Step 3 — Show Claude Code the new files

Type into Claude Code (it can read repo files itself, but be explicit):

```
Please read these files in this order before responding:
  1. SPEC.md
  2. SPEC_v0_5.md
  3. SPEC_v1.md
  4. TASKS_v1.md
  5. CLAUDE.md
  6. HANDOFF_v1.md
```

Wait for it to confirm it's read them.

### Step 4 — Paste the kickoff message

Copy this verbatim and paste into Claude Code:

---

> v0 and v0.5 are done and deployed. I've used v0.5 for a week. The full v1 scope is in `SPEC_v1.md`. The ordered build plan is in `TASKS_v1.md`. v0 and v0.5 contracts are still in force — v1 is additions, not a rewrite.
>
> **Major scope notes for v1:**
>
> - v1 = Research Coach. The headline capabilities are real web search with citations, structured chart output, and a multi-agent deliberation flow (Flow C).
> - Challenges/projects and proactive check-ins are deferred to v1.5 — do NOT build them now even though `proposed_challenge` exists in legacy schemas.
> - Web search uses Tavily by default via a new provider-agnostic adapter (mirrors the LLM adapter pattern).
> - Charts are emitted as structured `ChartSpec` objects; rendering is v2 (frontend).
>
> **Week of v0.5 data:**
> unfortunately i do not have a week full of data
> - Flow distribution: [FILL IN: X% A, Y% B, Z% onboarding]
> - Fallback rate: [FILL IN: X% of agent_calls with fallback_reason set]
> - p50/p95 latency by agent: [FILL IN]
> - Top recurring failure modes I caught by eye that the system shipped anyway: [FILL IN]
> - Most common Critic rejection reason: [FILL IN]
>
> **Before you start anything:**
>
> 1. Read `SPEC_v1.md` and `TASKS_v1.md` end-to-end. Confirm back in 8-12 bullets:
>    - The seven pillars of v1 (§1).
>    - What v1 does NOT do, split between v1.5 deferrals (§2) and v2+ deferrals (§3).
>    - Why web search uses an adapter pattern (§4.1).
>    - When Researcher uses KB-first vs web-first (§5.2).
>    - The citation hygiene rules (§5.5).
>    - When charts are emitted and when they are not (§6.4).
>    - Why Supervisor is distinct from Critic (§10.1).
>    - Why the LangGraph migration is Task 9, not Task 1.
>    - What "v1 done" means (§15, all 12 criteria).
>
> 2. List anything you want me to resolve before starting Task 0. Candidates:
>    - The four v0.5 bug fixes in Task 0 — confirm you understand the `extra_body` shape correction for Qwen3 specifically.
>    - Web search provider choice: I want Tavily as default (free tier sufficient for v1 eval). Confirm.
>    - KB authoring split: I write 5, you draft 10, I review every `source_notes` field before commit. Confirm.
>    - Tag extraction in Researcher planner: keyword matching against KB tags, not an LLM call (cost discipline). Confirm.
>    - Flow C timeout degradation: return Coach output, log fallback_reason. Confirm.
>    - Streaming responses: NO in v1, sync only. Confirm.
>    - LangGraph version pinning: pin to current stable, don't upgrade during v1. Confirm.
>
> 3. Only after I confirm your answers, begin Task 0 ("v0.5.1 patch") per `TASKS_v1.md`.
>
> Commit after every task with the exact commit message from the task. Do not combine tasks. Do not skip ahead. If a task's DoD fails, STOP — don't press on.

---

### Step 5 — Answer Claude Code's questions

It will come back with the summary plus its questions. My defaults for the predictable ones:

- **`extra_body` shape for Qwen3.** It's `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}` (NOT the OpenAI-style `thinking: false` that was in the v0.5 spec). Have it verify against current HF docs first; the API shape may have evolved.
- **Tavily.** Yes. If Claude Code asks about Exa or alternatives, say "Tavily for v1, adapter pattern lets us swap in v2 if needed."
- **KB authoring split.** Confirm yes. The 5 you write: quant_finance, investment_banking, software_engineering, academia_stem_phd_track, medicine_uk. Claude Code drafts the other 10 and you review every `source_notes` before commit.
- **Researcher planner tag extraction.** Keyword matching is fine; KB is small. If recall feels bad after Task 4 eval, escalate then.
- **Flow C timeout.** Coach output as fallback, `fallback_reason="flow_c_timeout"`.
- **No streaming.** Confirm. v2.
- **LangGraph version.** Pin and don't upgrade during v1.

### Step 6 — Let it execute Task 0

Task 0 is the v0.5.1 patch. Should be one short session. After commit, run the demo script manually and verify no `<think>` blocks leak into agent_calls payloads.

### Step 7 — Continue task by task

Each task is its own Claude Code session. Cadence:

- **Task 0:** quick patch, ~30 min.
- **Task 1:** migration, ~30 min.
- **Task 2:** web search adapter, one full sitting. The mock client matters — agents in Tasks 4/7/8 will use it heavily.
- **Task 3:** KB. SLOW DOWN. Curation work, not coding. Don't accept any numeric pay band without `source_notes`.
- **Task 4:** Researcher. Spend time on the prompt with real outputs in the loop. This is where citation hygiene is enforced.
- **Tasks 5-6:** DA + Synthesizer. One sitting each. Mostly contracts and prompts.
- **Task 7:** Flow C wiring. Iterate on Synthesizer prompt once you can see real outputs.
- **Task 8:** Supervisor. Narrow scope, fast.
- **Task 9:** LangGraph. Plan a full weekend. The "no test changes" rule keeps the refactor honest.
- **Task 10:** Quality eval. Mostly fixture authoring + script.

Total budget: 3-4 weeks of focused evenings.

---

## When to stop and call yourself back

Same cues as v0/v0.5, plus:

- **Task 0:** if after the `extra_body` fix the fallback rate doesn't drop noticeably in a quick test, the Qwen3 think-block issue may have a different root cause. Stop and investigate before piling Task 1+ on top.
- **Task 2:** if Tavily integration test consistently hits rate limits even within the free tier, switch to Exa or self-host a Brave Search proxy. Don't ship a flaky adapter.
- **Task 3:** if you catch yourself writing pay bands without source_notes, STOP. The KB's whole value is its honesty about sources.
- **Task 4:** if Researcher quality battery scores below 15/20 after two prompt iterations, the citation hygiene rules may be conflicting with the synthesis ability. Stop and consider relaxing the "fetch must contain claim verbatim" rule.
- **Task 7:** if Flow C Critic pass rate on its 5-fixture battery is under 3/5 after two Synthesizer prompt iterations, the integration pattern is wrong. Probably means Synthesizer needs a different output structure (e.g. explicit "tradeoffs" section vs free-form integration).
- **Task 8:** if Supervisor produces >2/5 false positives after tuning, soften the prompts. False positives erode trust faster than missed catches.
- **Task 9:** if LangGraph migration requires ANY change to v0.5/v1 integration tests to keep them passing, STOP. Either roll back or diagnose what state is leaking that the hand-rolled version encapsulated.
- **Task 10:** if Flow C battery scores below 15/20 (well below the 17/20 bar), the deliberation loop isn't producing better output than Flow B. Stop, write up findings, reassess Flow C shape. May be that DA+Synth is overkill for students and a simpler "Coach + structured tradeoffs" works better — but that's a v2 planning decision.

---

## Cost watch

v1 evals will cost more than anything before. Rough estimate:

- Researcher quality battery (Task 4): ~£5-10 per full run.
- Flow C eval (Task 10): ~£15-25 per full run.
- Total dev-time API spend across v1 if you re-run evals on iteration: probably £100-200 cumulatively.

Set a daily spending cap before Task 4. If costs surprise you upward, audit:

- Is web search caching actually being hit?
- Is Researcher being called from inside Critic/Synthesizer by accident?
- Is Critic retry loop bounded at 2?
- Is Supervisor on Haiku-tier (not Coach-tier)?

---

## A note on v1 being a strong hypothesis

This v1 spec was drafted before two weeks of v0.5 usage. If real data contradicts the assumed shape:

- If Flow C conditions fire < 5% of turns, Flow C may be over-engineered for now.
- If users never ask research-shaped questions, web search may be a v2 concern.
- If Supervisor near-misses from v0.5 cluster on a different dimension (e.g. over-certainty rather than fact contradictions), the three Supervisor checks should be re-chosen.

Treat the spec as a hypothesis. Real data is what tests it. Be willing to rewrite §1 of `SPEC_v1.md` if the data says so — better to revise the plan than to ship the wrong v1.

---

## After v1

Two-week observation window. Read transcripts. THEN plan v1.5 — the Active Coach (challenges as first-class, Initiator agent, check-in endpoint).

v1.5 is intentionally smaller in infrastructure than v1; the hard part is the temporal logic (when to initiate, how to track project state across turns, how to interpret follow-up replies). Don't start v1.5 from scratch — it builds on the v1 surfaces.
