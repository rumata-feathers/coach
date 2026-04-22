# How to hand this off to Claude Code

## The first prompt to paste

Open Claude Code in an empty directory. Paste these files in first:
- `SPEC.md`
- `CLAUDE.md`
- `TASKS.md`
- `README.md`

Then paste this as your first message:

---

> I'm building a career coaching system. The full architectural contract is in `SPEC.md`. Instructions for you are in `CLAUDE.md`. The ordered build plan is in `TASKS.md`.
>
> Read all three files now, then:
>
> 1. Confirm back to me in 5-10 bullets what you understand about the architecture, specifically calling out: the three-tier memory model, the agent contract pattern, the Critic gate, and the provider-agnostic LLM layer.
> 2. List any ambiguities or decisions you want me to resolve before you start Task 1.
> 3. Then, and only then, begin Task 1 ("Project skeleton + tooling") per `TASKS.md`.
>
> Do not skip steps 1 and 2. Do not start Task 1 until I confirm.

---

## What to expect

Claude Code will come back with its summary and questions. Expect questions like:
- "uv or Poetry?"
- "Alembic or plain SQL migrations?"
- "asyncpg or the Supabase Python client?"
- "Which embedding model for pgvector — OpenAI's `text-embedding-3-small`, Anthropic's upcoming embeddings, or an HF model?"

Answer them. Then let it go on Task 1.

## Cadence I recommend

- One task per Claude Code session, committed at the end
- After each task: run what it built, verify the "Done when" criterion, then move to the next
- At Task 7 (Orchestrator + Coach + Critic), slow down — this is where the architecture either works or doesn't. Spend time on prompts. Iterate on the Critic test battery.
- After Task 12, stop and use the system for a week before deciding on v1 scope

## When to stop and redesign

Bail out of the current plan and call me back if any of these happen:

- LangGraph becomes a fight (the v0 spec doesn't actually require LangGraph yet — hand-rolled asyncio is fine for the skeleton; defer LangGraph until the agent graph gets more complex in v1)
- The three memory tiers feel awkward in practice (they shouldn't — if they do, something's wrong in the schema or repos)
- Critic rejection rate is >60% or <5% in the test battery — both are symptoms, not features
- You notice yourself hardcoding prompts into agent `.py` files — this is the biggest failure mode, catch it early

## One more thing

I said "Python (FastAPI + LangGraph)" in the stack decision, but for v0 you do not actually need LangGraph yet. The v0 turn loop is ~50 lines of asyncio. LangGraph's value shows up at v1, when you have Researcher + Devil's Advocate + Synthesizer running in parallel with retries and branching. For v0, plain async is simpler and easier to debug. `TASKS.md` reflects this — LangGraph is not in the v0 dependencies. Add it at v1.
