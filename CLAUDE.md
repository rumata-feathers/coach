# Instructions for Claude Code

This file is for you, Claude Code. Read it before you start.

## Ground rules

1. **Read `SPEC.md` first and refer back to it constantly.** It is the contract. If you find yourself drifting from it, stop and check.

2. **Build in the order given in `TASKS.md`.** Do not skip ahead. Each task produces something runnable. Prove each works before moving on.

3. **Model IDs are not trusted.** Before writing any LLM call code, use web search to verify current Anthropic model IDs (e.g. Claude Sonnet 4.x, Claude Haiku 4.x). The spec mentions example IDs but they may be stale.

4. **Ask me before making architectural choices the spec does not resolve.** Examples: which embedding model to use, exact pgvector dimension, whether to use `asyncpg` vs Supabase Python client, whether to use Alembic for migrations. Propose options, let me pick.

5. **Prompts live in `config/prompts/*.j2`, never hardcoded in agent code.** Agent code loads and renders them. This is non-negotiable — we will iterate on prompts a lot.

6. **Every agent has typed Pydantic input and output.** No dicts at agent boundaries. Contract-first.

7. **Write tests as you go, not at the end.** Especially the Critic test battery — that's our quality canary.

8. **Commit after every task completes.** Descriptive commit messages. I want to be able to bisect.

## What to do when you hit ambiguity

Don't guess. Ask. Examples of things worth stopping on:

- "The spec says X but it conflicts with Y — which wins?"
- "Should this field be nullable?"
- "Supabase has feature Z that would make this easier — worth using?"
- "This is going to take a detour to do properly — okay to take it, or hack it for v0?"

Better to ask five questions up front than write 500 lines of code I'll reject.

## What NOT to build

Check `SPEC.md` section 14 ("anti-goals"). If you catch yourself about to write any of those — stop.

## Quality expectations

- Type hints everywhere. `mypy --strict` should pass on `src/`.
- Docstrings on all public classes and functions.
- `ruff` clean.
- No unused imports, no commented-out code.
- Every `TODO` must have a linked reason — "TODO: handle Unicode (defer to v1, see spec §14)".

## A note on the LLM adapter

Hugging Face is a requested alternate backend. Do it properly: the HF client should use the OpenAI-compatible Inference API endpoint so it can target any chat-tuned open model. This means the adapter is effectively an OpenAI-protocol client pointing at HF URLs. Keep that isolation clean — swapping providers must not require agent changes.

## When in doubt

The point of v0 is to make the architecture *feel real*. It's okay if it's slow. It's okay if it costs more than it should. It is NOT okay if:
- The memory tiers are collapsed
- The Critic is skipped
- Agents are not proper modules
- Prompts are hardcoded in Python

Those four things are the whole reason we're doing this instead of a ChatGPT wrapper.
