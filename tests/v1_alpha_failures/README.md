# v1 Alpha Failure Fixtures

Real production turns from the v1 alpha period that exhibit the four failure modes
identified in `SPEC_v2.md` §Context. Two fixtures per mode, each pulled directly
from the database with full conversation context.

## Purpose

v2 must fix all four failure modes. These fixtures are the ground truth — a v2
response to the same `user_message` and `user_facts` must **not** exhibit the
annotated failure. Tests in `tests/quality/test_v2_regression.py` (to be written
as part of v2 Phase 1) will load each fixture, replay the turn through the v2
pipeline, and assert the failure does not appear.

## Failure modes

### FM1 — Committee voice (`fm1_committee_voice/`)

The Synthesizer response reads as a transcript of agent deliberations rather than
one integrated view. Markers: third-person agent attribution ("The Coach suggests",
"the Devil's Advocate raises"), bracketed labels ("[Coach perspective]", "[DA
critique]"), and paragraph-level splits between named voices.

**v2 fix target:** Navigator produces a Directive; the Producer owns the single
voice. No agent name ever appears in the response text.

### FM2 — Premature advice (`fm2_premature_advice/`)

The system delivers specific recommendations before it has asked the questions
needed to make those recommendations non-generic. Pattern: turn_index ≤ 2 +
long response + named options. The surface keyword ("quant", "biochemistry")
drove the advice template; the user's actual constraint was never surfaced.

**v2 fix target:** Navigator routes to `elicit` move when the user model
confidence is below threshold. The Elicitor asks exactly one question. Directive
move to `recommend` is blocked until the UserModeler has sufficient signal.

### FM3 — Vague action items (`fm3_vague_action_items/`)

Action items are generic templates: "consider informational interviews",
"experiment with side projects", "explore internal rotational programmes".
These apply to any finance professional; they require zero knowledge of the
specific user to generate.

**v2 fix target:** The WorldModeler provides named employers, programmes, or
contacts drawn from Researcher findings. The Critic's `vague_action` failure
mode catches any action item that lacks a specific named target. If Research
cannot surface names, the Directive does not route to a `recommend` move.

### FM4 — Generic feel (`fm4_generic_feel/`)

The user's populated profile is not used in the response. The most visible form:
the hardcoded stall ("I want to give you a genuinely useful response here, but
I'm running into difficulty...") which uses zero user facts. Critic rejected
both fixture turns three consecutive times for `ungrounded`.

**v2 fix target:** The stall fallback is removed. Every response must reference
≥ 2 known user facts (Critic enforced). The Profiler's richer user model gives
the Navigator enough signal to always route to a grounded move.

## Fixture schema

Each `.json` file has:

```jsonc
{
  "failure_mode": "committee_voice | premature_advice | vague_action_items | generic_feel",
  "source": "v1_alpha_production",
  "turn_id": "<uuid from production DB>",
  "user_display_name": "<anonymised first name>",
  "turn_index": <int>,
  "flow_used": "B | C | ...",
  "user_message": "<exact text>",
  "assistant_message": "<exact v1 response that failed>",
  "critic_verdicts": [ ... ],   // present only for FM4 fixtures (already in DB)
  "annotations": {
    "problem": "<why this is a failure>",
    "evidence": ["<quoted substring 1>", ...],
    "v2_expectation": "<what the v2 response must do instead>"
  }
}
```

## How to add more fixtures

```bash
# Run the DB query for the failure mode you want
uv run python scripts/daily_report.py --since 168h   # see recent turns

# Or query directly:
uv run python -c "
import asyncio, asyncpg, os
async def q():
    conn = await asyncpg.connect(dsn=os.environ['SUPABASE_DB_URL'], ssl='require')
    rows = await conn.fetch('''
        SELECT turn_id, user_message, assistant_message
        FROM turns t JOIN users u ON u.user_id = t.user_id
        WHERE u.is_test = FALSE
          AND t.assistant_message ILIKE '%the Devil%'
        ORDER BY t.created_at DESC LIMIT 10
    ''')
    for r in rows: print(r['turn_id'], r['assistant_message'][:100])
asyncio.run(q())
"
```

Pick the clearest example, copy it into a new `fixture_NNN.json` in the relevant
subdirectory, and fill in the `annotations` block manually.
