-- Migration 004 — v1 agent schema additions
-- Adds: research_briefs table, supervisor_events table,
--       and new columns on turns (research_brief_id, synthesizer_output,
--       devils_advocate_output, chart_specs).
--
-- Additive only. Existing rows are unaffected.
-- Applied on top of 001_initial.sql + 002_evidence_queue.sql + 003_observability.sql.
--
-- See SPEC_v1.md §12 for the canonical definition.

-- ── turns additions ──────────────────────────────────────────────────────────

ALTER TABLE turns
    ADD COLUMN IF NOT EXISTS research_brief_id UUID,
    ADD COLUMN IF NOT EXISTS synthesizer_output JSONB,
    ADD COLUMN IF NOT EXISTS devils_advocate_output JSONB,
    ADD COLUMN IF NOT EXISTS chart_specs JSONB DEFAULT '[]'::jsonb;

-- ── research_briefs ──────────────────────────────────────────────────────────
-- One row per Researcher invocation. Stored for every turn that triggered
-- Flow C, including turns where findings came back empty (graceful failure
-- signal). Linked to turns via turns.research_brief_id.

CREATE TABLE IF NOT EXISTS research_briefs (
    brief_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id          UUID        REFERENCES turns(turn_id),
    question         TEXT        NOT NULL,
    findings         JSONB       NOT NULL,
    caveats          JSONB       DEFAULT '[]'::jsonb,
    next_questions   JSONB       DEFAULT '[]'::jsonb,
    used_kb_files    JSONB       DEFAULT '[]'::jsonb,
    web_searches_run JSONB       DEFAULT '[]'::jsonb,
    web_sources_consulted JSONB  DEFAULT '[]'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_briefs_turn ON research_briefs(turn_id);

-- ── supervisor_events ────────────────────────────────────────────────────────
-- One row per non-pass Supervisor event. Pass events are not persisted
-- (would add noise without signal). See SPEC_v1.md §10.

CREATE TABLE IF NOT EXISTS supervisor_events (
    event_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id      UUID        REFERENCES turns(turn_id),
    event_type   TEXT        NOT NULL,   -- fact_contradiction | off_topic | unsafe
    severity     TEXT        NOT NULL,   -- low | med | high
    details      JSONB,
    action_taken TEXT        NOT NULL,   -- pass | warn | retry | block
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_supervisor_events_turn
    ON supervisor_events(turn_id);

CREATE INDEX IF NOT EXISTS idx_supervisor_events_type
    ON supervisor_events(event_type, created_at DESC);
