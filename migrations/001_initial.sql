-- ============================================================================
-- Career Coach v0 — initial schema.
-- See SPEC.md §5 for the architectural contract.
--
-- Three cognitive tiers, never collapsed:
--   1. Structured facts  — stable, keyed attributes (age, location, ...)
--   2. Episodic memory   — sessions + turns + pgvector embeddings
--   3. Semantic memory   — hypotheses + append-only evidence rows
-- Plus an agent_calls log for observability.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- for gen_random_uuid()

-- -------------------------------------------------------------------------
-- Users (v0: single hardcoded test user, auth deferred)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    display_name  TEXT
);

-- -------------------------------------------------------------------------
-- Tier 1: Structured facts
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS structured_facts (
    fact_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key         TEXT NOT NULL,               -- e.g. "age", "location", "education_stage"
    value       JSONB NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    source      TEXT,                        -- "user_stated" | "inferred" | "system"
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON structured_facts(user_id);

-- -------------------------------------------------------------------------
-- Tier 2: Episodic memory
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    session_theory  TEXT                     -- Understander's working theory
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, started_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    turn_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id             UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    turn_index          INT NOT NULL,
    user_message        TEXT,                -- nullable: some short-circuit turns have no Coach output
    assistant_message   TEXT,
    intent_packet       JSONB,               -- output of Understander
    flow_used           TEXT,                -- "A" | "B" | "C"
    critic_verdicts     JSONB,               -- array of verdicts across retries
    tokens_used         JSONB,               -- {"understander": 123, "coach": 456, ...}
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_user    ON turns(user_id, created_at DESC);

-- Embeddings live in a sidecar table so turns are cheap to read when we
-- don't need similarity search. Dimension must match EMBEDDING_DIM in .env.
CREATE TABLE IF NOT EXISTS turn_embeddings (
    turn_id    UUID PRIMARY KEY REFERENCES turns(turn_id) ON DELETE CASCADE,
    embedding  vector(384)                   -- sentence-transformers/all-MiniLM-L6-v2
);

-- ivfflat needs ANALYZE + tuning once we have data; fine for v0.
CREATE INDEX IF NOT EXISTS idx_turn_embeddings_cos
    ON turn_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- -------------------------------------------------------------------------
-- Tier 3: Semantic / hypotheses (append-only with evidence)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    statement        TEXT NOT NULL,
    confidence       REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active', 'dormant', 'retired', 'superseded')),
    open_questions   JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_updated     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_reviewed    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_hyp_user ON hypotheses(user_id, status);

CREATE TABLE IF NOT EXISTS hypothesis_evidence (
    evidence_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id    UUID REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE,
    turn_id          UUID REFERENCES turns(turn_id) ON DELETE SET NULL,
    source_type      TEXT,                   -- "user_statement" | "reflection" | "challenge_outcome" | "forced_choice"
    excerpt          TEXT,
    weight           REAL NOT NULL,          -- positive supports, negative contradicts
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_hyp  ON hypothesis_evidence(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_evidence_turn ON hypothesis_evidence(turn_id);

-- -------------------------------------------------------------------------
-- Agent call log (observability)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_calls (
    call_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id          UUID REFERENCES turns(turn_id) ON DELETE SET NULL,
    agent_name       TEXT NOT NULL,
    model_used       TEXT,
    input_payload    JSONB,
    output_payload   JSONB,
    latency_ms       INT,
    tokens_in        INT,
    tokens_out       INT,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_calls_turn ON agent_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_agent_calls_name ON agent_calls(agent_name, created_at DESC);
