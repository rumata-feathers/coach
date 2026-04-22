-- Migration 003: Add retry_count and fallback_reason to agent_calls.
--
-- Additive-only: existing rows get retry_count=0 and fallback_reason=NULL.
-- No data loss, no renames.

ALTER TABLE agent_calls
    ADD COLUMN IF NOT EXISTS retry_count    INT  NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS fallback_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_agent_calls_fallback
    ON agent_calls(agent_name, fallback_reason)
    WHERE fallback_reason IS NOT NULL;
