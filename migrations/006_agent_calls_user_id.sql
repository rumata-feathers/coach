-- migration 006: add user_id to agent_calls
-- ============================================================
-- agent_calls rows created before _node_persist have turn_id = NULL
-- (the turn row doesn't exist yet when agents run). Adding user_id
-- lets us aggregate per-user token usage from agent_calls directly
-- without needing a turn FK join.
--
-- Existing rows get NULL (no back-fill; old data is pre-schema).
-- The column is nullable so agents that have no user context (e.g.
-- standalone test calls) can still insert without errors.

ALTER TABLE agent_calls
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);

CREATE INDEX IF NOT EXISTS idx_agent_calls_user
    ON agent_calls(user_id, created_at DESC);
