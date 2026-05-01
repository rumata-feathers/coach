-- Migration 005: deployment version stamp on every turn.
--
-- Adds a TEXT column so each turn carries the code/prompt version that
-- produced it. Populated at write time from Settings.deployment_version
-- (e.g. "v0.5.1@a1b2c3d" on Railway, "local" in development).

ALTER TABLE turns ADD COLUMN IF NOT EXISTS deployment_version TEXT;

CREATE INDEX IF NOT EXISTS idx_turns_version ON turns(deployment_version);
