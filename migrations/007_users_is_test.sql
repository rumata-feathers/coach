-- migration 007: add is_test flag to users
-- ============================================================
-- Test-created users (from integration/quality tests and dev
-- eval scripts) land in the same DB as real users, making
-- daily reports noisy. This column lets the report filter them
-- out by default without losing the data.
--
-- Default is FALSE so the API-created real users need no change.
-- Tests and eval scripts set is_test = TRUE at INSERT time.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_users_is_test
    ON users(is_test)
    WHERE is_test = FALSE;
