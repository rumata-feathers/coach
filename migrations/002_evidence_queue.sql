-- Migration 002: Add hypothesis_hint to hypothesis_evidence for distillation.
--
-- The Profiler queues evidence rows with hypothesis_id = NULL (pending
-- distillation). hypothesis_hint stores the Profiler's free-form label
-- so the distillation job has a semantic seed when creating/matching hypotheses.

ALTER TABLE hypothesis_evidence
    ADD COLUMN IF NOT EXISTS hypothesis_hint TEXT;
