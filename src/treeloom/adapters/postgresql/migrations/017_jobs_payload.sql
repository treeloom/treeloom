-- Add payload (JSONB) and attempts (INT) columns to the jobs table.
--
-- payload: stores job-specific data that must survive a worker restart,
--          e.g. changed_files for incremental (webhook) jobs.
--
-- attempts: counts how many times a job has been attempted; used by the
--           retry/dead-letter logic to cap retries at
--           MAX_JOB_ATTEMPTS before moving the job to DEAD_LETTER status.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS payload JSONB;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;
