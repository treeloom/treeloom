-- Job grouping ("Jobs" in the operator UI). A job_group is a single submission
-- of 1..N index tasks; each row in the existing `jobs` table is one "Task" and
-- now links to its parent group via `jobs.group_id`.
--
-- NAMING CAVEAT (deliberate): the physical `jobs` table is NOT renamed -- it is
-- deeply wired into the queue/runners/crash-recovery/webhooks. In the new
-- API/UI a `jobs` row is called a "Task" and a `job_groups` row is a "Job".
-- Code term != UI term.
--
-- timestamps stay DOUBLE PRECISION (unix epoch seconds) to match jobs.start_time
-- and the Job dataclass (see 004_jobs.sql).

CREATE TABLE IF NOT EXISTS job_groups (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- fleet | repo | directory | file | webhook | graph
    created_at  DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_by  TEXT,                       -- user id; nullable for legacy/anon
    task_count  INTEGER NOT NULL DEFAULT 0
);

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS group_id TEXT REFERENCES job_groups(id);
CREATE INDEX IF NOT EXISTS jobs_group_id_idx ON jobs (group_id);
CREATE INDEX IF NOT EXISTS job_groups_created_at_idx ON job_groups (created_at DESC);

-- Backfill: give every pre-existing job a synthetic group-of-1 so no history is
-- orphaned and the "every job has a group_id" invariant holds. Idempotent:
-- the group id is derived from the job id, INSERT is ON CONFLICT DO NOTHING, and
-- the UPDATE only touches still-unlinked rows. Re-running is a no-op.
INSERT INTO job_groups (id, label, kind, created_at, created_by, task_count)
SELECT
    'grp_' || j.id,
    COALESCE(
        NULLIF(j.source, ''),
        NULLIF(j.source_path, ''),
        NULLIF(j.source_url, ''),
        NULLIF(j.source_id, ''),
        j.id
    ),
    COALESCE(NULLIF(j.kind, ''), 'repo'),
    COALESCE(j.start_time, 0),
    NULL,
    1
FROM jobs j
WHERE j.group_id IS NULL
ON CONFLICT (id) DO NOTHING;

UPDATE jobs SET group_id = 'grp_' || id WHERE group_id IS NULL;
