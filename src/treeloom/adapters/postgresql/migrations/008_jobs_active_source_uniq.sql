-- Belt-and-suspenders dedup at the DB level: at most one queued/running
-- job per source_id at any given time. The application layer also checks
-- via `find_active_for_source`, but a partial unique index closes the
-- microsecond race window between the check and the INSERT.
--
-- Cleanup first: any pre-existing duplicates (queued/running rows sharing
-- a source_id) get collapsed — keep the oldest by start_time, mark the
-- rest as failed-by-supersede, and drop their queue rows.

BEGIN;

-- Pre-stage the ids to supersede so the DELETE and UPDATE see the same
-- snapshot. Excludes source_id = '' (legacy / non-source jobs).
CREATE TEMPORARY TABLE _superseded_jobs ON COMMIT DROP AS
SELECT id
  FROM (
      SELECT id,
             row_number() OVER (
                 PARTITION BY source_id
                 ORDER BY start_time NULLS LAST, id
             ) AS rn
        FROM jobs
       WHERE status IN ('queued', 'running')
         AND source_id <> ''
  ) ranked
 WHERE rn > 1;

-- Remove their queue entries (workers won't try to pick them up).
DELETE FROM job_queue
 WHERE job_id IN (SELECT id FROM _superseded_jobs);

-- Mark them failed so /jobs and the dashboard show the resolution.
UPDATE jobs
   SET status = 'failed',
       error = 'Superseded by earlier in-flight job for same source (migration 008)',
       finished_at = extract(epoch from now())
 WHERE id IN (SELECT id FROM _superseded_jobs);

-- Now enforce the invariant going forward. Excludes source_id = ''
-- so legacy/non-source jobs without a source_id don't conflict with
-- each other (matches the cleanup scope and the application-level
-- `find_active_for_source` short-circuit).
CREATE UNIQUE INDEX IF NOT EXISTS jobs_active_source_uniq
    ON jobs (source_id)
 WHERE status IN ('queued', 'running')
   AND source_id <> '';

COMMIT;
