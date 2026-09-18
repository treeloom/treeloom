-- Job pickup queue. Holds only job_ids — actual job state lives in the
-- jobs table. Workers pop with SELECT ... FOR UPDATE SKIP LOCKED so
-- multiple indexer processes can pull from the same queue without
-- double-running a job.
--
-- ON DELETE CASCADE on the FK means deleting a job row drops its queue
-- entry too. We rely on workers DELETEing the row after they finish, so
-- the queue should stay small (only pending work).
CREATE TABLE IF NOT EXISTS job_queue (
    job_id TEXT PRIMARY KEY REFERENCES jobs (id) ON DELETE CASCADE,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS job_queue_enqueued_at_idx ON job_queue (enqueued_at);
