-- Per-file error log for indexing jobs. The `jobs` table tracks only
-- a count; this table records WHICH files failed and HOW, so operators
-- can investigate patterns (always TypeScript? always >1MB? always
-- specific paths?) and eventually retry just the failures.
--
-- Composite PK on (job_id, file_path) means a file's record is upserted
-- if it errors multiple times within the same job — useful when a worker
-- retries internally and we want to log the final outcome.
CREATE TABLE IF NOT EXISTS job_file_errors (
    job_id        TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    error_kind    TEXT NOT NULL,  -- 'timeout' | 'exception'
    error_message TEXT NOT NULL DEFAULT '',
    elapsed_s     DOUBLE PRECISION NULL,
    occurred_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, file_path)
);

CREATE INDEX IF NOT EXISTS job_file_errors_job_id_idx ON job_file_errors (job_id);
CREATE INDEX IF NOT EXISTS job_file_errors_error_kind_idx ON job_file_errors (error_kind);
