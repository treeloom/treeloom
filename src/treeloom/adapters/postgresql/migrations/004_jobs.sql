-- Indexer job state. Replaces the previous SQLite-backed jobs.db.
-- Job timestamps stay as DOUBLE PRECISION (unix epoch seconds) to match the
-- Job dataclass; converting to TIMESTAMPTZ would force a domain change.
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT '',
    source_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    kind TEXT NOT NULL DEFAULT 'repo',
    total_files INTEGER NOT NULL DEFAULT 0,
    processed_files INTEGER NOT NULL DEFAULT 0,
    committed_files INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    start_time DOUBLE PRECISION,
    finished_at DOUBLE PRECISION,
    error TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    current_file TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    source_branch TEXT NOT NULL DEFAULT '',
    commit_sha TEXT NOT NULL DEFAULT '',
    errors INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
CREATE INDEX IF NOT EXISTS jobs_source_id_idx ON jobs (source_id);
