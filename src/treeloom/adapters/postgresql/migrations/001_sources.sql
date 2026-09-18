-- Migration 001: sources
-- Stores metadata for every repository/directory indexed by treeloom.

CREATE TABLE IF NOT EXISTS source_records (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    branch      TEXT NOT NULL DEFAULT '',
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_count  INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_by  TEXT
);

CREATE INDEX IF NOT EXISTS idx_source_records_url_branch
    ON source_records (url, branch);
