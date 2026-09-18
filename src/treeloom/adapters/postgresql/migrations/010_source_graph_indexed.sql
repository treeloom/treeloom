-- Track per-source graph-indexing state so chunks can be indexed in
-- one pass and graph extraction deferred to a separate `kind=graph` job.
--
-- Default TRUE so existing rows look already-graphed (preserves
-- behavior for installs that have only ever run the all-in-one path).
-- New chunks-only jobs flip to FALSE on completion; the graph-only
-- job flips back to TRUE.
ALTER TABLE source_records
    ADD COLUMN IF NOT EXISTS graph_indexed    BOOLEAN     NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS graph_indexed_at TIMESTAMPTZ NULL;
