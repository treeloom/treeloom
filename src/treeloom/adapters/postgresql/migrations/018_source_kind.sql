-- Record how a source was indexed (repo | directory | file) so the fleet
-- auto-refresh loop can re-enqueue the SAME kind of job instead of
-- blindly re-indexing everything as a `repo` job. Re-indexing a file-indexed
-- source as a repo job fails the `os.path.isdir` check; re-indexing a
-- directory-indexed source as a repo job silently changes its file set.
--
-- Default 'repo' so existing rows (and the common case) keep working; new
-- index jobs persist their actual kind on completion. Legacy rows are
-- additionally shape-checked at refresh time (a file path → a file job).
ALTER TABLE source_records
    ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'repo';
