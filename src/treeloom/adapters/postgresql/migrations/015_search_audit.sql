-- Migration 015: who-searched-what audit trail
--
-- One row per authorization decision on a read endpoint. Answers
-- "what did user Y search?" and "who accessed repo X?" for security review.
-- Best-effort writes (the indexer swallows failures here, like job_file_errors)
-- so auditing never breaks the search path.

CREATE TABLE IF NOT EXISTS search_audit (
    id                BIGSERIAL PRIMARY KEY,
    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id           TEXT,                       -- NULL only in open mode
    principal_groups  TEXT[] NOT NULL DEFAULT '{}',
    action            TEXT NOT NULL,              -- search | graph_explore | find_*
    scope             TEXT NOT NULL,              -- 'source' | 'shared'
    source_id         TEXT,                       -- set for source-scoped queries
    decision          TEXT NOT NULL               -- 'allow' | 'deny'
);

CREATE INDEX IF NOT EXISTS search_audit_user_idx ON search_audit (user_id);
CREATE INDEX IF NOT EXISTS search_audit_source_idx ON search_audit (source_id);
CREATE INDEX IF NOT EXISTS search_audit_occurred_idx ON search_audit (occurred_at);
