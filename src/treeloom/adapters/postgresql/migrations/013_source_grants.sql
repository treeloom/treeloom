-- Migration 013: per-source access grants
--
-- A grant ties a principal (a user or a group) to a source with an allow/deny
-- effect. The authorization decision (domain/authorization/access.py) is:
--   deny wins > all_access > explicit allow > owner(created_by) > deny.
-- So a `deny` row carves an exclusion out of an otherwise all_access principal
-- (the CI-agent "all repos except these" case).
--
-- principal_type is 'user' or 'group'; principal_id is the users.id /
-- groups.id. We intentionally do NOT FK principal_id (it's polymorphic) — the
-- store validates the referent. source_id mirrors source_records.id (no FK so
-- a grant can be pre-created before a source is indexed, and survives reindex).

CREATE TABLE IF NOT EXISTS source_grants (
    principal_type  TEXT NOT NULL CHECK (principal_type IN ('user', 'group')),
    principal_id    TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    effect          TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_type, principal_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_source_grants_source_id ON source_grants (source_id);
CREATE INDEX IF NOT EXISTS idx_source_grants_principal
    ON source_grants (principal_type, principal_id);
