-- Migration 012: groups + group membership
--
-- Groups are a community-tier authorization primitive: a named principal that
-- can be granted access to sources (see 013_source_grants.sql) and that users
-- belong to. Membership is managed locally; an external identity provider maps
-- external group claims onto these same rows.
--
-- `all_access` marks a group whose members may query the shared/cross-repo
-- index (subject to explicit deny grants) — e.g. a `ci-agents` group that
-- sees every repo except a sensitive few.

CREATE TABLE IF NOT EXISTS groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    all_access  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id  TEXT NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    user_id   TEXT NOT NULL,
    PRIMARY KEY (group_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_group_members_user_id ON group_members (user_id);
