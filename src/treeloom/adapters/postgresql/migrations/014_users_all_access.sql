-- Migration 014: per-user all_access flag
--
-- Mirrors groups.all_access at the user level. A user with all_access (or a
-- member of an all_access group) may run shared/cross-repo queries, minus any
-- source they have an explicit deny grant for. Admins (role='admin') bypass
-- all of this regardless of the flag.

ALTER TABLE users ADD COLUMN IF NOT EXISTS all_access BOOLEAN NOT NULL DEFAULT FALSE;
