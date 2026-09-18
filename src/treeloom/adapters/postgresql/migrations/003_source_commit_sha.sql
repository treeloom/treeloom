-- Migration 003: track git HEAD SHA at index time
-- Lets callers detect when an indexed source is out of date relative to
-- the current HEAD of the underlying repository. Empty for non-git inputs.

ALTER TABLE source_records
    ADD COLUMN IF NOT EXISTS commit_sha TEXT NOT NULL DEFAULT '';
