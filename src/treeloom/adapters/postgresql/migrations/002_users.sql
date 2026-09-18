-- Migration 002: users
-- Stores API-key-authenticated users with role-based access.

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    email         TEXT NOT NULL DEFAULT '',
    api_key_hash  TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users (api_key_hash);
CREATE INDEX IF NOT EXISTS idx_users_active ON users (active) WHERE active = TRUE;
