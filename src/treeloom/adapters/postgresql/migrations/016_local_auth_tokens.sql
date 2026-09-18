-- Migration 016: local authentication tokens
--
-- Adds DB-backed sessions, personal access tokens (PATs), and service API
-- keys. Also extends the users table with a nullable password_hash column
-- for local login (null = token-only user).
--
-- Legacy migration: any existing users.api_key_hash values are backfilled
-- into personal_access_tokens so they continue to work through the new
-- token-resolution pipeline.

-- 1. Password hash on users (nullable — null = no local login)
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;

-- 2. Sessions (DB-backed opaque tokens, HttpOnly cookie)
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT        PRIMARY KEY,
    user_id      TEXT        NOT NULL REFERENCES users(id),
    token_hash   TEXT        NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions (token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id    ON sessions (user_id);

-- 3. Personal access tokens (user-owned, per-token scopes)
CREATE TABLE IF NOT EXISTS personal_access_tokens (
    id           TEXT        PRIMARY KEY,
    user_id      TEXT        NOT NULL REFERENCES users(id),
    name         TEXT        NOT NULL,
    token_hash   TEXT        NOT NULL UNIQUE,
    scopes       TEXT[]      NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    revoked      BOOLEAN     NOT NULL DEFAULT FALSE,
    UNIQUE (user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_pat_token_hash ON personal_access_tokens (token_hash);
CREATE INDEX IF NOT EXISTS idx_pat_user_id    ON personal_access_tokens (user_id);

-- 4. API keys (service/admin-owned, per-token scopes)
CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT        PRIMARY KEY,
    name         TEXT        NOT NULL UNIQUE,
    token_hash   TEXT        NOT NULL UNIQUE,
    scopes       TEXT[]      NOT NULL DEFAULT '{}',
    all_access   BOOLEAN     NOT NULL DEFAULT FALSE,
    created_by   TEXT        REFERENCES users(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    revoked      BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_api_keys_token_hash ON api_keys (token_hash);

-- 5. Legacy migration: backfill existing api_key_hash values into PATs.
--    Idempotent via ON CONFLICT (token_hash) DO NOTHING.
INSERT INTO personal_access_tokens (id, user_id, name, token_hash, scopes, created_at)
SELECT
    md5(random()::text || u.id),
    u.id,
    'legacy',
    u.api_key_hash,
    CASE
        WHEN u.role = 'admin' THEN ARRAY['search','index','admin']
        ELSE ARRAY['search','index']
    END,
    NOW()
FROM users u
WHERE u.api_key_hash <> ''
  AND u.api_key_hash IS NOT NULL
ON CONFLICT (token_hash) DO NOTHING;
