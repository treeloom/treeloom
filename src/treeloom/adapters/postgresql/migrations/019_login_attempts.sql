-- Login attempt audit for rate-limiting POST /auth/login.
--
-- A sliding-window counter of failed/succeeded login attempts keyed by
-- (username, client_ip). The throttle queries `recent_failures` over a
-- time window; on success the row set for that (username, client_ip) is
-- cleared so a legitimate user who eventually types the right password is
-- not penalized by their earlier typos.
--
-- This is intentionally a separate table (not folded into search_audit):
-- it is written on EVERY login attempt and read on the hot login path, so
-- it gets its own narrow composite index.
CREATE TABLE IF NOT EXISTS login_attempts (
    username     TEXT NOT NULL,
    client_ip    TEXT NOT NULL,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    succeeded    BOOLEAN NOT NULL DEFAULT FALSE
);

-- Supports the `recent_failures` window query (WHERE username = $1 AND
-- client_ip = $2 AND succeeded = FALSE AND attempted_at >= $3) and the
-- `clear` delete.
CREATE INDEX IF NOT EXISTS login_attempts_lookup_idx
    ON login_attempts (username, client_ip, attempted_at);
