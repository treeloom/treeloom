-- Registry of embedding TEI backends. Replaces the .env-only config
-- (EMBEDDING_URLS / EMBEDDING_FALLBACK_URLS) so operators can add and
-- remove backends without restarting workers.
--
-- Workers LISTEN on `embedding_backends_changed`; mutating endpoints
-- emit `NOTIFY embedding_backends_changed` in the same transaction as
-- the row write. On notify, EmbeddingProxy.reload() rebuilds its
-- backend dict, carrying over per-URL in_flight_tokens/failures so
-- live requests keep their counters.
--
-- gen_random_uuid() comes from pgcrypto (enabled below if missing).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS embedding_backends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL UNIQUE,
    klass TEXT NOT NULL CHECK (klass IN ('gpu', 'cpu')),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID NULL
);

CREATE INDEX IF NOT EXISTS embedding_backends_enabled_idx
    ON embedding_backends (enabled);
