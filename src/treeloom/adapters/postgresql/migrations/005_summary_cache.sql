-- Per-chunk LLM summary cache. Replaces ~/.treeloom/summary_cache.sqlite.
-- Shared across indexer workers so horizontal scale doesn't re-pay for the
-- same chunk summarizations.
--
-- Key is content-addressable: SHA1(chunk_text) + LLM_MODEL + PROMPT_VERSION.
-- Bump PROMPT_VERSION in llm_adapter.py to invalidate (key changes; old rows
-- stay but are never read again).
CREATE TABLE IF NOT EXISTS summary_cache (
    sha1 TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (sha1, model, prompt_version)
);
