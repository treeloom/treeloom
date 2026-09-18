-- Preflight throughput-model coefficients. Each row is one tunable
-- constant; rows with language='' are global, others override per-
-- language. Updated via `python -m treeloom.preflight --calibrate
-- <source_id>` after a successful indexing run.
--
-- The model and the rule set live in `src/treeloom/preflight/`.
CREATE TABLE IF NOT EXISTS preflight_coefficients (
    metric_key TEXT NOT NULL,
    language   TEXT NOT NULL DEFAULT '',
    value      DOUBLE PRECISION NOT NULL,
    calibrated_from_source_id TEXT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (metric_key, language)
);

-- Seed defaults — calibrated roughly from the airbnb_javascript bench
-- run (32 files, 23s) and TypeScript observations. Operators with
-- different LLM/embedder setups should run --calibrate.
INSERT INTO preflight_coefficients (metric_key, language, value) VALUES
    ('parse_s_per_mb',         '',           1.0),
    ('embed_s_per_chunk',      '',           0.04),
    ('llm_s_per_chunk',        '',           2.0),
    ('milvus_s_per_file',      '',           0.05),
    ('graph_s_per_file_const', '',           0.3),
    ('graph_s_per_entity',     '',           0.005),
    ('neo4j_warmup_s',         '',           5.0),
    -- per-language entity density (entities per LOC)
    ('entities_per_loc',       'typescript', 0.50),
    ('entities_per_loc',       'tsx',        0.50),
    ('entities_per_loc',       'javascript', 0.30),
    ('entities_per_loc',       'jsx',        0.30),
    ('entities_per_loc',       'python',     0.20),
    ('entities_per_loc',       'rust',       0.25),
    ('entities_per_loc',       'go',         0.25),
    ('entities_per_loc',       'java',       0.30),
    ('entities_per_loc',       'csharp',     0.30),
    ('entities_per_loc',       'cpp',        0.25),
    ('entities_per_loc',       'c',          0.20),
    ('entities_per_loc',       '',           0.20)  -- fallback
ON CONFLICT (metric_key, language) DO NOTHING;
