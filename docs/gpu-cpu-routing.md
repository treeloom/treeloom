# Predictive GPU/CPU Routing for TEI Embedding

> **Historical — routing mechanics superseded.** This postmortem is kept for the
> GPU-OOM history and the `GPU_MAX_TOKENS` threshold, which are still accurate.
> The mechanics below are not: the character heuristic and `_pick_url()` no longer
> exist, chunking now uses chonkie's `CodeChunker` (not `SemanticChunker`), and
> embedding backends live in the Postgres `embedding_backends` registry rather
> than `.env` URLs. For how routing works today, see
> [ADR-001](adr-001-cost-aware-embedding-proxy.md) (`adapters/tei/embedding_proxy.py`).
>
> Read this if TEI (HuggingFace Text Embeddings Inference, the embedding
> server) is crashing with out-of-memory errors during indexing, or to
> understand why `GPU_MAX_TOKENS` exists.

## Problem

The tree-sitter `SemanticChunker` can produce single chunks covering entire files when it can't find semantic boundaries. These oversized chunks exceed the TEI model's max context (8192 tokens for jina-embeddings-v2-base-code), causing GPU OOM crashes that kill the entire TEI process. Restarting the GPU server costs ~15 seconds of downtime + warmup, and causes cascading failures for subsequent files.

## Root Cause

`CodeIndexer.parse_and_chunk()` uses chonkie's `SemanticChunker` with tree-sitter. For files where tree-sitter doesn't identify clear semantic boundaries (large functions, monolithic scripts, JSON, YAML), the chunker falls back to producing a single chunk. A 29KB Python file becomes one chunk of ~9,800 estimated tokens — well past the 8192 limit.

Example: `provision_uat.py` (29,460 chars, 1 chunk, ~9,820 est tokens) caused OOM in every indexing run.

## Solution: Dual TEI + Predictive Routing

### Architecture

```
                    ┌─────────────────┐
                    │    Indexer      │
                    │  (host:8001)    │
                    └───────┬─────────┘
                            │ embed(texts)
                    ┌───────▼─────────┐
                    │  _pick_url()    │
                    │  est_tokens =   │
                    │  max(len(t)//3) │
                    └───┬─────────┬───┘
               >4096    │         │   <=4096
              ┌─────────▼─┐   ┌──▼──────────┐
              │ CPU TEI   │   │  GPU TEI    │
              │ :8083     │   │  :8082      │
              │ cpu-1.9   │   │  cuda-1.9   │
              │ (ONNX)    │   │  (Candle)   │
              └───────────┘   └──┬──────────┘
                                 │ on OOM/disconnect
                                 │ (reactive fallback)
                                 ▼
                           CPU TEI :8083
```

### Key Files

- `src/treeloom/adapters/tei/embedding_proxy.py` — **(current)** `EmbeddingProxy` does cost-aware routing (real HF tokenizer); GPU/CPU pin when max chunk tokens > `GPU_MAX_TOKENS`; reactive failover across backends. See `docs/adr-001-cost-aware-embedding-proxy.md`.
- `docker-compose.yml` — **(current)** `tei-embedding` (GPU) + `tei-embedding-cpu` (CPU); their host ports come from `TEI_EMBEDDING_PORT` / `TEI_EMBEDDING_CPU_PORT` in `.env` (8082 / 8083 in the setup below).
- `.env` — `EMBEDDING_URL`, `EMBEDDING_FALLBACK_URL` (single-URL forms, still accepted as seeds when the plural `EMBEDDING_URLS` / `EMBEDDING_FALLBACK_URLS` are unset), `GPU_MAX_TOKENS`.

### Env Vars

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_URL` | required | Primary (GPU) TEI endpoint. Still read, as the single-URL fallback for `EMBEDDING_URLS` (comma-separated GPU-class). Today both only **seed** the Postgres `embedding_backends` registry on first start and are ignored afterwards. |
| `EMBEDDING_FALLBACK_URL` | (optional) | CPU TEI endpoint for large chunks + OOM recovery (single-URL fallback for `EMBEDDING_FALLBACK_URLS`; same seed-once rule) |
| `GPU_MAX_TOKENS` | 4096 | **(current)** Threshold: batches whose largest chunk exceeds this many tokens go to a CPU-class backend |

### How It Works

1. **Predictive** (`_pick_url`): Before each batch, estimate `max(len(chunk_text) // 3)`. If > `GPU_MAX_TOKENS`, route entire batch to CPU. Logs: `[embed] routing to CPU: predicted N tokens > 4096`

2. **Reactive** (fallback): If GPU returns 424/502/503 or any `httpx.HTTPError` (connection drop, read error, etc.), retry that batch on CPU. Logs: `[embed] GPU OOM/disconnect ... falling back to CPU`

3. GPU server has `restart: unless-stopped` — it recovers automatically after OOM crashes. The reactive fallback catches transient errors during restart.

### Pre-Scanning (Future Enhancement)

For repos with known problematic files, a pre-scan can classify files before indexing:

```python
from treeloom.indexer import CodeIndexer   # shim over adapters/tree_sitter/indexer.py
idx = CodeIndexer()
# For each file: chunks = idx.parse_and_chunk(fp)
# if max(len(c["text"])//3 for c in chunks) > GPU_MAX_TOKENS:
#     → route to CPU
```

This eliminates ALL GPU crashes and gives accurate ETA. On featbit (1785 files), the pre-scan found 37 files (2%) needing CPU routing, with the worst being `package-lock.json` at ~188,840 est tokens.

## Other Fixes Applied During featbit Reindex

### Milvus Connection
- **Issue**: our Milvus ran in Kubernetes behind an ingress that returned 502 despite the pod being healthy
- **Workaround**: `kubectl -n milvus port-forward svc/milvus 19531:19530`, then `MILVUS_URI=http://localhost:19531`
- **Code**: Added `MILVUS_URI` env var (takes precedence over `MILVUS_HOST`/`MILVUS_PORT`)

### pymilvus 3.0 Recursion Depth
- **Issue**: Large `mc.insert()` calls hit Python recursion limit
- **Fix**: Batch inserts in groups of 100 (`MAX_INSERT_BATCH`)

### Dimension Mismatch
- **Issue**: Existing Milvus collection had 1024-dim schema, but jina model produces 768-dim
- **Fix**: Set `VECTOR_DIM=768` in `.env`, dropped old collection to recreate with correct dim

### Port Mapping
- **Issue**: `EMBEDDING_URL=http://localhost:8080` but TEI was on 8082
- **Fix**: Corrected to `http://localhost:8082`

## featbit Reindex Results

- Files: 1785 total, all indexed
- Chunks: 1771 (14 files produced no chunks — unsupported types or empty)
- Duration: ~68 minutes (with GPU restarts; would be ~45 min with pre-scan)
- Errors: 0
- GPU crashes avoided by predictive routing: 37 files proactively routed to CPU
