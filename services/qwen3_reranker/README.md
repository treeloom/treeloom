# qwen3-reranker — drop-in reranker service

Serves `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` (a verified score-identical
sequence-classification port of `Qwen/Qwen3-Reranker-0.6B`) behind the same
`POST /rerank` contract as TEI, because TEI cannot host Qwen3 rerankers
(`'classifier' model type is not supported for Qwen3`, TEI cuda-1.9).

## Choosing a reranker

Both options stay available; `RERANKER_URL` in `.env` decides. Measured on
`benchmarks/queries/featbit-clean-100.jsonl` (deterministic: `--no-hyde
--no-graph`, summary vector off, rerank pool 50, RTX 3060):

| | bge-reranker-v2-m3 (TEI :8081) | Qwen3-Reranker-0.6B (this service, :8086) |
|---|---|---|
| recall@1 | 0.66 | **0.70** |
| recall@5 | 0.84 | **0.90** |
| MRR | 0.730 | **0.786** |
| search latency (pool 50) | **~0.9 s/query** | ~3 s/query |
| serving | TEI (Rust, continuous batching) | transformers loop (this service) |
| VRAM | ~1.2 GB | ~1.5 GB |
| params | 568M | 596M |

**Quality:** Qwen3 wins decisively (+0.04 / +0.06 / +0.056). **Latency:** Qwen3
costs ~3× per search, mostly serving overhead (naive transformers loop + the
~90-token prompt template per pair), not model size. **The standard
operational choice may still be bge** when search latency matters more than
ranking quality; flip `RERANKER_URL` back to `http://localhost:8081` and
restart the indexer — no other change needed.

Note: the same A/B run on the old artifact-heavy query set showed Qwen3
*losing* — always benchmark rerankers on answerable queries
(`featbit-clean-100.jsonl`).

## Running

```bash
docker compose --profile qwen3 up -d --build qwen3-reranker
# or set COMPOSE_PROFILES=qwen3 in .env — run.sh then starts it automatically
curl localhost:8086/health
```

Set `RERANKER_URL=http://localhost:8086` in `.env` and restart the host
indexer. The TEI bge service can stay up alongside (it's the fallback).

## Tuning

- `RERANK_BUCKET` (default **on**): sort the candidate set by token length
  before batching so a batch of short chunks isn't padded up to one long chunk's
  length. The forward pass is GPU-COMPUTE-bound and its cost is `batch_size ×
  longest_doc_in_batch`, so this removes cross-padding FLOP waste. **~21% faster
  on realistically-varied code chunks** (clean-100 deterministic A/B vs the old
  in-order batching, RTX 3060), **quality-neutral** (r@1 0.64 vs 0.63, r@5 0.84
  tied, MRR 0.727 vs 0.722 — within the fp16 batch-composition jitter the
  service already has; top-5 identical). Scores are mapped back to the caller's
  original order.
- `RERANK_ATTN` (default `sdpa`): attention impl. `sdpa` (PyTorch
  scaled_dot_product_attention) beats HF eager; falls back to eager if rejected.
- `RERANK_COMPILE` (default **off**): `torch.compile(dynamic=True)`. Gave +27%
  on *fixed* shapes but **recompiles on every new padded shape** — real chunks
  have continuously varied lengths (bucketing makes this worse), measured as 34 s
  recompile spikes on live `/search`, a net loss. The Dockerfile installs
  `build-essential` so it *can* compile (inductor/triton needs a C compiler);
  only enable it once you add fixed-bucket sequence padding (round seq length up
  to a small fixed set) so inductor compiles a handful of graphs and stops.
- `RERANK_INFER_BATCH` (default 16): pairs per forward pass. Bumping to 48 was
  only ~8% faster (50 long docs 1950→1789 ms) — the GPU is compute-bound, not
  overhead-bound, so batch size has little headroom. Verified: splitting a rerank
  across two GPU processes did NOT speed it up and two concurrent loads each took
  2× as long — no compute headroom despite low VRAM, so replicas/CUDA-streams add
  throughput-under-load, not single-query latency. The levers are fewer tokens
  (bucketing, lower max-length, smaller pool) or faster kernels (sdpa), not
  parallelism.
- `RERANK_MAX_LENGTH` (default 1600): tokens per (query, doc) pair incl. template.
- `RERANK_INSTRUCTION`: the `<Instruct>:` line; default is code-search-specific.
- `MODEL_ID`: don't bother with `tomaarsen/Qwen3-Reranker-4B-seq-cls` — tested
  2026-06-11 on a dedicated 12 GB GPU (fp16 ~8 GB): **worse than the 0.6B on
  every metric** (r@1 0.65 / r@5 0.84 / MRR 0.715 vs 0.70 / 0.90 / 0.786) at
  ~6.6 s/query. Unlike the 0.6B port, the 4B port has no score-identical
  validation against the official model, which may explain it.

The prompt template constants (`PREFIX`/`SUFFIX` in `app.py`) are part of the
model contract — Qwen3-Reranker scores garbage without them. Do not edit.
