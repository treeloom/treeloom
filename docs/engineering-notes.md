# Engineering notes

Reference material moved out of the root `CLAUDE.md` on 2026-09-22 so the always-loaded file holds only the invariants. Everything here was previously in `CLAUDE.md`; sections are verbatim. Facts date from the point they were written (most recent refresh 2026-09-21).

## Running the stack

```bash
./run.sh
```

`run.sh` does, in order: `pip install -e .`, `docker compose build mcp-server ui`, brings up `postgres`, `tei-embedding` (or `tei-embedding-cpu` with the `cpu` profile), `tei-reranker`, and `ui` — plus `neo4j` + `milvus` when `COMPOSE_PROFILES` includes `local-infra` (the quickstart default), `qwen3-reranker` with the `qwen3` profile, and the `mcp-server` container (deprecated HTTP+SSE transport) with the `http-mcp` profile — then runs `uvicorn treeloom.indexer_service:app` on port 8001 in the foreground. The indexer must run on the host (not in Docker) so it can reach arbitrary user paths. `run.sh` still builds the `mcp-server` image (cheap, keeps the opt-in path ready) but does NOT start that container unless `COMPOSE_PROFILES` includes `http-mcp` — the default MCP transport is stdio via the `treeloom-mcp` console script, spawned directly by your MCP client with no container involved. If you do opt into `http-mcp`, that container reaches the indexer via `host.docker.internal:8001`.

Service ports:
- MCP server: stdio via the `treeloom-mcp` console script (no port). The
  deprecated HTTP+SSE transport served by the `mcp-server` container is
  opt-in: `COMPOSE_PROFILES=http-mcp`, published on `127.0.0.1` only. That
  compose profile is the whole switch — the container's `CMD` always runs
  `uvicorn treeloom.main:app` (SSE) regardless of any env var; it never reads
  `TREELOOM_MCP_TRANSPORT`. `TREELOOM_MCP_TRANSPORT` is read only by the
  `treeloom-mcp` console script (`mcp_stdio.main()`), and only `stdio` (the
  default) is meaningful there — do NOT set `TREELOOM_MCP_TRANSPORT=sse` in a
  shared `.env`: `src/treeloom/__init__.py` loads the repo `.env` from any
  cwd an editable install runs in, so it would silently turn your
  `treeloom-mcp` script into a second, colliding SSE server instead of
  toggling the container.
- Indexer FastAPI: `http://localhost:8001` (host)
- TEI embedding: `8082`, TEI reranker: `8081`
- Neo4j Bolt: `7687`, HTTP: `7474` (default auth `neo4j/treeloom_pass`)
- Milvus: `localhost:19530` with the `local-infra` compose profile (in-compose standalone), or an external instance via `MILVUS_URI`

Copy `.env.example` to `.env` before first run.

## Simple deployment profile & backend dispatch

`TREELOOM_PROFILE=simple` (see `docs/simple-mode.md`, `.env.simple.example`,
`./run-simple.sh`, `docker-compose.simple.yml`) is the no-GPU evaluator path:
embedded LanceDB (`~/.treeloom/lancedb`) + embedded SQLite graph
(`~/.treeloom/graph.db`) + OpenAI API for embeddings/HyDE/summaries + an
in-process fastembed CPU reranker, with only postgres in Docker by default —
MCP access is stdio (no container); the deprecated HTTP+SSE `mcp-server`
container is an opt-in extra behind `COMPOSE_PROFILES=http-mcp`, same as full
mode.
The profile only `setdefault`s env vars (`_apply_profile_defaults` in
`src/treeloom/__init__.py`); the real switches are four selector vars usable
independently:

- `VECTOR_STORE=milvus|lancedb|chromadb` (chromadb is dense-only/degraded —
  Chroma's sparse/BM25 indexing is Cloud-only)
- `GRAPH_STORE=neo4j|sqlite`
- `EMBEDDING_PROVIDER=tei|openai` (openai bypasses the `embedding_backends`
  Postgres registry entirely)
- `RERANKER_PROVIDER=http|local`

The root shims (`treeloom/retriever.py`, `graph_store.py`, `embedder.py`,
`reranker.py`) are the canonical dispatchers and `_LEGACY_MODULES` points at
them. **Never import `adapters/milvus/vector_store` or
`adapters/neo4j/graph_store` directly from startup-path code** — both
`require_env` their service vars at import and crash simple mode
(`tests/unit/test_simple_profile.py` guards this in a hermetic subprocess).
`required_settings()` in `infrastructure/config.py` computes the mandatory
env vars per selector combo. The store-agnostic search orchestration
(`graph_search`, rerank, graph rescoring, ChunkHit shaping) lives in
`application/retrieval.py` — every vector backend gets the full pipeline; a
new backend only implements the milvus module surface returning milvus-shaped
hits. `LLM_API_KEY`/`OPENAI_API_KEY` adds a Bearer header to LLM calls;
`LLM_COMPAT_CHAT_TEMPLATE_KWARGS=0` is required against api.openai.com (it
400s on the Qwen3 thinking toggle and `_chat` swallows the error → silent
HyDE/summary loss).

## Two-process architecture

There are two FastAPI apps and they are not interchangeable:

- `treeloom/main.py` (HTTP+SSE transport, opt-in, deprecated upstream) — wraps the MCP server over SSE. The default transport is stdio via `application/mcp_stdio.py`, which serves MCP clients directly (no FastAPI app, no port, one process per client, spawned by the client itself). **The MCP server is a pure HTTP proxy**: every tool calls the indexer via `INDEXER_URL` (its only service env var); it imports no Milvus/Neo4j/TEI client code.
- `treeloom/indexer_service.py` (port 8001, on host) — owns all filesystem reads, chunking, embedding, Milvus inserts, Neo4j writes, AND all retrieval (`/search`, `/graph-explore`).


## Common dev tasks

```bash
# Install in editable mode
pip install -e .

# Start the indexer alone (containers must be up)
uvicorn treeloom.indexer_service:app --host 127.0.0.1 --port 8001

# Index a local repo via the MCP indexer HTTP API.
# When AUTH_ENABLED=true, every mutating call, GET /sources, AND search
# (/search, /graph-explore — they skip the auth middleware but authenticate
# and authorize per repo themselves; 401 anonymous unless
# TREELOOM_SEARCH_OPEN=1) require `-H "Authorization: Bearer $TREELOOM_KEY"`.
# /health, /status, and GET /jobs remain reachable without one — but under AUTH_ENABLED=true the
# job views are REDACTED for an unauthenticated caller: progress counters
# and status are returned, `source`/`current_file`/`error`/`message` and the
# per-file error log's paths and exception text are nulled, with
# `redacted: true` on the response. Auth off = unchanged, nothing redacted.
curl -X POST http://localhost:8001/index-repo \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TREELOOM_KEY" \
  -d '{"path": "~/source/myrepo"}'

# Re-index a source even if it's already DONE (skip the dedup short-circuit)
curl -X POST http://localhost:8001/index-repo \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TREELOOM_KEY" \
  -d '{"path": "~/source/myrepo", "force": true}'

# Check whether an indexed source is out of date relative to current HEAD
curl -H "Authorization: Bearer $TREELOOM_KEY" \
  http://localhost:8001/sources/<source_id>/staleness

# Rebuild only the graph for one source (no re-embedding) — pass 2 of two-pass indexing
curl -X POST http://localhost:8001/index-graph \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TREELOOM_KEY" \
  -d '{"source_id": "<source_id>"}'

# Bulk-rebuild the graph for EVERY indexed source (repair an empty-graph import).
# Hits the indexer directly; auth via $TREELOOM_KEY (or $TREELOOM_MCP_API_KEY).
# Rebuilds all sources by default — do NOT trust graph_indexed here (see below).
python -m treeloom.rebuild_graphs --wait
python -m treeloom.rebuild_graphs --dry-run          # preview, enqueue nothing
python -m treeloom.rebuild_graphs --only-missing     # only graph_indexed=false (two-pass)
python -m treeloom.rebuild_graphs --source-id <id> --source-id <id2>

# Fleet ops (docs/fleet-operations.md). Bulk-onboard many repos
# from a YAML/JSON manifest (idempotent — re-running is a no-op); print the
# fleet health rollup ("is the fleet healthy?" in one call).
python -m treeloom.fleet onboard repos.yaml --wait        # --dry-run / --force
python -m treeloom.fleet health --staleness --entities    # or --json
curl http://localhost:8001/fleet?summary_only=true

# Indexer status (live progress while running) — public
curl http://localhost:8001/status

# The benchmark CLI is `python -m treeloom.benchmark <subcommand>` with
# subcommands: queries | run | ab | agentic | gold | rollup | clean (argparse, see
# application/benchmark/__init__.py). Query JSONL files live in
# benchmarks/queries/ — fields: id, query, relevant_files (ground truth),
# entity{id,name,type}, repo, difficulty, strategy.
#
# REFERENCE SET: benchmarks/queries/featbit-clean-100.jsonl (baseline r@1 0.66 /
# r@5 0.84 / MRR 0.730 with jina+bge @ pool 50, deterministic flags). The old
# featbit-enhanced query sets were artifact-heavy — ~26% of their queries were
# unanswerable (non-discriminating entity names, duplicate texts with different
# ground truths) and they INVERTED a reranker A/B verdict; they have been removed.
# Regenerate the clean set: python benchmarks/make_clean_queries.py
#
# For the documented grep-vs-treeloom eval workflow (clean → agentic → rollup, with
# significance), see docs/benchmark-eval.md.
#
# VERDICT (n=100 × 4 repos incl. C#/Java/JS + featbit control,,
# docs/benchmark-findings.md): treeloom is SEMANTIC search, NOT symbol lookup. On
# symbol-BEARING queries grep wins/ties (it string-matches the identifier) — don't
# benchmark only those, they understate treeloom. On symbol-FREE queries treeloom
# beat a grep/read agent on answer quality (pooled correctness p<0.0001; +Cohere
# also recall p=0.00067) at lower tokens — deepseek-chat, pre-rebuild epoch.
# 2026-09 REFRESH (docs/benchmark-results-2026-09.md; featbit pinned v5.4.9, agents
# deepseek-flash/claude-sonnet-5/gpt-5.5, independent gpt-4o judge): treeloom is
# 11–47% CHEAPER than grep in all 6 cells (significant per query in 5/6, every
# symbol-free cell), but correctness is statistically TIED everywhere (best p=0.061).
# The quality edge is AGENT-CAPABILITY-DEPENDENT: weaker agents (gpt-4o) collapse
# with grep, capable ones recover at higher cost. Lead public claims with COST, never
# "better answers". ALWAYS run a symbol-free cell (a `queries --llm-symbol-free` set,
# then `clean --symbol-free`) as standard. Graph RESCORING's value is
# reranker-dependent (see the constraint below): ~0 over Cohere, a real rank-1 lift
# under the production Qwen3 reranker; the graph layer's durable value is the
# neighbor/community PAYLOAD fed to the agent.
#
# The run CLI exposes only --no-hyde/--no-hybrid/--no-graph and requires a scope
# (--path-prefix or --cross-repo-search). For use_summary_vector or rerank_pool,
# call run_benchmark() (application/benchmark/runner.py) directly — its
# feature_flags dict merges verbatim into the /search request body.

# Generate benchmark queries for a source dir
python -m treeloom.benchmark queries --source-dir ~/source/featbit --count 20

# Retrieval-quality run against the live indexer /search (precision/recall/MRR
# to stdout). Feature-flag toggles, not modes.
python -m treeloom.benchmark run --queries benchmarks/queries/featbit-clean-100.jsonl \
  --search-url http://localhost:8001 --path-prefix /path/to/featbit \
  [--no-hyde --no-hybrid --no-graph]

# AB-compare retrieval feature-flag configs
python -m treeloom.benchmark ab --queries benchmarks/queries/featbit-clean-100.jsonl --graph

# Agentic token+quality benchmark: a real LLM agent answers each query once per
# arm — grep/glob/read_file tools, treeloom search_code, and optionally
# zilliztech/claude-context and aider's repo-map
# (--arms grep,treeloom,claude-context,repomap) — and we compare real API token
# usage + answer quality (recall@k/MRR + LLM-judge).
# The claude-context arm spawns its MCP server via npx (override with
# CLAUDE_CONTEXT_MCP_CMD) and indexes --repo through it before the first query;
# it needs MILVUS_ADDRESS (+MILVUS_TOKEN) and embedding env (EMBEDDING_PROVIDER=OpenAI,
# OPENAI_BASE_URL — TEI's /v1 works, OPENAI_API_KEY, EMBEDDING_MODEL) or it exits
# at startup. The repomap arm builds aider's real RepoMap once per run in an
# isolated env (`uv run --with aider-chat`, override with REPOMAP_PYTHON_CMD —
# never install aider-chat into this venv, its pins conflict) and re-sends it in
# the system prompt every turn (the honest cost of that approach); budget via
# --repomap-tokens (default 8192). The code-graph-rag arm spawns vitali87's
# `cgr mcp-server` (isolated py3.12 uvx env, override CGR_MCP_CMD) — needs
# Memgraph on localhost:7688 (NOT 7687 = our Neo4j) and CYPHER_* env for its
# text-to-Cypher LLM; its `openai` provider speaks the OpenAI Responses API
# (deepseek 404s — point CYPHER_ENDPOINT at the local llama.cpp /v1, which
# supports /v1/responses). Released package has NO C# grammar — caveat any
# results on C#-heavy repos. The summary's generic `comparisons` block covers
# every arm pair.
# Gold answers are generated once and cached under benchmarks/gold/<repo>.jsonl.
python -m treeloom.benchmark gold    --queries benchmarks/queries/featbit-clean-100.jsonl --limit 5
python -m treeloom.benchmark agentic --queries benchmarks/queries/featbit-clean-100.jsonl \
  --repo ~/source/featbit --search-url http://localhost:8001 \
  --arms grep,treeloom --max-turns 12 [--model M] [--native-tools|--no-native-tools] \
  [--regen-gold --limit N --difficulty {easy,medium,hard} --strategy <name>]
```

The agentic benchmark needs a capable tool-calling LLM (set `BENCHMARK_QUALITY=STANDARD`
or point `NOMCP_LLM_URL`/`NOMCP_LLM_MODEL` at one — `resolve_llm_config()`; a small local
model can't follow the tool protocol) and a repo that is BOTH indexed (treeloom arm)
and present on disk with ground-truth queries (grep arm). The agent protocol defaults
**per vendor** (since 2026-07-05): native OpenAI-style `tool_calls` for hosted
openai/deepseek/anthropic routes, ReAct text for local models — the
`supports_native_tools` axis on `resolve_model_route`; `--native-tools` /
`--no-native-tools` force either, and the effective protocol is recorded as
`agent_protocol` in `_summary.json`. Agentic runs REFUSE floating model aliases
(`deepseek-chat`/`deepseek-reasoner`, bare `gpt-4o`/`gpt-4o-mini`, `*-latest`) for both
agent and judge unless `--allow-floating-model` or `TREELOOM_ALLOW_FLOATING_MODELS=1` —
permitted runs stamp `floating_model_warning: true` (DeepSeek publishes no dated IDs, so
DeepSeek runs always need the override). Every `_summary.json` records vendor-reported
`served_models` so alias drift is visible after the fact; `--debug-transcripts` persists
full transcripts (incl. tool observations) into result rows. Before any paid run, pass
the smoke gate: `scripts/agentic_smoke.sh` (3 queries/model — termination, grounding,
mean correctness ≥ 3.0, `served_models` present). **BASELINE EPOCH (2026-07-05):** the
protocol switch means pre-rebuild agentic numbers (before 2026-07-05, ReAct epoch) are
NOT comparable with post-rebuild native-loop numbers — never mix them in a rollup; the
new-epoch reference cells are
`benchmarks/results/agentic/2026-07-05T205924_featbit_summary.json` and
`2026-07-05T215301_featbit-symfree_summary.json` (deepseek cost tier, floating
alias) plus the PINNED anchor pair on `gpt-4o-2024-08-06` (immune to alias
drift): `2026-07-05T232746_featbit_summary.json` (treeloom corr 3.89 vs grep
3.41, p=0.006, −44% tokens) and `2026-07-05T234145_featbit-symfree_summary.json`
(treeloom corr 3.23 vs grep 2.13, p=2.3e-10, −88% tokens, grep cap-rate 25.8%
vs 0.8% — grep collapses on symbol-free at this tier). **CURRENT reference cells
(2026-09-21)** supersede the deepseek pair for current models: featbit pinned at
`v5.4.9`, sets `featbit-v549-clean-100` / `featbit-v549-symfree-100`, summaries
`2026-09-21T{132010,135248,150023,154716,150024,154211}_featbit-v549*_summary.json`
(cost −11..−47% vs grep, correctness tied). The gpt-4o anchor stays the weak-agent
record.

Unit tests: `pytest tests/unit -q` (1,656 Detroit-style tests as of 2026-09-21, no services needed;
run as `env -u PYTHONPATH python -m pytest tests/unit` — a system `/opt/ros` entry on
`PYTHONPATH` otherwise hijacks pytest plugin autoload and dies on `ModuleNotFoundError: lark`).
pytest `testpaths` covers only tests/unit). Retrieval *quality* is validated via the
benchmark harness, not unit tests. The `run`/`ab`
subcommands print JSON to stdout; the `agentic` subcommand writes per-query rows to
`benchmarks/results/agentic/<UTC-timestamp>_<repo>.jsonl` plus a `_summary.json` aggregate
(per-arm mean tokens/turns/recall@k/MRR/judge, and a `comparison` block with tokens-saved %,
quality-per-1k-tokens, and treeloom win-rates).


## Constraints to respect

- **tree-sitter is unpinned** — `tree-sitter-language-pack` (bundled via `chonkie[code]`) manages parsers. Do NOT add `tree-sitter-languages` or pin `tree-sitter` below 0.24.
- **`CodeChunker` needs the parser injection in `_make_chunker`** (`adapters/tree_sitter/indexer.py`). Since tree-sitter-language-pack 1.8.x, `get_parser()` returns a Rust pyo3 parser whose API chonkie's `CodeChunker.chunk()` can't drive (`parse()` wants `str`, `Tree.root_node` is a method) — it raises a `TypeError` that chonkie's `finally` block masks as `UnboundLocalError`. `_make_chunker` overwrites `chunker.parser` with `tree_sitter.Parser(get_language(lang))` (the Python binding, same path `graph_extractor.py` uses). Removing that line silently sends every file down the TokenChunker/character fallback. Chunkers are built per call — parsers are not safe for concurrent `parse()` and `parse_and_chunk` runs in threads (`asyncio.to_thread`). CodeChunker can overshoot `CHUNK_SIZE` by ~6% (node-group joins); harmless — the embedding model (`jina-embeddings-v2-base-code`) takes 8192 tokens. A/B on featbit (3×50 queries, `--no-graph` for determinism) showed retrieval parity with TokenChunker (recall@5 0.460 both, MRR 0.334/0.335); CodeChunker is kept because chunks average ~24% fewer tokens (378 vs 496) and 0% start mid-block (vs 29%), so search responses are cheaper and snippets start at definition boundaries.
- **Markdown is ingested via a separate, non-tree-sitter path.** `.md`/`.markdown`/`.mdx` live in `MARKDOWN_MAP`, NOT `LANGUAGE_MAP` (`indexer.py`) — keeping them out of `LANGUAGE_MAP` is load-bearing, not legacy: `tree-sitter-markdown`'s C++ grammar asserts and crashes the indexer, and `graph_extractor._detect_language` reads `LANGUAGE_MAP`, so a markdown file falls through to the safe `_fallback_module_entity` (Module-only node, no grammar). Chunking routes through `_make_markdown_chunker` (chonkie `RecursiveChunker.from_recipe("markdown")`, a pure-Python splitter whose top level breaks on heading markers `######..#` then paragraphs/lines), never `CodeChunker`; its chunks expose the same `.text/.start_index/.end_index` surface so the line-mapping is unchanged, and chunk failures still fall back to `_character_chunks`. `detect_language` returns `"markdown"` for these extensions; `SUPPORTED_EXTENSIONS` (= `LANGUAGE_MAP` ∪ `MARKDOWN_MAP`) is the single source of truth for the repo/directory walk filter (`_collect_files`) so it can't drift. Markdown chunks become first-class `search_code` hits (vector-only — no graph signals beyond the Module node). Out of scope: doc link-graph, frontmatter, code-fence extraction.
- **`LANGUAGE_QUERIES` keys must equal `LANGUAGE_MAP` values** (`graph_extractor.py`), and only the `class`/`function`/`import`/`call` keys are consumed. A mismatched key fails silently: the language falls back to a Module-only entity (no graph signals, no `find_definition`, no chunk headers). This bit twice — `"c_sharp"` vs `"csharp"` left every .cs file graphless, and rust's `"struct"` key never ran. Quantified captures like `(base_list (identifier) @x)?` inside a `@class.def` pattern duplicate the def capture on multi-base classes and desync the index-paired name/def lists — put base captures in a separate pattern without `@class.def`.
- **Reranker is a configured choice, decided by `RERANKER_PROVIDER` (`http`|`local`|`local-st`|`voyage`|`cohere`|`zeroentropy`) + `RERANKER_URL`** (read at indexer startup — restart to switch). `local-st` (`adapters/rerank/st_cross_encoder.py`) runs a HF sentence-transformers CrossEncoder in-process (GPU if available) for weights fastembed can't load (e.g. `zeroentropy/zerank-*`, `RERANKER_LOCAL_MODEL` default the Apache-2.0 `zerank-1-small`); needs `sentence-transformers` (the `local-st` extra) and a numpy/transformers-compatible env. The `zeroentropy` cloud provider (`rerank-2`/`zerank-1`/`zerank-1-small`, `ZEROENTROPY_API_KEY`) benchmarked 0.66–0.67 r@1 — middle tier, above the GPU bge control but below Cohere `rerank-v4.0-pro`; on featbit `zerank-1-small` ≈ control. The `local` provider runs fastembed ONNX cross-encoders in-process on CPU (`RERANKER_LOCAL_MODEL`, default `jinaai/jina-reranker-v1-turbo-en` ~150MB at 0.58/0.74/0.645 on clean-100 vs 0.63/0.80/0.698 GPU-bge control; `jinaai/jina-reranker-v2-base-multilingual` ~1.1GB TIES the GPU control (0.60/0.84/0.689) at ~16 s/query) — simple-profile default, no service needed. The **hosted-API providers** (`adapters/rerank/cloud_reranker.py`, `_PROVIDERS` registry + `_CLOUD_PROVIDERS` dispatch set) — `voyage` (`RERANKER_MODEL` default `rerank-2.5`/`rerank-2.5-lite`) and `cohere` (`rerank-v4.0-pro`/`rerank-v3.5`) — are GPU-less quality at sub-second latency, all measured at/above the GPU bge control on clean-100 (Cohere `rerank-v4.0-pro` 0.74/0.85/0.787 is the only reranker that beats the control with per-query significance, p=0.011; full table in `docs/simple-mode.md`). Key resolution: `RERANKER_API_KEY` (generic override) else the **active provider's** specific key (`VOYAGE_API_KEY`/`COHERE_API_KEY`) — NOT a fixed precedence, so multiple keys in `.env` don't cross-wire (jina stubbed). For `http`: Two supported options, both measured on `benchmarks/queries/featbit-clean-100.jsonl` (pool 50, deterministic flags): `BAAI/bge-reranker-v2-m3` via TEI on `:8081` (r@1 0.66 / r@5 0.84 / MRR 0.730, ~0.9 s/query — the latency choice) and `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` via `services/qwen3_reranker` on `:8086` (0.70 / 0.90 / 0.786, ~3 s/query — the quality choice). **TEI cannot host Qwen3 rerankers** (`'classifier' model type is not supported for Qwen3`, even on cuda-1.9); the drop-in service wraps the seq-cls port with transformers and the model's mandatory prompt template (see its README — do not edit the template). Enable with `COMPOSE_PROFILES=qwen3` (run.sh auto-starts it). Beware the dead-reranker failure mode: if `RERANKER_URL` points at a down service, scores fall back flat and recall silently craters — smoke-test `/rerank` after switching. Old benchmark caveat: on the pre-clean artifact query set this same A/B showed Qwen3 *losing* — only trust reranker comparisons on answerable queries. **Don't bother with `Qwen3-Reranker-4B-seq-cls`**: tested 2026-06-11 (fp16 on a dedicated 12GB GPU), it scored *worse* than the 0.6B on every metric (0.65/0.84/0.715 vs 0.70/0.90/0.786) at ~6.6 s/query — the 4B seq-cls port lacks the 0.6B's score-identical-to-official validation. 0.6B is the sweet spot.
- **Retrieval A/Bs run as an isolated second indexer** (port 8002), never against production state. Inline shell env beats `.env` (`os.environ.setdefault` in `src/treeloom/__init__.py`), so launch with overrides: `DATABASE_URL=...treeloom_ab` pointing at a fresh database — migrations auto-run, and critically the `embedding_backends` registry re-seeds from `EMBEDDING_URLS` (against the production DB that env var is IGNORED after first seeding); own `MILVUS_COLLECTION` + `VECTOR_DIM` for embedding arms; `skip_graph: true` on index jobs so the shared Neo4j (and its community embeddings) stays clean. Reranker arms need no re-index: read the production collection search-only (no `MILVUS_COLLECTION`/`VECTOR_DIM` overrides, swap `RERANKER_URL`) — but never POST `/index-*` to that instance. Drop the experiment DB and `treeloom_ab_*` Milvus collections afterward.
- **TEI batch limit is 32** for embedding, ~20 for reranking. `embedder.py` batches embeddings; `retriever.rerank` batches inputs to TEI's `/rerank` (controlled by `RERANK_BATCH_SIZE`, default 20) and uses `raw_scores: True` for cross-batch comparability. The pre-rerank pool is `MILVUS_TOP_K_PRE_RERANK` (default 50; per-request override via `rerank_pool` on `/search`). A repo-scoped featbit sweep (2026-06, `benchmarks/results/embedding-ab/rerank_pool_sweep.json`) showed pool 50 = 150 = 300 on recall@1/recall@5/MRR while 150 costs 2.8× the search latency; below 50 quality drops fast (40→0.72, 20→0.70 recall@5). Consider a larger per-request pool for `cross_repo` searches — the sweep didn't cover them.
- **Local TEI in compose runs on GPU 0** with `--dtype float16`, pinned via `device_ids: ["0"]`. Extra remote TEI backends listed in `EMBEDDING_URLS` act as optional overflow capacity for indexing many/large repositories and may be offline otherwise — day-to-day embedding and reranking are served by the local compose `tei-embedding` / `tei-reranker` services. Remember the `embedding_backends` Postgres registry, not `.env`, decides which URLs are actually used after first seeding. For laptops without an NVIDIA GPU, use `tei-embedding-cpu` (also fp16, `cpu` profile in run.sh) and skip the GPU services.
- **Milvus**: the `local-infra` compose profile runs a single-container Milvus standalone (embedded etcd, plain HTTP on `localhost:19530`); production deployments typically point at an external/managed instance instead. `MILVUS_HOST`/`MILVUS_PORT` are required env (fail-loud, no code default); `MILVUS_URI` defaults to `https://$MILVUS_HOST:$MILVUS_PORT`, so the plain-HTTP local standalone needs `MILVUS_URI=http://localhost:19530` set explicitly. Keep `.env.example` aligned with a working endpoint (a stale address here has caused silent failures twice).
- **MCP protocol requires the full handshake** before tool calls: connect SSE → receive `session_id` → send `initialize` → receive response → send `notifications/initialized` → then `tools/call`. The SSE message endpoint (`sse.handle_post_message`) is mounted as a raw ASGI app via `Mount` in `application/main.py` — NOT a FastAPI route — because it sends its own `202` response; wrapping it in `@app.post` double-responds and breaks every call after `initialize`. The agentic benchmark's treeloom arm does NOT use MCP; it calls the indexer `POST /search` directly (`adapters/benchmark/treeloom_tool.py`).
- **Neo4j driver is 6.x async** — use `result.fetch(n)` with an integer, not bare `fetch()`.
- **Don't mount `/home` into Docker** — use `host.docker.internal` (already set up via `extra_hosts: host-gateway`). The host indexer covers filesystem access.
- **`store_graph` uses `MERGE` for both source and target** so external import targets auto-create as `ExternalModule`. Don't switch to `MATCH`. **This also applies to the `Source` node in the entity-upsert block** (`MERGE (s:Source {id: $source_id}) WITH s UNWIND $entities …`): the `Source` node is only enriched with properties by `upsert_source()` in `_finalize_job` — which runs *after* every file is processed — so a `MATCH (s:Source …)` here finds nothing mid-job and silently drops every `Entity`, leaving `find_definition`/graph traversal globally empty. The relationship pass would still `MERGE` bare `Entity` nodes (id only, no `name`/`file_path`), so the failure is invisible to chunk counts and `errors`.
- **Milvus collection has an explicit schema** with `vector` (dense code), `summary_vector` (dense LLM summary, nullable), `sparse_vector` (server-side BM25 from `chunk_text`). `init_collection()` is idempotent (`has_collection` check). Schema-changing edits require dropping the collection and re-indexing every source.
- **LLM is required at startup** (fail-loud via `require_env`): `LLM_URL` + `LLM_MODEL`. Used for HyDE query expansion and per-chunk summary generation. Falls back silently to no-HyDE / no-summary if the LLM is unreachable at runtime — both indexing and search still work without it.
- **Hybrid search via Milvus `hybrid_search`** combines dense `vector`, optional dense `summary_vector`, optional dense HyDE-vector, and sparse BM25 via `RRFRanker(k=RRF_K)`. Toggle off per-call with `use_hybrid=False`.
- **Graph RESCORING is ON by default; the neighbor/community PAYLOAD is always on. Payload-without-rescoring is a first-class OPT-IN mode.** The graph layer adds value in two independent ways: (1) the neighbor + community-summary PAYLOAD fed to the agent — always returned by `graph_search` regardless of any flag — and (2) post-rerank RESCORING that fuses community-summary cosine + entity centrality + name-in-query into `final_score` via weights `GRAPH_ALPHA/BETA/GAMMA/DELTA`. **The value of (2) is reranker-dependent.** Ablation D3 (`docs/benchmark-findings.md`) showed (2) adds ~0 over a **strong** reranker (Cohere `rerank-v4.0-pro`) — there, `use_graph_scoring=false` ties/beats it and is cheaper. But a later retrieval re-validation under the **production** reranker (`Qwen3-Reranker-0.6B` on `:8086`) found (2) still materially lifts rank-1 — guava symbol-free **recall@1 0.16→0.49, MRR 0.448→0.621** (featbit unchanged, PowerToys neutral) — because a weaker reranker leaves room for graph symbol-promotion to fix rank-1. So rescoring **stays ON by default** (`USE_GRAPH_SCORING=1`); set `USE_GRAPH_SCORING=0` (payload-only) **only with a Cohere-class reranker**, where it's redundant and cheaper. Toggle per-request via `use_graph_scoring=true|false` on `/search` (and the `search_code`/`search_code_enhanced`/`graph_explore` MCP tools). `use_graph_scoring=false` does NOT suppress the payload anywhere; only the rescore branch in `application/retrieval.graph_search` is gated by it (`--vector-only` in the agentic benchmark is the lean mode that *also* drops the payload client-side). Benchmark CLI: `run --no-graph` / `--no-graph-scoring` turn rescoring off; `--graph-scoring` pins it on (the default). Community-summary embeddings + centrality (consumed ONLY by rescoring) are written to Neo4j by `community.run_post_index_signals()` at the end of every index job. The MCP-process community embedding cache is lazy and never invalidated within process — restart the MCP server to refresh.
- **Per-file error log** (`job_file_errors` table, migration `011_job_file_errors.sql`, store at `adapters/postgresql/job_file_error_store.py`). Every file that times out or raises during `_walk_and_index` or `_run_index_graph_job` gets a row recording `error_kind` (timeout/exception), `error_message`, `elapsed_s`, and `occurred_at`. `jobs.errors` stays as the cheap counter; this table is the queryable detail. Surfaces via `GET /jobs/{id}/errors` and MCP tool `list_job_errors(job_id)`. Best-effort writes — DB failure during error-logging is swallowed so the indexer's error-handling path never breaks. A job where **every** attempted file errored and produced 0 chunks now finalizes `failed` (not `done`) and **skips the source-registry upsert** — the all-broken-backend case used to read as success. The resolver is `_terminal_status_for_index(total_chunks, attempted_units, errors)`; partial success (some chunks) and genuine no-ops (0 errors / empty files) stay `done`.
- **Two-pass indexing** (migration `010_source_graph_indexed.sql`). `POST /index-repo|directory|file` accept `skip_graph: bool = false`; when true, chunks + embeddings are indexed but graph extraction is skipped and `source_records.graph_indexed` is set to FALSE. `POST /index-graph {source_id}` (auth required) creates a `kind="graph"` job that runs graph extraction over an already-chunks-indexed source, using the same JobStore/queue/Semaphore/Neo4j-retry stack as the regular path; calls `graph_store.delete_source(source_id)` first so re-runs are idempotent. Between pass 1 and pass 2, `search_code` works on vectors; `search_code_enhanced` and graph traversal return degraded results until the graph job completes. Exposed as MCP tools `index_graph(source_id)` and `rebuild_all_graphs(only_missing=False)`; the three `index_*` tools take `force: bool`. `rebuild_all_graphs` and the `python -m treeloom.rebuild_graphs` CLI default to rebuilding **all** sources because `graph_indexed` is NOT a trustworthy "needs a graph" signal — a job that ran with `skip_graph=false` but hit the old `MATCH (s:Source)` bug marked itself `graph_indexed=true` while writing zero entities, so `--only-missing`/`only_missing=true` would skip exactly the broken sources. `GET /sources` includes `graph_indexed` for that opt-in filter.
- **Search-response chunk shape is the `ChunkHit` contract.** `graph_search`/`graph_enhanced_search` (in `application/retrieval.py`) return `chunks` as **flat** dicts via `chunk_hit_from_milvus(r).model_dump()` (`file_path`, `snippet`, `start_line`, `end_line`, `language`, `score`, `source_id`, `graph_enhanced`) — NOT raw Milvus hits (`{id, distance, entity:{chunk_text,…}}`). The indexer `/search` + `/graph-explore` endpoints return these dicts unchanged and the MCP server does `ChunkHit(**c)`, so any new search path must run results through `chunk_hit_from_milvus` or the MCP layer 500s on missing `file_path`/`snippet`. `score` surfaces the graph-rescored `final_score` when present, else the reranker `relevance_score`, else raw `distance`. `_parse_search_response` in the MCP server defensively skips rows that fail `ChunkHit` validation rather than failing the whole response. **The `search_code`/`search_code_enhanced` MCP tools default to `response_format="markdown"`** — `_search_response_to_markdown` renders the dict as fenced code blocks under a `path:lines (score)` heading, ~16–22% cheaper to tokenize than the JSON dict FastMCP would emit; pass `response_format="json"` for the structured dict (programmatic callers). Snippets are also blank-line/trailing-whitespace stripped in `graph_search` before emit (leading indentation preserved; line ranges live in metadata, so nothing is lost). **Provenance**: chunks additionally carry optional `citation` (e.g. `myrepo@a1b2c3d4e5f6:src/Foo.py:10-42`); `commit_sha` is **no longer repeated per chunk** — it lives only in the top-level `sources` map keyed by source_id, alongside `indexed_at` (ISO 8601), `path`, `url`, `branch`, and `permalink_base` (only when url + commit_sha both present). Set `check_staleness: true` on the `/search` request (or on the `search_code`/`search_code_enhanced` MCP tools) to add `is_stale` + `current_sha` via a live `git ls-remote`/`rev-parse` per source. See `docs/result-provenance.md`.
- **Opt-in search `response_mode` — facet & summary_tail.** A `response_mode` field on the `/search` request (and `/graph-explore`, the `search_code`/`search_code_enhanced` MCP tools, and the `treeloom-facet`/`treeloom-summary-tail` benchmark arms) selects server-side body-shaping. `full` (default) is **byte-identical** to earlier — these modes are NEVER default, opt-in only. `facet` drops the snippet code body entirely and surfaces the `_chunk_header` text (enclosing class + defined symbols + signature) as a `header` field; the agent fetches ranges via `read_file`. `summary_tail` keeps the top-`SUMMARY_TAIL_HEAD` (=2) ranks' full snippets and replaces lower-ranked bodies with the indexed LLM summary (`summary_cache` Postgres lookup via `llm.cache_get_many`, keyed by `SHA1(chunk_text)`+model+prompt_version). **The summary SHA1 is computed over the ORIGINAL Milvus `entity["chunk_text"]`, captured BEFORE the header-prepend/blank-line-strip mutate the snippet** — threaded as a parallel `raw_texts` list aligned to the post-merge hits via an `id()` map (merged adjacent hits are fresh `model_copy` objects whose id is absent → `None` → un-summarizable → snippet fallback). **A missing summary ALWAYS falls back to the full snippet, never an empty body** (empty reads as "complete" to the agent = starvation). The pure transform is `_apply_response_mode(chunks, raw_texts, response_mode, summary_by_sha)` in `application/retrieval.py` (unit-tested in `tests/unit/test_response_modes.py`); `graph_search`/`graph_enhanced_search` take a `response_mode` param. `ChunkHit.snippet` is now defaulted (`""`, not required) so facet's body-less dict survives `ChunkHit(**c)` in the MCP layer — `file_path` stays the only required key. Bad `response_mode` → 400 at the endpoint. `_search_response_to_markdown` renders facet as `path:lines (score) — citation` + a header line (no code fence) and summary_tail/full bodies as fences. **Benchmark it** (the decisive metric is **mean turns** — a token drop with a turn rise is a NET LOSS): `python -m treeloom.benchmark agentic --queries benchmarks/queries/featbit-clean-100.jsonl --repo <featbit> --search-url http://localhost:8001 --arms grep,treeloom,treeloom-facet,treeloom-summary-tail` and again with `--llm-symbol-free`. summary_tail needs `summary_cache` populated for the repo (`SELECT count(*) FROM summary_cache`).
- **Token-optimization verdict: DON'T trim the search payload — `treeloom-full` is already the token-efficient config in an agent loop.** Every payload-shrinking idea was benchmarked agentically and every one lost or washed because of **compensatory fetch**: a smaller search response makes the agent declare it needs more and `read_file`/re-query, which raises **mean turns** and erases (or reverses) the token saving. Confirmed independently across `response_mode=facet` (turns +18%, tokens +20% — rejected), `response_mode=summary_tail` (turns flat, ~−7% tokens, quality ns — a marginal non-default win at best, and showed it loses at *every* summary verbosity tier: brief tier turns 4.55→4.90, tokens **+8.8%**), community-summaries-off / `adaptive_topk` / `strip_imports` / `query_class_payload` (all rejected), and the summary-verbosity sweep. **None were defaulted on** — they all ship as opt-in flags (default `full`/off). The *only* default-on token win was **serialization**, not content: markdown `response_format` (−26% per payload) + per-chunk `commit_sha`→top-level `sources` dedup + snippet blank-line strip; those don't remove information the agent then re-fetches. **Methodology rules (don't relearn the hard way):** (1) the governing metric is **mean turns**, judged together with payload tokens — a token drop with a turn rise is a NET LOSS; (2) raise `--obs-char-limit` (use 16000) when measuring payload deltas — the default 4000 cap truncates observations and *hides* the very deltas you're testing; (3) for a cheap pre-screen before a ~1h agentic cell, use the **sufficiency-proxy screen** (`application/benchmark/screen.py`): one LLM call/query → answer-or-`NEED_MORE`; **`need_more_rate` is the trustworthy leading indicator of compensatory fetch — `mean_correctness_answered` is CONFOUNDED by punt rate** (a config that punts hard queries via NEED_MORE inflates its own mean-over-answered = selection bias, not quality); (4) per-request `summary_prompt_version` on `/search` reads an alternate indexed summary tier without a restart (`cache_get_many(prompt_version=)`); (5) **verify the actually-served LLM/reranker/embedding models + GPU/CPU/API placement before any long run** — `.env` and `nvidia-smi` both lie (`.env` `LLM_MODEL` named a 3B while the box served a 35B; `nvidia-smi` misses containerized GPU procs — read the container startup log).
- **Preflight predicts indexing time** (`src/treeloom/preflight/`, migration `009_preflight_coefficients.sql`). CLI: `python -m treeloom.preflight <path>`. Endpoint: `POST /preflight` — requires auth when `AUTH_ENABLED=true`; the `path` mode is gated by `PREFLIGHT_ALLOWED_ROOTS` (colon-separated absolute paths) so the endpoint can't be used as an arbitrary-filesystem-enumeration oracle, and the `url` mode shallow-clones into a tempdir. Auto-runs inside `/index-repo` and `/index-directory` when `auto_preflight=true` (default) on local paths — surfaces yellow/red warnings + recommended env overrides in the `JobAck` without blocking. Coefficients live in Postgres (`preflight_coefficients`); update them after a known-good indexing run via `python -m treeloom.preflight --calibrate <source_id>`. Skip-patterns suggestions in the report are honored by the indexer only when `TREELOOM_FEATURE_SKIP_PATTERNS=1` is set (feature-flagged in v1); otherwise they're informational.
- **Embedding backends live in Postgres** (`embedding_backends` table, migration `007_embedding_backends.sql`). The `.env` vars `EMBEDDING_URLS` / `EMBEDDING_FALLBACK_URLS` are seeded into the table on first startup (when the table is empty) and ignored thereafter — the registry is authoritative. Mutations to the table emit `NOTIFY embedding_backends_changed`; each worker holds a dedicated asyncpg connection (NOT from the pool) that LISTENs on that channel and calls `EmbeddingProxy.reload()`. Reload carries over per-URL `in_flight_tokens`/`failures` counters by reusing the same `Backend` instance. Empty-registry reloads keep the last good config + log a warning. Admin-only REST CRUD lives at `/embedding-backends` (GET/POST/DELETE).
- **Per-chunk summary cache** lives in Postgres (`summary_cache` table, migration `005_summary_cache.sql`, keyed by `SHA1(chunk_text) + LLM_MODEL + PROMPT_VERSION`). Shared across indexer workers so horizontal scale doesn't re-pay for the same chunk summarizations. Bump `PROMPT_VERSION` in `llm_adapter.py` when prompts change — old rows stay in the table but are never read again. Like the JobStore, the cache helpers fail loud (`RuntimeError`) if the Postgres pool is unavailable.
- **Indexing is dedup'd by `source_id`**. POSTs to `/index-repo` / `/index-directory` / `/index-file` short-circuit to the prior DONE job if the same source is already indexed. Pass `force=true` to actually re-index. Source-id is derived from `(path|url) + branch` via `make_source_id`.
- **At most one active job per `source_id`**. The endpoints check `JobStore.find_active_for_source(source_id)` and refuse a new submission if a `queued` or `running` job exists for the same source — `force=true` does NOT bypass this (it only overrides the DONE-dedup short-circuit). A partial unique index `jobs_active_source_uniq` (migration `008_jobs_active_source_uniq.sql`) enforces the invariant at the DB level too, so a race past the app-level check still can't create duplicates. Webhook full-index jobs use the same guard.
- **Job state lives in Postgres** (`jobs` table, migration `004_jobs.sql`). `DATABASE_URL` must be set and reachable — the indexer fails loud on startup otherwise. On restart, RUNNING / QUEUED jobs are dispatched to the right runner by their persisted `kind`, and runners skip the first `committed_files` files instead of calling `_reset_source` — so a crash mid-index resumes where it left off without duplicating Milvus chunks. `committed_files` advances only when chunks are actually flushed to Milvus. Incremental (webhook) jobs persist their `changed_files` list in `jobs.payload` JSONB (migration `017`), so they can be picked up by any worker and retried; on restart RUNNING incremental jobs are re-queued and retried up to `MAX_JOB_ATTEMPTS` times before dead-lettering.
- **Two load-test harnesses for the webhook path.** `scripts/loadtest_incremental.py` drives *synthetic* file paths (good for queue/backpressure/memory stress, but the paths don't exist in the clone so the `merge_to_searchable` histogram records *time-to-fail*, not real latency). `scripts/loadtest_merge_searchable.py` (closes criterion 3, `docs/incremental-indexing.md` §10) measures the **real** merge→searchable: it stands up N throwaway git repos, serves them over **`git://` via `git daemon`**, makes real commits, fires signed inline webhooks (`WEBHOOK_TRUST_INLINE_FILES=1`), and polls `/search` for a unique probe token. Three traps it encodes: (1) the incremental clone uses `GIT_ALLOW_PROTOCOL=http:https:ssh:git`, so a bare **local path is rejected** (`transport 'file' not allowed`) — use `git://`, not a directory; (2) an isolated indexer on a fresh DB re-seeds `embedding_backends` from `.env`'s `EMBEDDING_URLS` — if those point at offline backends, every `embed()` raises and jobs finish 0-chunks; override `EMBEDDING_URLS=http://localhost:8082`; (3) the webhook matches a source by **exact `url == repo_url`** (index fixtures with `{"url": ...}`), and one-active-job-per-`source_id` means sustained concurrency needs **N distinct repos**, not one.
- **Job queue is Postgres-backed** (`job_queue` table, migration `006_job_queue.sql`). Workers pop with `DELETE ... WHERE job_id = (SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING job_id` so N indexer processes can share a queue without double-running a job. The queue carries only `job_id`s; workers re-fetch the Job row from `JobStore` and reconstruct the runner coroutine via `dispatch_job(job)` in `application/indexer_runners.py`. Incremental (webhook) jobs now go through the same Postgres queue — their `changed_files` list is persisted in `jobs.payload` JSONB (migration `017_jobs_payload.sql`) so workers can pick them up cross-process. Crash-recovery: if a worker dies mid-run, the queue row is already gone (it was DELETEd at pop time) but the JobStore status stays `RUNNING`, and the next indexer-process startup re-enqueues it; incremental jobs marked RUNNING on startup are re-queued (not resumed, since the payload is persisted). Polling interval `JOB_QUEUE_POLL_INTERVAL` (default 1.0s); per-process worker count `INDEX_CONCURRENCY` (default 8). Incremental jobs are **retryable** (`_RETRYABLE_KINDS`): on failure the worker increments `jobs.attempts` and re-enqueues up to `MAX_JOB_ATTEMPTS` (default 3) times; beyond that the job enters `DEAD_LETTER` status and `treeloom_dead_letter_jobs_total` is incremented. Backpressure: if queue depth ≥ `MAX_QUEUE_DEPTH` (default 500) the webhook returns 429 with `Retry-After: 30` and increments `treeloom_webhook_shed_total`. Oversized payloads (`len(changed_files) > MAX_CHANGED_FILES`, default 5000) fall back to a full repo re-index job instead.
- **`SourceRecord` and the `Source` Neo4j node carry `commit_sha`** captured at index time. `GET /sources/{id}/staleness` resolves current HEAD via `git rev-parse` (local path) or `git ls-remote` (URL) and returns `{indexed_sha, current_sha, is_stale}`. `is_stale` is `null` for non-git inputs.
- **Fleet operations (`docs/fleet-operations.md`)** — multi-repo ops at hundreds of sources, all built on the per-source primitives (no new persistence). **Bulk onboarding**: `python -m treeloom.fleet onboard <manifest.yaml> [--force --wait --dry-run]` POSTs `/index-repo` per entry; idempotency is free (the DONE-dedup short-circuit), so re-running an unchanged manifest enqueues nothing. Manifest parsing is pure in `domain/sources/manifest.py` (`parse_manifest` → `ManifestEntry`; YAML or JSON via `yaml.safe_load`; optional top-level `defaults:` merged into each `repos:` entry or a bare list; exactly one of `path`/`url`; unknown keys rejected — URL safety stays server-side). **Fleet health rollup**: `GET /fleet` (+ `?summary_only` / `?staleness` / `?entities`) and `python -m treeloom.fleet health` answer "is the fleet healthy?" in one call — joins the SourceRecord registry with the latest job per `source_id` (last-error) via the pure `application/fleet.py:build_fleet_health`. Cheap by default; `?staleness=true` does live git HEAD resolution **capped at `FLEET_STALENESS_MAX_SOURCES`** (default 200, sets `summary.staleness_truncated` above it — use the freshness gauges at scale); `?entities=true` runs ONE aggregate `graph_store.entity_counts_by_source()` query (added to both neo4j+sqlite stores) and is absent-not-fatal if the graph store is down. Auth/visibility mirror `/sources`. **Scheduled refresh**: an in-process loop (mirrors the freshness sampler), **default OFF** via `FLEET_AUTO_REFRESH_ENABLED`, with `FLEET_REFRESH_INTERVAL_SECONDS` (900), `FLEET_REFRESH_MAX_PER_TICK` (25 enqueues/tick) and `FLEET_REFRESH_MAX_SOURCES` (200 staleness-checks/tick — above it the loop checks a rotating window per tick instead of firing hundreds of serial `git ls-remote`); `select_sources_to_refresh` (pure) picks stale-git-only, oldest-first, capped, skipping sources with an active job; each enqueue bumps `treeloom_fleet_refresh_enqueued_total`. Non-git/directory sources (staleness `null`) are never auto-refreshed. Re-index is **kind-faithful**: a source is re-enqueued as the same kind it was first indexed (`repo`/`directory`/`file`) via the new `source_records.kind` column (migration `018`, persisted in `_finalize_job`; a graph-only re-pass preserves the existing kind) — `resolve_reindex_kind` (pure) trusts the stored kind and falls back to filesystem shape for legacy rows (a file path → a `file` job, never a `repo` job that would fail the runner's `isdir` check). Simple mode is single-writer — not a fleet-scale tier.
- **Auth model**: when `AUTH_ENABLED=true`, every endpoint except `/health`, `/status`, and `GET /jobs` requires `Authorization: Bearer <api_key>` — including `/search` and `/graph-explore`, which bypass the auth middleware but authenticate and authorize per repo in the endpoint (401 anonymous unless `TREELOOM_SEARCH_OPEN=1`). `GET /sources` and `GET /sources/{id}/staleness` additionally filter results to the caller's own records (`created_by == user.id`); admins see all. Sources with no `created_by` are treated as legacy and remain visible. Git URLs and branches passed to `/index-repo` are validated against argument-injection (leading `-`, `::` smuggling) before any `git clone` call; clones run with `GIT_ALLOW_PROTOCOL=http:https:ssh:git` and gitpython's `allow_unsafe_protocols=False`.
- **Local auth (`docs/authz.md`)**: username/password login issues an HttpOnly `treeloom_session` cookie backed by the `sessions` table (TTL via `TREELOOM_SESSION_TTL_HOURS`, Secure flag via `TREELOOM_COOKIE_SECURE`). Two scoped token tables: `personal_access_tokens` (user-owned PATs, scopes `search`/`index`/`admin`, capped at the owner's role) and `api_keys` (service/admin-owned, with `all_access`). Auth resolution order: session cookie → PAT → API key → legacy `users.api_key_hash` (still works; migrated into a `legacy` PAT row by migration `016`). First admin seeded via `TREELOOM_BOOTSTRAP_ADMIN_USER`/`TREELOOM_BOOTSTRAP_ADMIN_PASSWORD` when users table is empty. `_require_scope(request, scope)` gates `DELETE /sources/{id}` and `POST /index-graph` (scope `index`); the remaining mutating index endpoints authenticate but don't check a scope yet. `POST /build-community` is admin-only (it is global and unscoped, so there is no per-source ACL to apply). There is NO admin-role bypass in it (that would make per-token scopes meaningless for admin-owned tokens); admins pass because full-role auth paths populate every scope from `scopes_for_role`. Managed from dashboard Users & Groups / Access Tokens tabs. Migration `016_local_auth_tokens.sql`.
- **Per-repo search authorization (`docs/authz.md`)** — read endpoints (`/search`, `/graph-explore`, `/find-*`) enforce **scope-level rejection**, not per-hit filtering. A `source_id`-pinned query is `403`'d unless the caller is authorized for that source; a shared/cross-repo query (no `source_id`) is `403`'d unless the caller has `all_access`, with their denied sources applied as a `source_id NOT IN (...)` prefilter. The decision (`domain/authorization/access.py`, pure + unit-tested) is **admin > deny > all_access > allow > owner(created_by) > deny** over the caller's principals (`("user",id)` + each `("group",gid)`). Grants are `source_grants(principal_type, principal_id, source_id, effect∈{allow,deny})`; groups + `all_access` (on `users` and `groups`) are the **community-tier** ACL engine — managed via admin REST (`/groups*`, `/sources/{id}/grants`, `PATCH /users/{id}`). Enforcement runs at the `_authorize_scope` choke point in `application/indexer_authz.py` and is **off unless `AUTH_ENABLED=true`**; `TREELOOM_SEARCH_OPEN=1` keeps search anonymous even with auth on (single-tier escape hatch). The graph-neighbor leak is closed: neo4j+sqlite `get_neighbors_batch`/`traverse`/`find_callers|callees|references` take `source_id`/`exclude_source_ids` and scope via the `(:Source)-[:CONTAINS]->` / `source_entities` join. **`VECTOR_STORE=chromadb` fails closed** on shared-index exclusions (can't express `NOT IN`). Every decision is logged best-effort to `search_audit` (admin `GET /audit/search`). **The MCP credential policy is a two-way split by tool kind** (`application/mcp_server.py`): write/admin tools call `_tool_auth_headers(ctx)` — precedence caller bearer → `TREELOOM_MCP_TOKEN` (the caller's own credential via the environment, the stdio model) → the gated service account (`TREELOOM_MCP_API_KEY`, only when `TREELOOM_MCP_SERVICE_ACCOUNT=true`) → nothing. The nine per-repo-scoped read tools (`search_code`, `search_code_enhanced`, `hydrate_chunks`, `explain_code`, `graph_explore`, `find_definition`, `find_callers`, `find_references`, `list_job_errors`) call the distinct `_read_tool_auth_headers(ctx)` instead — same precedence MINUS the service-account step, by design: it must **never** reach the shared service account, because per-repo authorization is enforced at the indexer against the caller's own `request.state.user`, and letting a read fall back to a shared key would grant every caller on that transport the service account's cross-repo visibility instead of their own scoped grants — the confused-deputy class the write-path policy exists to prevent for writes, reopened on the read path if the two policies were ever merged. `TREELOOM_MCP_TOKEN` is safe on the read path because it represents the caller's own credential, not a shared one, so per-repo scoping still applies to it. Do not "simplify" `_read_tool_auth_headers` to call `_tool_auth_headers` — the parametrized regression guard for this is `test_read_tool_never_uses_service_account_even_when_opted_in` in `tests/unit/test_mcp_credential_forwarding.py`, covering all nine read tools (a first implementation pass routed all nine through the write policy and passed every test that existed then, because only `search_code` had a guard). The MCP transport also validates `Origin` (`application/mcp_origin_guard.py`, MCP-spec MUST) and, on the opt-in HTTP+SSE transport, is published on `127.0.0.1` only. Migrations `012`–`015`.
- **OpenTelemetry tracing** (`src/treeloom/infrastructure/tracing.py`). The indexer initializes a TracerProvider in its startup hook (before `validate_config()`) and auto-instruments FastAPI, HTTPX, AsyncPG, and stdlib logging. **`index_file` is the root span — one trace per file.** Previously the four `_run_index_*_job` runners wrapped their per-file loops in a single `index_job` span; that produced one giant trace per job that blew Tempo's `max_bytes_per_trace` (5MB default) on any non-tiny repo. The runners now emit only a tiny `job.failed` span on fatal exceptions; per-file `index_file` spans carry `treeloom.job_id` / `treeloom.source_id` / `treeloom.kind` so all files in one job are still discoverable via TraceQL `span.treeloom.job_id = "<id>"`. Inner spans (`parse_and_chunk`, `embed.chunks`, `summaries.*`, `embed.summaries`, `milvus.init_collection`, `milvus.insert_chunks`, `graph.index_file`) hang under `index_file`. Batch flush spans in `_run_index_directory_job` (`embed.chunks` / `milvus.insert_chunks` after `MAX_BATCH_SIZE` is reached) are intentionally separate root traces since they span multiple files. Four hand-traced adapters: `neo4j.<fn>` via `_retry_on_disconnect`, `milvus.<method>` via `_execute_with_reconnect`, `tei.embed` in `embedding_proxy._post_one`, `llm.chat` in `llm_adapter._chat`. Attribute names are stable and use the `treeloom.` prefix — renaming any breaks the Grafana dashboard (`assets/grafana/treeloom-indexer-traces.json`). Per-chunk LLM summary spans gated behind `OTEL_TRACE_PER_CHUNK=true` (off by default). For local dev the bundled collector and `.env.example` default `OTEL_TRACES_SAMPLER=always_on` and skip tail sampling; for shipping to a central collector switch to `parentbased_traceidratio` at 0.1 (tail sampling lives in the central collector). structlog event dicts include `otelTraceID` / `otelSpanID` via a processor in `infrastructure/logging.py`. To turn tracing off entirely, set `OTEL_SDK_DISABLED=true`.

## Prompt enhancer

`prompt_enhancer.py` generates query variants from Neo4j entities for benchmark ablation. All 6 strategies (`docstring`, `namespaced`, `entity_context`, `intent`, `cross_cutting`, `problem_driven`) are deterministic — no LLM calls. `prompt_enhancer.py` is currently standalone (the in-repo `benchmark queries` generator does not invoke it); the `strategy` field on each query JSONL row is used for `--strategy` filtering in the `run`/`agentic` subcommands. In ablations, `entity_context` was the best strategy at 0.80 R@1.
