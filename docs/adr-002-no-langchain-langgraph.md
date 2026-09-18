# ADR-002: No LangChain / LangGraph

- Status: Accepted
- Date: 2026-05-29
- Owner: indexer / MCP server

> An ADR records a decision at a point in time. File paths below are as they
> were on the date above; the root-level `src/treeloom/*.py` modules named here
> are now thin import shims over `src/treeloom/application/` and
> `src/treeloom/adapters/`. The decision stands: `pyproject.toml` declares no
> LangChain or LangGraph dependency.

## Context

Treeloom's stack contains several pieces that, on the surface, look like the
exact shapes LangChain and LangGraph are built for:

- An LLM adapter for HyDE query expansion and per-chunk summary generation
  (`src/treeloom/adapters/llm_api/llm_adapter.py`).
- An embedding client over TEI (HuggingFace Text Embeddings Inference)
  (`src/treeloom/embedder.py`, `src/treeloom/adapters/tei/embedding_proxy.py`).
- A vector store (Milvus) accessed through a thin wrapper
  (`src/treeloom/retriever.py`).
- A reranker call (also TEI).
- A multi-step retrieval pipeline in `search_code` /
  `search_code_enhanced`: embed → Milvus search → graph traverse → community
  summaries → rerank → post-rerank graph-signal weighting → format.

A reasonable question is whether replacing these with LangChain primitives
(LLMs, Embeddings, VectorStore, Retriever, Reranker) and/or LangGraph
(orchestrating the search pipeline as an explicit graph) would reduce custom
code and improve maintainability.

## Decision

We do not adopt LangChain or LangGraph. Adapters stay in-tree under
`src/treeloom/adapters/` and orchestration stays as direct async calls in
`mcp_server.py` and `indexer_service.py` *[2026-09: now
`src/treeloom/application/retrieval.py`, called by the indexer's `/search`
endpoint; the MCP server is a pure HTTP proxy]*.

## Rejected alternatives

### LangChain for the adapter layer (LLM / Embeddings / VectorStore / Reranker)

- The adapters LangChain would replace are already small and intentionally
  tuned to constraints LangChain abstractions hide:
  - TEI caps embedding batches at 32 (the proxy sends up to 128 and splits
    when TEI rejects the batch) and reranks at ≤20 inputs with
    `raw_scores: True` for cross-batch comparability (see `CLAUDE.md` and
    `retriever.py`).
  - The embedding path is cost-aware token-weighted routing across multiple
    TEI backends with a GPU-OOM pin against `GPU_MAX_TOKENS`
    (ADR-001). LangChain's `Embeddings` interface has no concept of
    per-backend in-flight token accounting.
  - Milvus access uses an explicit hybrid schema (`vector`, `summary_vector`,
    `sparse_vector` BM25) combined via `hybrid_search` with `RRFRanker(k=RRF_K)`
    — LangChain's `VectorStore` abstraction does not model multi-vector
    hybrid retrieval with server-side BM25 in a way we could use without
    immediately dropping back to the raw `pymilvus` client.
  - The LLM adapter participates in a Postgres-backed `summary_cache` keyed
    on `SHA1(chunk_text) + LLM_MODEL + PROMPT_VERSION`; that cache is
    shared across worker processes and is the throughput-defining piece of
    the indexer. Wrapping the call site in a LangChain `LLM` either hides
    the cache from the framework or duplicates it.
- LangChain abstractions are leaky in the directions we care about
  (batching, retry, observability attribute names, error taxonomy). Our
  hand-traced OTel spans (`tei.embed`, `llm.chat`, `milvus.<method>`,
  `neo4j.<fn>`) use stable `treeloom.*` attribute names that the Grafana
  dashboard depends on; LangChain's own instrumentation does not match
  that contract.

### LangGraph for the search pipeline

- The pipeline is linear (embed → search → traverse → communities → rerank
  → score-combine → format) with one conditional branch
  (`search_code_enhanced` does a second filtered Milvus search before
  rerank). Modeling it as a `StateGraph` adds boilerplate without enabling
  anything we currently want: no human-in-the-loop, no tool-loop, no
  branching on intermediate results, no checkpointing of partial state.
- The current implementation has a useful property worth preserving: a
  reader can `grep` a tool name in `mcp_server.py` and see the entire
  query path inline. Splitting it across LangGraph nodes trades that for
  a graph definition plus N node functions.
- The benchmark harness (`python -m treeloom.benchmark`, code under
  `src/treeloom/application/benchmark/`) exercises the pipeline as a
  function call. A LangGraph node graph adds a serialization step (state
  dicts in/out of nodes) that the benchmark does not need.

### Partial adoption (e.g., LangChain just for the LLM)

- Single-purpose adoption pulls in the full transitive dep surface for one
  call site. `llm_adapter._chat` is ~one function plus a Postgres cache
  lookup; the cost-benefit does not work out.

We may revisit if we start adding genuinely agentic behavior (tool-using
retrieval agents, multi-turn query refinement with branching, planner /
executor loops). Those are LangGraph's home ground; today's pipeline is not.

## Consequences

### Positive

- Adapters stay small, directly testable, and aligned with the actual
  backend constraints (TEI batch caps, Milvus hybrid schema, Postgres
  summary cache, cost-aware routing).
- OTel attribute names and the Grafana dashboard contract stay under our
  control.
- The query path remains greppable from one file *[2026-09: that file is now
  `application/retrieval.py`]*.
- No additional transitive dependency surface (LangChain pulls in a large
  ecosystem we would otherwise not load).

### Negative

- We own and maintain the adapter code. New backends (e.g. a hosted
  embedding provider with rate limits, a different vector store) are an
  in-tree change rather than swapping a LangChain class.
- Engineers familiar with LangChain idioms will not find them here and
  have to read the actual call sites instead.
- If the search pipeline grows real branching or agentic loops, this ADR
  should be revisited — not LangChain at the adapter layer, but LangGraph
  for orchestration is the more plausible second look.

## References

- ADR-001 — cost-aware embedding proxy (the kind of backend-specific
  routing LangChain's `Embeddings` interface does not express).
- `src/treeloom/application/mcp_server.py` — the MCP tools (each a plain HTTP
  call to the indexer); the search pipeline itself is
  `src/treeloom/application/retrieval.py`.
- `src/treeloom/retriever.py` — Milvus hybrid search and reranker batching.
- `src/treeloom/adapters/llm_api/llm_adapter.py` — LLM call site and Postgres summary cache.
