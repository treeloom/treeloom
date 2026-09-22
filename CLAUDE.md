# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

- **Verification-first.** Before starting any task, state up front how you will
  verify it — the concrete check: the exact command to run, the test to add/run,
  the endpoint to hit, the output or metric to inspect. After finishing, actually
  run that verification and report the results, including failures, skipped steps,
  and partial outcomes. "Done" means *verified*, not "should work" — if you can't
  verify, say so explicitly and explain why.

## What this is

Treeloom is a GraphRAG system for code search: tree-sitter parsing, chonkie chunking, HuggingFace TEI embeddings, Milvus vectors, a Neo4j code graph, and an MCP server (stdio by default). The reference material that used to live here — how to run the stack, index repos, benchmark, the job queue, fleet ops, auth, tracing — is in `docs/engineering-notes.md` (start there) and the topic docs under `docs/`. Local `treeloom-*` skills cover the same ground by topic.

## Tests

```bash
env -u PYTHONPATH python -m pytest tests/unit -q
```

Unit tests need no services. The `env -u PYTHONPATH` is required: a system `/opt/ros` entry on `PYTHONPATH` otherwise hijacks pytest plugin autoload and dies on `ModuleNotFoundError: lark`. Retrieval *quality* is validated by the benchmark harness, never by unit tests.

## Invariants — do not violate without reading the linked notes

- **tree-sitter is unpinned** — `tree-sitter-language-pack` (bundled via `chonkie[code]`) manages parsers. Do NOT add `tree-sitter-languages` or pin `tree-sitter` below 0.24.
- **`_make_chunker` must keep overwriting `chunker.parser`** with `tree_sitter.Parser(get_language(lang))` (`adapters/tree_sitter/indexer.py`). Removing that line silently sends every file down the TokenChunker/character fallback. Chunkers are built per call — parsers are not safe for concurrent `parse()`.
- **`.md`/`.markdown`/`.mdx` stay in `MARKDOWN_MAP`, never `LANGUAGE_MAP`** — `tree-sitter-markdown`'s grammar asserts and crashes the indexer. Markdown chunks via `_make_markdown_chunker`, not `CodeChunker`.
- **`LANGUAGE_QUERIES` keys must equal `LANGUAGE_MAP` values** (`graph_extractor.py`); only `class`/`function`/`import`/`call` are consumed. A mismatched key fails silently (Module-only entity, no graph signals) — this bit twice (`"c_sharp"` vs `"csharp"`, rust's `"struct"`). Never put quantified captures like `(base_list (identifier) @x)?` inside a `@class.def` pattern — they desync the paired name/def lists.
- **Never import `adapters/milvus/vector_store` or `adapters/neo4j/graph_store` from startup-path code** — both `require_env` their service vars at import and crash simple mode. Go through the root shims (`treeloom/retriever.py`, `graph_store.py`, `embedder.py`, `reranker.py`); `tests/unit/test_simple_profile.py` guards this.
- **The MCP server is a pure HTTP proxy** — every tool calls the indexer via `INDEXER_URL`; it imports no Milvus/Neo4j/TEI client code. All filesystem, indexing, and retrieval work lives in `indexer_service.py` (host, port 8001).
- **`store_graph` uses `MERGE` for source, target, AND the `Source` node in the entity-upsert block.** Don't switch any to `MATCH`: the `Source` node is only enriched after every file is processed, so a `MATCH` mid-job finds nothing and silently drops every `Entity`.
- **Neo4j driver is 6.x async** — use `result.fetch(n)` with an integer, not bare `fetch()`.
- **Milvus collection has an explicit schema** (`vector`, nullable `summary_vector`, server-side BM25 `sparse_vector`). Schema-changing edits require dropping the collection and re-indexing every source.
- **Don't mount `/home` into Docker** — use `host.docker.internal` (already set up via `extra_hosts`). The host indexer covers filesystem access.
- **Any new search path must run results through `chunk_hit_from_milvus`** (`application/retrieval.py`) — the MCP layer does `ChunkHit(**c)` and 500s on raw Milvus hits missing `file_path`/`snippet`.
- **Do NOT set `TREELOOM_MCP_TRANSPORT=sse` in a shared `.env`** — `src/treeloom/__init__.py` loads the repo `.env` from any cwd, so it would turn the `treeloom-mcp` stdio script into a second, colliding SSE server. The HTTP+SSE container is toggled only by `COMPOSE_PROFILES=http-mcp`.
- **Reranker is chosen by `RERANKER_PROVIDER` + `RERANKER_URL` at indexer startup** — restart to switch. Dead-reranker failure mode: if the URL points at a down service, scores fall back flat (all ~1.0) and recall silently craters — smoke-test `/rerank` after switching and before any benchmark.
- **Graph rescoring stays ON by default (`USE_GRAPH_SCORING=1`)**; the neighbor/community payload is always on and is never suppressed by any flag. Set `USE_GRAPH_SCORING=0` only with a Cohere-class reranker, where it is redundant.
- **Don't trim the search payload.** Every payload-shrinking idea (facet, summary_tail, community-off, adaptive top-k, strip-imports, batch-hydrate, combinations) lost or washed agentically because of compensatory fetch — the agent re-queries and mean turns rise. Ship such ideas opt-in only, never as defaults; the governing metric is mean turns judged with tokens.
- **Do not "simplify" `_read_tool_auth_headers` to call `_tool_auth_headers`** (`application/mcp_server.py`). The nine per-repo read tools must never fall back to the shared service account (`TREELOOM_MCP_API_KEY`), or every caller inherits its cross-repo visibility. Regression guard: `test_read_tool_never_uses_service_account_even_when_opted_in` in `tests/unit/test_mcp_credential_forwarding.py`.
- **Before any long benchmark or agentic run, verify the actually-served LLM, reranker, and embedding models and their GPU/CPU/API placement** — `.env` and `nvidia-smi` both lie. Pass `scripts/agentic_smoke.sh` before any paid agentic cell, and never mix pre- and post-2026-07-05 agentic numbers in a rollup.

## Where things are documented

- `docs/engineering-notes.md` — everything formerly in this file: running the stack, simple mode, indexing and dev tasks, benchmark CLI and verdicts, jobs/queue, fleet, auth/authz, tracing, prompt enhancer.
- `docs/benchmark-eval.md`, `docs/benchmark-findings.md`, `docs/benchmark-results-2026-09.md`, `docs/token-optimization-rejected.md` — evidence behind the benchmark invariants above.
- `docs/simple-mode.md`, `docs/authz.md`, `docs/fleet-operations.md`, `docs/incremental-indexing.md`, `docs/observability-runbook.md`, `docs/result-provenance.md` — topic runbooks.
