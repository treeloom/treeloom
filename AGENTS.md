# Treeloom Agent Guidance

CLAUDE.md is the canonical, detailed guidance for AI agents working in this
repo — read it first. The short version:

- **DDD architecture** — `domain/` (pure logic + ports), `adapters/` (TEI,
  Milvus, Neo4j, PostgreSQL, rerankers), `application/` (FastAPI/MCP wiring),
  `infrastructure/` (config, DI). Domain NEVER imports
  adapters.
- **Two-tier testing** — Detroit-style unit tests (`pytest tests/unit`, no
  services needed; mock only external services, never internal classes) +
  the retrieval-quality benchmark harness (`python -m treeloom.benchmark`).
- **Evidence-based gates** — retrieval/behavior changes ship only when the
  benchmark harness shows they help (see docs/benchmark-eval.md).
- **Split deployment** — indexer on the host (port 8001) owns all data
  access; the MCP server is a pure HTTP proxy to it — stdio by default
  (`treeloom-mcp`, spawned by the MCP client, no container), or the
  deprecated HTTP+SSE container on port 8000 behind `COMPOSE_PROFILES=http-mcp`.
- **Verification-first** — state how you'll verify before starting; run that
  verification before claiming done.
