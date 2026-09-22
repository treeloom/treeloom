<p align="center">
  <img src="assets/branding/treeloom-logo-horizontal.svg" alt="Treeloom" width="420">
</p>

<p align="center"><strong>Parse Trees · Weave Graphs</strong></p>

<p align="center">
  <a href="https://github.com/treeloom/treeloom"><img src="https://img.shields.io/badge/github-treeloom-3fb950?logo=github" alt="GitHub"></a>
  <a href="https://github.com/treeloom/treeloom/actions/workflows/ci.yml"><img src="https://github.com/treeloom/treeloom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11+-3fb950?logo=python" alt="Python 3.11+"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License MIT"></a>
</p>

---

Treeloom is a **GraphRAG code search engine** that understands code structurally,
not as flat text. It parses every file into an abstract syntax tree via
[tree-sitter](https://tree-sitter.github.io/), then weaves those parsed
pieces into a connected graph in [Neo4j](https://neo4j.com/), where calls,
imports, inheritance, and references become threads connecting the nodes.

Search traverses that woven structure — **semantically** through vector
embeddings and **structurally** through the graph — instead of grepping
flat text. In a September 2026 benchmark on a pinned production codebase, across
three current agent models, an agent using treeloom matched the answer quality of a
grep/glob/read agent at **11–47% lower cost** — with the largest savings on the
questions where you don't know the name of what you're looking for. See
[Benchmarks](#benchmarks).

## How It Works

```
Source Code  ──►  tree-sitter  ──►  Abstract Syntax Trees
                                          │
                                   (parse the trees)
                                          │
                                          ▼
                                    Neo4j Graph DB
                                          │
                                   (weave the graph)
                                          │
                                          ▼
                              Milvus Vector Store  ──►  Search API
```

1. **Parse**: tree-sitter parses every file into AST chunks — functions,
   classes, blocks, with their hierarchical relationships preserved
2. **Embed**: Text chunks are embedded via TEI (Jina, BGE, etc.) into
   Milvus vector store
3. **Weave**: Neo4j captures calls, imports, inheritance, and references
   as graph edges between parsed nodes
4. **Search**: Hybrid retrieval combines dense vector search, BM25
   keyword matching (Milvus and LanceDB), graph-enhanced scoring, and HyDE query expansion

### What is HyDE?

**Hy**pothetical **D**ocument **E**mbeddings (Gao et al., 2022). Instead of
embedding a raw query like "how does auth middleware work?" — which may share
little vocabulary with the actual code — Treeloom asks the LLM to *hallucinate*
a short code snippet that would answer the question, then embeds *that*. The
hypothetical code's embedding is far closer to real matching code in vector
space than the raw query would be. It's query expansion via imagination.

## Quick Start

### Simple mode (no GPU, one API key)

Evaluating on a laptop? The simple profile swaps Milvus → embedded LanceDB,
Neo4j → embedded SQLite, and TEI/local-LLM → the OpenAI API (embeddings +
HyDE/summaries), with reranking on the CPU. One Docker container (Postgres) by default.

```bash
git clone https://github.com/treeloom/treeloom.git && cd treeloom
cp .env.simple.example .env   # then paste your OPENAI_API_KEY into it
./run-simple.sh
```

See [docs/simple-mode.md](docs/simple-mode.md) for the full walkthrough,
cost expectations, and the capability matrix vs the full stack.

### Full stack

Prerequisites: Docker (with compose), Python 3.11+, and an
OpenAI-compatible LLM endpoint for HyDE/summaries — the `.env.example`
default expects [Ollama](https://ollama.com) on the host
(`ollama pull qwen2.5-coder:7b`); if the LLM is unreachable, indexing and
search still work, just without HyDE and chunk summaries.

```bash
git clone https://github.com/treeloom/treeloom.git
cd treeloom
cp .env.example .env   # quickstart defaults: everything local

# One-shot: pip install -e ., build mcp-server (image only — not started by
# default), bring up Postgres + TEI + Neo4j + Milvus
# (COMPOSE_PROFILES=local-infra), and run the host-side indexer in the
# foreground on :8001. MCP access is stdio (treeloom-mcp) by default, no
# container; add http-mcp to COMPOSE_PROFILES for the deprecated HTTP+SSE
# transport.
./run.sh
```

No NVIDIA GPU? Add `cpu` to `COMPOSE_PROFILES` in `.env` and set
`EMBEDDING_URL=http://localhost:8083` — embedding runs on the CPU TEI
image (slower indexing, identical results). Already running Neo4j/Milvus
elsewhere? Drop `local-infra` from `COMPOSE_PROFILES` and point
`NEO4J_URI` / `MILVUS_URI` at your instances.

If you'd rather start the pieces by hand:

```bash
pip install -e .

# Docker side: Postgres, TEI embedding + reranker, Neo4j, Milvus
docker compose up -d postgres tei-embedding tei-reranker neo4j milvus

# Host side: the indexer must NOT run in Docker — it reads arbitrary
# user paths off the host filesystem.
uvicorn treeloom.indexer_service:app --host 127.0.0.1 --port 8001
```

> The MCP server is **not** a compose service by default: it runs as a stdio
> subprocess spawned by your MCP client (`treeloom-mcp`). The deprecated
> HTTP+SSE transport is available with `COMPOSE_PROFILES=http-mcp`.

Prebuilt images for the Treeloom services (`treeloom/indexer`,
`treeloom/mcp-server`, `treeloom/ui`, `treeloom/qwen3-reranker`) are published
to Docker Hub on every release; `docker compose pull` fetches them instead of
building. See [docs/docker-images.md](docs/docker-images.md) for tags,
platforms, and the release procedure.

Point your MCP client at the `treeloom-mcp` console script — it reaches the
host indexer directly via `INDEXER_URL` (no `host.docker.internal` needed on
stdio):

```json
{
  "mcpServers": {
    "treeloom": {
      "command": "treeloom-mcp",
      "env": {
        "INDEXER_URL": "http://localhost:8001",
        "TREELOOM_MCP_TOKEN": "<your treeloom PAT or API key>"
      }
    }
  }
}
```

Once everything is up:

```bash
# Index a repo (host indexer, :8001)
curl -X POST http://localhost:8001/index-directory \
  -H 'Content-Type: application/json' \
  -d '{"directory": "/path/to/your/repo", "pattern": "**/*"}'

# Search (host indexer, :8001). A scope is required: path_prefix for one
# repo, or "cross_repo": true to deliberately search the whole index.
curl -X POST http://localhost:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does auth middleware work?", "path_prefix": "/path/to/your/repo"}'
```

Before kicking off a long index, run preflight to predict wall time and
get tuning hints:

```bash
python -m treeloom.preflight /path/to/repo
# or hit the endpoint. A `path` scan is refused (403) unless it sits under
# PREFLIGHT_ALLOWED_ROOTS (colon-separated) in the indexer's environment:
curl -X POST http://localhost:8001/preflight \
  -H 'Content-Type: application/json' \
  -d '{"path": "/path/to/repo"}' | python -m json.tool
```

Operator UI: the `ui/` standalone SPA (see `ui/README.md`) gives you Jobs,
Sources, and Submit tabs against the indexer. Run it with `npm install &&
npm run dev` from `ui/`. When `AUTH_ENABLED=true`, log in with a bearer
token (personal access token or API key); it's stored in localStorage.

When `AUTH_ENABLED=true`, the indexing endpoints (`/index-*`),
`GET /sources`, and search (`/search`, `/graph-explore`) require
`-H "Authorization: Bearer $TREELOOM_KEY"`; search is also checked against
per-repo grants. `/health`, `/status`, and `GET /jobs` stay open (job views are
redacted for anonymous callers). Set `TREELOOM_SEARCH_OPEN=1` to keep `/search`
anonymous while indexing stays protected (`/graph-explore` still needs a token).

## Architecture

```
┌─────────────────────────────────────────────┐
│                  MCP Server                  │
│             (stdio, per client)              │
│          Pure API consumer — zero DB access   │
└──────────────┬──────────────────────────────┘
               │ HTTP
┌──────────────▼──────────────────────────────┐
│               Indexer Service                │
│              (Host :8001)                    │
│    Owns all data access: Milvus, Neo4j, TEI, │
│    PostgreSQL, tree-sitter, embedding        │
└──────┬──────────────┬───────────┬───────────┘
       │              │           │
  ┌────▼────┐   ┌─────▼─────┐  ┌─▼──────────┐
  │ Milvus  │   │   Neo4j   │  │ PostgreSQL  │
  │(vector) │   │  (graph)  │  │ (persistence)│
  └─────────┘   └───────────┘  └─────────────┘
```

**DDD layers**: `domain/` (pure logic) → `adapters/` (I/O implementations) →
`application/` (FastAPI wiring) → `infrastructure/` (config, DI).
Domain never imports adapters.

## Features

- **Structural parsing**: tree-sitter AST chunking preserves function/class/block hierarchy
- **Graph weaving**: Neo4j captures calls, imports, inheritance, references
- **Hybrid search**: Dense vectors + BM25 + graph-enhanced scoring + HyDE \
  *(BM25 available with Milvus and LanceDB; ChromaDB uses embedding-only hybrid search)*
- **Dual GPU/CPU embedding**: Token-cost-aware routing keeps oversized chunks off the GPU, falls back to CPU
- **Multi-vector backend**: Milvus (default), LanceDB, ChromaDB — selected with `VECTOR_STORE` \
  *(non-default backends are install extras: `pip install -e .[lancedb]`, `.[chromadb]`, or `.[all-backends]`)*
- **Community detection**: Louvain algorithm identifies code clusters
- **Webhook indexing**: Auto-reindex on merged pull requests (GitHub and Gitea webhooks)
- **Auth & authorization**: API key auth, per-repo access grants, groups
- **MCP protocol**: Standard Model Context Protocol server for AI agent integration

## Benchmarks

**Short answer.** Across three current agent models, an agent using treeloom matched
the answer quality of a grep/glob/read agent **at 11–47% lower cost**, with fewer turns,
in every configuration we measured. The cost win is statistically significant per query
on symbol-free questions for every model. Treeloom's answer-quality score was higher in
5 of 6 configurations, **but not significantly so in any** — so the claim is *the same
answers for less*, not *better answers*.

The benchmark is *agentic*: a real LLM agent answers 100 curated questions about
[featbit](https://github.com/featbit/featbit) — a ~1,900-file C#/TypeScript production
codebase, pinned at tag `v5.4.9` — once per retrieval approach, with identical prompts
and a 12-turn cap. Answers are scored against gold references by an independent judge
(`gpt-4o-2024-08-06`, not any of the agents). Two question populations run separately:
**symbol-bearing** (the identifier is in the question) and **symbol-free** (a
behavioural description with no name to search for).

Treeloom vs the grep/glob/read (Claude-Code-style) agent, September 2026, n=100 per row:

| agent model | questions | cost vs grep (per-query p) | turns (grep → treeloom) | 12-turn caps (grep / treeloom) | correctness (grep → treeloom, p) |
|---|---|---|---|---|---|
| deepseek-flash | symbol-bearing | **−11%** (p=0.057) | 4.7 → 3.6 | 0 / 0 | 4.52 → 4.49 (p=0.851) |
| deepseek-flash | **symbol-free** | **−30%** (p=5.6e-07) | 6.7 → 4.2 | 11 / 4 | 4.40 → 4.56 (p=0.359) |
| claude-sonnet-5 | symbol-bearing | **−14%** (p=0.00041) | 4.8 → 3.3 | 2 / 1 | 4.41 → 4.47 (p=0.286) |
| claude-sonnet-5 | **symbol-free** | **−25%** (p=4.7e-06) | 6.2 → 3.3 | 11 / 3 | 4.14 → 4.43 (p=0.185) |
| gpt-5.5 | symbol-bearing | **−20%** (p=0.035) | 5.5 → 5.0 | 5 / 6 | 4.30 → 4.38 (p=0.345) |
| gpt-5.5 | **symbol-free** | **−47%** (p=3.1e-17) | 8.2 → 5.2 | 24 / 8 | 4.05 → 4.41 (p=0.061) |

What the numbers say:

- **Treeloom is cheaper.** The per-query cost difference is significant in 5 of 6 rows
  (deepseek-flash on symbol-bearing questions is borderline, p=0.057) and on every
  symbol-free row. The saving is largest on the most expensive model, where every
  extra exploratory turn costs more.
- **On symbol-free questions grep runs out of road.** It hit the 12-turn cap on 11–24%
  of questions, significantly more often than treeloom on every model. Capable agents
  usually still reach the answer; they just pay for the search.
- **On symbol-bearing questions grep is hard to beat at finding the file** — it
  string-matches the identifier. Ranking is a statistical tie on two models, and grep
  ranks the right file first significantly more often on deepseek-flash (p=0.023).
  Treeloom still gets there for less.
- **vs vector RAG with identical embeddings** (zilliztech/claude-context,
  deepseek-flash): treeloom ranks the right file first significantly more often —
  recall@1 0.68 vs 0.48 (p=0.0017) on symbol-bearing and
  0.44 vs 0.28 (p=0.007) on symbol-free questions — at
  11–22% fewer tokens, with answer quality tied.

Full tables, per-query significance, setup, spend, and reproduction steps:
[docs/benchmark-results-2026-09.md](docs/benchmark-results-2026-09.md).

### Semantic search, not symbol lookup

**What you ask matters more than which tool you use.** When the identifier is already in
the question, grep finds it directly and treeloom's advantage is cost alone (11–20%
cheaper). When you can only describe the behaviour, grep has nothing to match and
flounders; treeloom's cost advantage widens to 25–47%. *Don't use a semantic engine for
symbol lookup — grep already nails it.* Use treeloom for "how does this work?" and
"where is the code that does X?".

The graph layer earns its keep mainly through the **neighbor and community context it
feeds the agent**. Whether graph *rescoring* of results adds anything on top depends on
the strength of the reranker — see [docs/benchmark-findings.md](docs/benchmark-findings.md)
and [docs/graph-ablation.md](docs/graph-ablation.md).

### Earlier results (June 2026)

An earlier multi-repo run — featbit plus **PowerToys** (C#), **guava** (Java) and
**three.js** (JS), n=100 per repo, agent `deepseek-chat` — found treeloom **significantly
better** than grep on symbol-free answer quality (pooled p<0.0001), with the grep agent
collapsing outright on PowerToys (recall 0.27, 43k tokens of failed searches). That run
also measured aider's repo-map (an 8k-token symbol map re-sent every turn: 5.1× grep's
tokens, and worse retrieval — see `docs/benchmark-refresh-runbook.md` §1) and a cost comparison including Claude Opus 4.8 (−16% vs
grep). Two things limit how far to lean on it: it used an earlier version of the
benchmark harness whose numbers this project does not mix with current ones, and its
significant answer-quality edge **did not replicate** on featbit with current models,
whose agents keep grep competitive. The cost direction did replicate.

That pattern — a quality edge with weaker agents, cost-only with stronger ones — shows up
within the current harness too: a July 2026 run with **gpt-4o** as the agent (weaker than
the models above, and judging its own answers) found treeloom significantly better on
symbol-free answer quality on featbit (p=2.3e-10), with grep capping out on 26% of
questions. When grep has nothing to match, a weaker agent gives up; a capable one keeps
searching and pays for it. Details:
[docs/benchmark-findings.md](docs/benchmark-findings.md) and
[docs/cost-across-models.md](docs/cost-across-models.md).

### Caveats, stated plainly

- **One repository** in the September run (featbit: C# back end, TypeScript front end),
  and one run per configuration — no repeated-run variance estimate.
- **The judge scale is compressed.** Correctness means sit between 4.05 and 4.56 out of
  5, with 71–81 ties per 100 questions, which makes small quality differences hard to
  detect.
- **Recall counts files the agent opened**, so an arm that takes more turns scores higher
  on it mechanically; read grep's recall numbers with that in mind.
- A treeloom search takes roughly 3 seconds, most of it reranking — latency traded for
  fewer agent turns.
- Raw per-query result rows are not shipped in the repo; the query sets and gold answers
  are. Reproduce with `python -m treeloom.benchmark agentic --help`.

We also ran a milestone of **token-optimization** experiments trying to shrink the
search payload, and rejected nearly all of them: a smaller response just makes the
agent re-read (compensatory fetch), raising turns and erasing the saving. The only
default-on win was *serialization* (markdown encoding, −26%), not *content*. The
methodology and every rejected experiment — including across five agent models, where
trimming gets **worse** as capability rises — are written up in
[docs/token-optimization-rejected.md](docs/token-optimization-rejected.md).

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-code` | TEI embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | TEI reranker model |
| `VECTOR_DIM` | `768` | Must match embedding model |
| `LLM_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM for HyDE/summaries (Ollama default) |
| `DATABASE_URL` | required | Postgres connection string (job state + summary cache) |

See `.env.example` for all options.

## Observability

Treeloom emits OpenTelemetry traces from the indexer for every job and
per-stage span (`parse_and_chunk`, `embed.chunks`, `summaries.*`,
`milvus.*`, `graph.index_file`). The bundled `docker-compose.yml` defines an
OTel collector (head sampling, `always_on` by default — tail sampling belongs
in a central collector, see `docs/observability-runbook.md`), Tempo and
Grafana; `run.sh` does not start them, so bring them up with
`docker compose up -d otel-collector tempo grafana`. Point Grafana at the Tempo
datasource (`assets/observability/grafana-datasources.yaml`) and import
`assets/grafana/treeloom-indexer-traces.json` for the pre-built per-stage
dashboard. Configure via the standard
`OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_TRACES_SAMPLER` env vars;
span names use the stable `treeloom.` attribute prefix. Set
`OTEL_SDK_DISABLED=true` to turn it off entirely.

## Testing

```bash
# The bare `pip install -e .` from Quick Start above is not enough — the
# full suite also exercises every optional-backend adapter. Pull in dev
# (pytest) + all-backends + simple:
pip install -e ".[dev,all-backends,simple]"
pytest tests/unit -q     # 1,656 Detroit-style unit tests (no services needed)
```

Retrieval quality is validated separately against a live stack via the
benchmark harness (`python -m treeloom.benchmark run|ab|agentic`).

## Documentation

[docs/index.md](docs/index.md) maps every document under `docs/`: a one-line
summary of each, how far to trust it, and a recommended reading order for
evaluating, operating, and changing Treeloom.

## Brand

See [assets/branding/](assets/branding/) for logo files, favicon, and full brand
guidelines. The Thread-Tree mark is the canonical Treeloom symbol.

## License

MIT
