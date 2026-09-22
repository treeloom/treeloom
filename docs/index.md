# Treeloom documentation

A map of everything under `docs/`: what each document is for, how far to
trust it, and the order to read them in. Last checked against the tree on
2026-09-22.

Every document falls into one of four trust classes, and the class matters
more than the title:

| Class | Meaning |
|---|---|
| **Rule** | Binding. If a change contradicts it, the change is wrong, or the rule must be changed first. |
| **Feature doc** | The maintained description of a shipped feature. Kept current with the code. |
| **Evidence** | The maintained record of what has been measured and decided. Living documents that gain dated sections; later sections qualify earlier ones. |
| **Snapshot** | Honest numbers from one dated run. Trust them for that configuration on that date only, and check whether a later document qualified the conclusion. |
| **Historical** | Kept for the record behind a banner that says which parts still hold. Cite only those parts. |

## Recommended reading order

**Evaluating Treeloom** (about an hour):

1. The [README](https://github.com/treeloom/treeloom/blob/main/README.md): what Treeloom is, the "semantic search, not
   symbol lookup" framing, and the Quick Start.
2. [Simple deployment mode](simple-mode.md) if you have no GPU, or the
   README's full-stack Quick Start if you do.
3. [Benchmark results, September 2026](benchmark-results-2026-09.md): the
   current headline numbers and their caveats.
4. [Result provenance](result-provenance.md): what a search hit contains and
   how to cite it.

**Operating it** (after the above):

5. [Docker images](docker-images.md): published images, tags, versioning, and
   the release procedure.
6. [Authorization model](authz.md): users, groups, tokens, and per-repo
   access. Read this before exposing the indexer to anyone but yourself.
7. [Incremental indexing](incremental-indexing.md): keeping an index current
   from push webhooks.
8. [Fleet operations](fleet-operations.md): onboarding and monitoring many
   repositories at once.
9. [Observability runbook](observability-runbook.md): metrics, logs, and
   traces into a central stack.

**Changing it** (before touching retrieval or proposing an experiment):

10. [Benchmark eval workflow](benchmark-eval.md): how to measure grep versus
    Treeloom on your own repositories, with significance.
11. [Benchmark findings](benchmark-findings.md): the verdict ledger. Read it
    bottom-up; the later dated sections supersede the earlier ones.
12. [Token optimization: rejected](token-optimization-rejected.md): every
    payload-trimming idea that lost, and why. Read before proposing another.
13. The two ADRs, [ADR-001](adr-001-cost-aware-embedding-proxy.md) and
    [ADR-002](adr-002-no-langchain-langgraph.md), which forbid two specific
    design directions.

## Getting started and operating

| Document | Class | Summary |
|---|---|---|
| [simple-mode.md](simple-mode.md) | Feature doc | The no-GPU evaluation profile: embedded LanceDB and SQLite, one API key for embeddings and summaries, a CPU reranker, and only Postgres in Docker. Includes the measured local-versus-cloud reranker comparison. |
| [docker-images.md](docker-images.md) | Feature doc | The four published Docker Hub images, their platforms, the CalVer tag scheme, how to pull a release, and how a maintainer cuts one. |
| [result-provenance.md](result-provenance.md) | Feature doc | The provenance fields on every search hit, the citation grammar, the top-level `sources` map, and staleness checking against the current commit. |
| [authz.md](authz.md) | Feature doc | How Treeloom decides who may read which repository: local login and tokens, groups and grants, the decision precedence, and the data flow from an MCP client to the stores. |
| [incremental-indexing.md](incremental-indexing.md) | Feature doc | How a GitHub or Gitea push webhook becomes a changed-files job, retries and dead-lettering, backpressure, and the two load-test harnesses. |
| [fleet-operations.md](fleet-operations.md) | Feature doc | Bulk onboarding from a manifest, the fleet health rollup, and the opt-in scheduled refresh for repositories whose HEAD has moved. |
| [observability-runbook.md](observability-runbook.md) | Feature doc | Wiring metrics, logs, and traces from a deployment into a central OpenTelemetry, Prometheus, Loki, Tempo, and Grafana host. Written as a reference plan; some proposed compose profiles do not exist yet. |
| [engineering-notes.md](engineering-notes.md) | Feature doc | The reference material moved out of `CLAUDE.md`: running the stack, simple mode, indexing and dev tasks, the benchmark CLI and its verdicts, jobs and queue, fleet, auth, tracing, and the prompt enhancer. Dense and verbatim; read the section you need. |

## Evidence and benchmarks

| Document | Class | Summary |
|---|---|---|
| [benchmark-results-2026-09.md](benchmark-results-2026-09.md) | Evidence, current headline | The September 2026 refresh on a pinned corpus with three current agents and an independent judge. Treeloom matched grep's answer quality at 11 to 47 percent lower cost; the correctness edge was not significant in any cell. The README table is copied from here. |
| [benchmark-eval.md](benchmark-eval.md) | Evidence, workflow | The reproducible four-step comparison (`queries`, `clean`, `agentic`, `rollup`), query-hygiene pitfalls, and the house reading of p-values. |
| [benchmark-refresh-runbook.md](benchmark-refresh-runbook.md) | Evidence, workflow | How the 2026-09 refresh was planned, priced, and run: model choices, excluded arms, and spend. The template for the next refresh. |
| [benchmark-findings.md](benchmark-findings.md) | Evidence, ledger | The full grep-versus-Treeloom record since the first run, in dated sections. Later sections qualify earlier ones, including the graph-rescoring re-validation under the production reranker. |
| [token-optimization-rejected.md](token-optimization-rejected.md) | Evidence, negative results | Every attempt to shrink the search response, each benchmarked with a real agent and rejected because the agent fetched back what was removed. Its "open threads" section is the nearest thing to a roadmap. |
| [graph-ablation.md](graph-ablation.md) | Snapshot, June 2026 | One run switching the two graph mechanisms off in turn. Symbol promotion carries most of the value; the "rescoring adds little" result was later shown to depend on reranker strength, as the banner explains. |
| [cost-across-models.md](cost-across-models.md) | Snapshot, June 2026 | The first cost-in-dollars comparison across four model tiers. Superseded as the headline by the September refresh; still the latest Opus number. |

## Design records

| Document | Class | Summary |
|---|---|---|
| [adr-001-cost-aware-embedding-proxy.md](adr-001-cost-aware-embedding-proxy.md) | Rule | Why embedding requests are balanced by token cost in-process rather than by connection count in a proxy, and why oversized chunks are pinned to CPU. |
| [adr-002-no-langchain-langgraph.md](adr-002-no-langchain-langgraph.md) | Rule | Why Treeloom does not adopt LangChain or LangGraph, and the one narrow case in which LangGraph could be reconsidered. |

## Historical

| Document | Class | Summary |
|---|---|---|
| [gpu-cpu-routing.md](gpu-cpu-routing.md) | Historical | The GPU out-of-memory postmortem that led to the token threshold. The routing mechanics it describes no longer exist; ADR-001 is current. |

## Documents outside this folder

- [README.md](https://github.com/treeloom/treeloom/blob/main/README.md): the public front door and Quick Start.
- [CONTRIBUTING.md](https://github.com/treeloom/treeloom/blob/main/CONTRIBUTING.md): contribution gates, including that
  retrieval changes need a benchmark run.
- [SECURITY.md](https://github.com/treeloom/treeloom/blob/main/SECURITY.md): how to report a vulnerability.
- [CLAUDE.md](https://github.com/treeloom/treeloom/blob/main/CLAUDE.md): the short rulebook for maintainers and coding
  agents: the working agreement, the test command, and the invariants that
  must not be violated. The reference material behind them is in
  [engineering-notes.md](engineering-notes.md).
- [assets/branding/BRAND.md](https://github.com/treeloom/treeloom/blob/main/assets/branding/BRAND.md): logo files,
  palette, and usage rules.

## Keeping this index current

Add a row when a document is added, and move it between tables when its
class changes: a snapshot that is superseded gains a banner and a note here;
a plan that ships becomes a feature doc. Summaries describe what a reader will
find, never a number, so they don't drift when results are refreshed.

% Sidebar navigation for the Sphinx/Read the Docs build. Hidden so the tables
% above stay the page body; the captions become the sidebar sections.

```{toctree}
:caption: Getting started and operating
:maxdepth: 1
:hidden:

simple-mode
docker-images
engineering-notes
result-provenance
authz
incremental-indexing
fleet-operations
observability-runbook
```

```{toctree}
:caption: Evidence and benchmarks
:maxdepth: 1
:hidden:

benchmark-results-2026-09
benchmark-eval
benchmark-refresh-runbook
benchmark-findings
token-optimization-rejected
graph-ablation
cost-across-models
```

```{toctree}
:caption: Design records and history
:maxdepth: 1
:hidden:

adr-001-cost-aware-embedding-proxy
adr-002-no-langchain-langgraph
gpu-cpu-routing
```


