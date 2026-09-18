# What the graph layer contributes (ablation)

**Date:** 2026-06-12

A single retrieval-quality experiment that switches treeloom's two graph mechanisms
off one at a time to see how much each contributes to ranking. Read this if you are
wondering whether the code knowledge graph (Neo4j/SQLite) is worth running. It is a
point-in-time result on one repository; the banner below records how a later run
qualified it.

> **Qualified by later work.** The "rescoring adds almost nothing" result below
> was measured under the bge reranker on featbit only. A later re-validation
> under the production `Qwen3-Reranker-0.6B` found rescoring lifts guava
> symbol-free recall@1 0.16 → 0.49 (MRR 0.448 → 0.621), with featbit unchanged —
> see the section "D2/D3: what the graph layer actually contributes" in
> [benchmark-findings.md](benchmark-findings.md#d2d3-2026-06-15-what-the-graph-layer-actually-contributes).
> Rescoring's value
> is reranker-dependent, and `USE_GRAPH_SCORING=1` remains the default. The
> symbol-promotion finding stands.

**Harness:** featbit-clean-100 (100 curated queries against the
[featbit](https://github.com/featbit/featbit) codebase, 1,350+ indexed files on the
2026-06-12 working checkout, C#/TypeScript — file counts differ between docs because the
checkout moved until the 2026-09 refresh pinned tag `v5.4.9`).
A second, search-only indexer on port 8002 reading the production Milvus collection
(`jina-embeddings-v2-base-code` embeddings) and production Neo4j graph. Reranker:
`BAAI/bge-reranker-v2-m3` served by HuggingFace Text Embeddings Inference (TEI) on a
GPU at `:8081`. `--no-hyde` in all three runs (HyDE query expansion is an LLM call
and makes results nondeterministic). Pre-rerank pool: 50 candidates.
Metrics: **recall@k** = fraction of queries whose ground-truth file is in the top k;
**MRR** = mean reciprocal rank of the first ground-truth hit.
The graph state (Louvain communities, centrality scores) was frozen across all
three runs — no community rebuild was triggered — so graph-build nondeterminism
does not apply here.

## Background: two graph mechanisms

Treeloom's graph layer contributes in two distinct ways:

1. **Symbol promotion** — when a query contains a token that matches a known
   entity name in the graph (a class, function, or module), `_promote_definition`
   calls `find_entities_by_name` to locate the canonical definition chunk and
   promotes it to rank 1, short-circuiting the reranker order for that query.
   This is controlled by the `SYMBOL_PROMOTE` env var (default `1`, enabled).

2. **Graph-signal rescoring** — after reranking, community-summary cosine
   similarity, entity centrality, and a name-in-query bonus are fused with the
   normalized reranker logit score using weights `GRAPH_ALPHA/BETA/GAMMA/DELTA`.
   This reorders chunks within the post-rerank pool based on their position in
   the code knowledge graph. It is on by default (`USE_GRAPH_SCORING=1`) and is
   disabled by the `--no-graph` benchmark flag (or the `use_graph_scoring=false`
   `/search` request parameter).

**Important:** `--no-graph` disables only graph-signal rescoring. Symbol promotion
is a separate code path that runs unless `SYMBOL_PROMOTE=0` is set in the
environment. This means a run with `--no-graph` but without `SYMBOL_PROMOTE=0`
is **not** graph-free — it still benefits from exact-identifier definition lookup.
The three configs below isolate the contribution of each mechanism.

## Aggregate results

| Config | Graph mechanisms active | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| (a) pure | none (`SYMBOL_PROMOTE=0`, `--no-graph`) | 0.520 | 0.760 | 0.617 |
| (b) promote | symbol promotion only (`--no-graph`) | **0.630** | **0.800** | **0.698** |
| (c) rescore | symbol promotion + graph-signal rescoring | **0.630** | **0.800** | **0.700** |

## Pairwise per-query MRR win/loss/tie

"Wins" = right-hand config improves MRR on that query; "losses" = left-hand config
was better. Ties are excluded before the sign test.

| Pair | Wins (right) | Losses | Ties | p (sign test) |
|---|---|---|---|---|
| a → b (pure → +promote) | 12 | 1 | 87 | 0.003 |
| b → c (promote → +rescore) | 2 | 0 | 98 | 0.500 |
| a → c (pure → full graph) | 14 | 1 | 85 | 0.001 |

## Interpretation

**Symbol promotion is the load-bearing mechanism.** Moving from pure vector+rerank
to symbol promotion adds +11 recall@1 points (0.52 → 0.63), +4 recall@5, +8 MRR
points. The per-query split is 12W/1L (p=0.003) — strongly significant. When a
query names a specific identifier, the graph lets Treeloom find the definition
file directly; without it, the dense retriever has to surface that file on its own.

**Graph-signal rescoring adds almost nothing on this benchmark.** Enabling
community-summary cosine + centrality + name-in-query rescoring (b → c) changes
the aggregate by 0 recall@1 / 0 recall@5 / +0.002 MRR. Only 2 of 100 queries
resolved (both in favor of rescoring, 0 against), but with n=2 the sign test
gives p=0.500 — consistent with no effect. This is not a finding that rescoring
cannot help; it is a finding that on this query set and this graph state, the
reranker and symbol promotion together leave almost no room for graph signals
to rearrange results.

These are honest numbers from a single benchmark run on one repository (featbit).
Conclusions should not be extrapolated to repositories with different graph density,
query styles, or community structure.
