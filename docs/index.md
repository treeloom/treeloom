# Treeloom documentation

Treeloom is a GraphRAG system for code search: tree-sitter parsing, chonkie
chunking, HuggingFace TEI embeddings, Milvus vectors, a Neo4j code graph, and
an MCP server that lets an agent search a codebase semantically instead of
grepping it. The project README covers installation and the quickstart; these
pages hold the deployment, operations, and benchmark material.

```{toctree}
:caption: Getting started
:maxdepth: 1

simple-mode
docker-images
```

```{toctree}
:caption: Operating Treeloom
:maxdepth: 1
:glob:

engineering-notes
incremental-indexing
fleet-operations
authz
result-provenance
observability-runbook
gpu-cpu-routing
```

```{toctree}
:caption: Benchmarks and evidence
:maxdepth: 1

benchmark-eval
benchmark-refresh-runbook
benchmark-results-2026-09
benchmark-findings
cost-across-models
graph-ablation
token-optimization-rejected
```

```{toctree}
:caption: Architecture decisions
:maxdepth: 1

adr-001-cost-aware-embedding-proxy
adr-002-no-langchain-langgraph
```
