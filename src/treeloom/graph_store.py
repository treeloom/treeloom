"""Graph-store dispatch — GRAPH_STORE env selects the backend module.

- `neo4j` (default): the production graph store (Bolt, async driver 6.x).
- `sqlite`: embedded single-file store (aiosqlite) — the simple-profile
  choice; no server, no credentials.

Both backends implement the same module-level function surface (the
`_EXPORTED` list below is the contract). This shim re-exports the selected
backend's functions so `from treeloom import graph_store` /
`from treeloom.graph_store import X` are backend-agnostic.
"""
import os

_backend = os.environ.get("GRAPH_STORE", "neo4j")

if _backend == "sqlite":
    from treeloom.adapters.sqlite import graph_store as _impl
else:
    from treeloom.adapters.neo4j import graph_store as _impl

# The full cross-backend contract: every function application code calls on
# the graph store. Adding a name here requires implementing it in BOTH
# backends.
_EXPORTED = [
    "clear_all",
    "close",
    "delete_entities_by_file",
    "delete_file_entities",
    "delete_source",
    "delete_source_communities",
    "ensure_schema",
    "entity_counts_by_source",
    "find_callees",
    "find_callers",
    "find_entities_by_name",
    "find_entity",
    "find_references",
    "get_all_community_summaries",
    "get_community_ids",
    "get_community_summaries_map",
    "get_entities_by_community",
    "get_entity_by_id",
    "get_entities_by_file",
    "get_entities_by_files_batch",
    "get_neighbors_batch",
    "list_entities",
    "list_sources",
    "load_community_embeddings",
    "pull_graph",
    "set_centralities_batch",
    "set_community_id",
    "set_community_ids_batch",
    "set_community_embeddings_batch",
    "store_graph",
    "traverse",
    "upsert_community_summaries_batch",
    "upsert_entity",
    "upsert_relationship",
    "upsert_source",
]

for _name in _EXPORTED:
    globals()[_name] = getattr(_impl, _name)

# Neo4j-only escape hatch still used by community_adapter's raw-Cypher
# queries; not part of the cross-backend contract.
if _backend != "sqlite":
    _get_session = _impl._get_session

del _name, _impl
