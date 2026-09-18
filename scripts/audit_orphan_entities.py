#!/usr/bin/env python3
"""Audit orphan :Entity nodes in Neo4j — READ ONLY.

Background
----------
~202k :Entity nodes were observed in production Neo4j with NO incoming
`(:Source)-[:CONTAINS]->`. The live delete path is mostly fine: the Neo4j
`delete_source` (src/treeloom/adapters/neo4j/graph_store.py) DETACH-DELETEs
entities that are not CONTAINED by any *other* Source, so re-index/delete does
not strand the entities it created via the CONTAINS pass.

The orphans come from two sources:

1.  **MERGE-target accumulation (the structural cause).** The relationship pass
    in `store_graph` does:

        UNWIND $rels AS rel
        MERGE (s:Entity {id: rel.source_id})
        MERGE (t:Entity {id: rel.target_id})
        SET t.type = CASE WHEN t.type IS NULL THEN 'ExternalModule' ELSE t.type END
        MERGE (s)-[r:<TYPE>]->(t)

    `MERGE (t:Entity {id: rel.target_id})` auto-creates the *target* of every
    CALLS / IMPORTS / INHERITS edge that does not resolve to a real, indexed
    definition (external libs, stdlib, unresolved cross-file symbols). These
    bare nodes get `id` only (plus `type='ExternalModule'` for targets), and
    are NEVER linked into `(:Source)-[:CONTAINS]->`. `delete_source` only walks
    the CONTAINS edge, so these are immortal — they accumulate forever.

2.  **Pre-existing accumulation** from before the current delete path
    (`delete_source` at neo4j/graph_store.py:379) was in place.

This script characterizes the orphan population so an operator can pick a
safe cleanup criterion. It does NOT mutate anything. Run the cleanup with
`scripts/cleanup_orphan_entities.py` after reviewing these counts.

Usage
-----
    python scripts/audit_orphan_entities.py        # connects via NEO4J_* env

Requires a LIVE Neo4j (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, same env the
indexer uses). This file is committed primarily as the canonical, reviewed set
of characterization queries — every query below can also be pasted straight
into the Neo4j Browser.
"""
from __future__ import annotations

import asyncio
import os

# ---------------------------------------------------------------------------
# Characterization queries. An "orphan" is an :Entity with no incoming
# (:Source)-[:CONTAINS]->. The definition is shared by the cleanup script.
# ---------------------------------------------------------------------------

ORPHAN_MATCH = (
    "MATCH (e:Entity) "
    "WHERE NOT EXISTS { MATCH (:Source)-[:CONTAINS]->(e) }"
)

QUERIES: dict[str, str] = {
    # Total orphan count — should be ~202k on the affected instance.
    "total_orphans": f"{ORPHAN_MATCH} RETURN count(e) AS n",

    # By label-set. ExternalModule targets typically dominate. Note an :Entity
    # may carry extra labels; `labels(e)` returns all of them.
    "by_labels": (
        f"{ORPHAN_MATCH} "
        "RETURN labels(e) AS labels, count(*) AS n ORDER BY n DESC"
    ),

    # By entity `type` property + whether it carries a real file_path.
    # Orphans created purely as MERGE targets have NO file_path (they were
    # never SET by the entity-upsert pass); real-but-unlinked entities WOULD
    # carry file_path (those would indicate a delete-path bug, not the
    # MERGE-target cause). This split is the load-bearing diagnostic.
    "by_type_and_filepath": (
        f"{ORPHAN_MATCH} "
        "RETURN coalesce(e.type, '<null>') AS type, "
        "       (e.file_path IS NOT NULL AND e.file_path <> '') AS has_file_path, "
        "       count(*) AS n "
        "ORDER BY n DESC"
    ),

    # Of the orphans, how many are the TARGET of at least one relationship
    # (i.e. genuinely referenced — deleting them removes a real edge endpoint)
    # vs. fully dangling (no in/out edges at all — pure garbage). The
    # relationship pass always MERGEs the edge after the node, so a pure
    # MERGE-target orphan should be a CALLS/IMPORTS/INHERITS target.
    "by_is_rel_target": (
        f"{ORPHAN_MATCH} "
        "RETURN exists { (e)<-[:CALLS|IMPORTS|INHERITS]-() } AS is_rel_target, "
        "       count(*) AS n "
        "ORDER BY n DESC"
    ),

    # Fully dangling orphans: no incoming AND no outgoing edges of any type.
    # If this is > 0 it is the safest possible delete set (removing them
    # cannot break any traversal).
    "fully_dangling": (
        f"{ORPHAN_MATCH} AND NOT (e)--() "
        "RETURN count(e) AS n"
    ),

    # Orphans that carry a file_path (NOT pure MERGE targets). A non-zero count
    # here would indicate entities the delete path failed to reap — worth
    # investigating before any bulk delete. Sampled for inspection.
    "orphans_with_filepath_sample": (
        f"{ORPHAN_MATCH} AND e.file_path IS NOT NULL AND e.file_path <> '' "
        "RETURN e.id AS id, e.type AS type, e.name AS name, "
        "       e.file_path AS file_path "
        "LIMIT 25"
    ),

    # Sanity: how many NON-orphan entities (properly CONTAINED) exist, so the
    # orphan fraction is interpretable.
    "contained_entities": (
        "MATCH (:Source)-[:CONTAINS]->(e:Entity) RETURN count(DISTINCT e) AS n"
    ),
}

# Interpretation guide printed alongside the results.
INTERPRETATION = """
Interpretation
--------------
- total_orphans ~= 202k confirms the reported population.
- by_is_rel_target: expect the overwhelming majority to have
  is_rel_target=true and type=ExternalModule with has_file_path=false. That is
  the MERGE-target population (cause #1) — EXPECTED, structural, and arguably
  legitimate graph data (they let find_callees/IMPORTS traversal reach
  external symbols).
- orphans_with_filepath_sample / by_type_and_filepath has_file_path=true:
  these are the SUSPICIOUS ones. A real indexed Class/Function with a
  file_path but no CONTAINS edge means the delete path (or the old MATCH-Source
  bug) stranded it. If the count is meaningful, treat THESE as the cleanup
  target — they are dead weight that also pollutes name/file lookups.
- fully_dangling: nodes with zero edges are pure garbage and the safest to
  delete unconditionally.

Cleanup policy (see scripts/cleanup_orphan_entities.py and ):
  RECOMMENDED default cleanup set = orphans that are NOT useful as external
  reference targets, i.e. fully-dangling orphans (no edges at all) PLUS
  orphans carrying a file_path but no CONTAINS (stranded real entities).
  ExternalModule MERGE-targets that ARE relationship targets are EXCLUDED from
  cleanup by default — they are the documented, intended product of the
  relationship pass; relabel/leave them rather than delete. Override with
  --include-rel-targets only after confirming they serve no traversal value.
"""


async def main() -> None:
    from neo4j import AsyncGraphDatabase

    uri = os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_BOLT_URL")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "treeloom_pass")
    if not uri:
        raise SystemExit(
            "Set NEO4J_URI (e.g. bolt://localhost:7687) to run this audit."
        )

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            for name, cypher in QUERIES.items():
                result = await session.run(cypher)
                rows = await result.fetch(1000)
                print(f"\n## {name}\n{cypher}")
                for r in rows:
                    print("  ", dict(r))
    finally:
        await driver.close()
    print(INTERPRETATION)


if __name__ == "__main__":
    asyncio.run(main())
