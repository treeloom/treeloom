#!/usr/bin/env python3
"""Clean up orphan :Entity nodes in Neo4j.

DRY-RUN BY DEFAULT. Prints the counts of what *would* be deleted and exits
without mutating anything. Pass `--apply` to actually DETACH DELETE.

Read `scripts/audit_orphan_entities.py` FIRST and run it against the live
instance — the cleanup criterion you choose must be backed by those counts.

What counts as an "orphan"
--------------------------
An :Entity with NO incoming `(:Source)-[:CONTAINS]->`. See the audit script
for why these exist (MERGE-target accumulation + pre-existing rows).

Cleanup policy (the decision for )
-------------------------------------------
There are two distinct orphan populations and they are NOT treated the same:

  A. MERGE-target ExternalModule nodes that ARE relationship targets
     (`(e)<-[:CALLS|IMPORTS|INHERITS]-()`). These are the INTENDED product of
     `store_graph`'s relationship pass — they let graph traversal reach
     external/unresolved symbols (numpy, stdlib, cross-file callees). They are
     EXCLUDED from cleanup by default. Recommendation: KEEP them and treat them
     as legitimate (they already carry `type='ExternalModule'`); do NOT change
     `store_graph` to stop creating them without a live infra A/B, since that
     would regress IMPORTS/CALLS traversal to external symbols. If you later
     decide they are noise, delete them with `--include-rel-targets`.

  B. Orphans that are (i) fully dangling (no edges at all) and/or (ii) carry a
     `file_path` but no CONTAINS edge (stranded real entities — dead weight
     that also pollutes find_definition / name lookups). These are the SAFE
     default cleanup set.

Default delete set = (B): orphans where
    NOT EXISTS { (:Source)-[:CONTAINS]->(e) }
    AND (
        NOT (e)--()                                   -- fully dangling
        OR (e.file_path IS NOT NULL AND e.file_path <> '')  -- stranded real
    )

`--include-rel-targets` widens the set to ALL orphans (population A + B),
i.e. drops the protective clause — only do this after the audit shows the
ExternalModule targets serve no value for your workload.

Deletion is batched (`CALL { ... } IN TRANSACTIONS OF N ROWS`) so a 202k purge
does not blow the transaction memory.

Usage
-----
    python scripts/cleanup_orphan_entities.py             # dry-run, safe set
    python scripts/cleanup_orphan_entities.py --apply     # delete safe set
    python scripts/cleanup_orphan_entities.py --include-rel-targets --apply
"""
from __future__ import annotations

import argparse
import asyncio
import os

ORPHAN_WHERE = "NOT EXISTS { MATCH (:Source)-[:CONTAINS]->(e) }"

# The protective clause that keeps useful external-reference targets alive.
SAFE_EXTRA = (
    "( NOT (e)--() "
    "OR (e.file_path IS NOT NULL AND e.file_path <> '') )"
)


def _match_clause(include_rel_targets: bool) -> str:
    where = ORPHAN_WHERE
    if not include_rel_targets:
        where += f" AND {SAFE_EXTRA}"
    return f"MATCH (e:Entity) WHERE {where}"


def _count_query(include_rel_targets: bool) -> str:
    return f"{_match_clause(include_rel_targets)} RETURN count(e) AS n"


def _delete_query(include_rel_targets: bool, batch: int) -> str:
    # CALL-in-transactions for memory-safe bulk delete. The inner subquery must
    # re-bind `e`; we pass it through the WITH.
    return (
        f"{_match_clause(include_rel_targets)} "
        "WITH e "
        "CALL { WITH e DETACH DELETE e } "
        f"IN TRANSACTIONS OF {batch} ROWS"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually delete. Without this flag the script is a dry-run.",
    )
    ap.add_argument(
        "--include-rel-targets", action="store_true",
        help="Also delete ExternalModule MERGE-target orphans (population A). "
             "Default keeps them.",
    )
    ap.add_argument(
        "--batch", type=int, default=10000,
        help="Rows per delete transaction (default 10000).",
    )
    args = ap.parse_args()

    from neo4j import AsyncGraphDatabase

    uri = os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_BOLT_URL")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "treeloom_pass")
    if not uri:
        raise SystemExit(
            "Set NEO4J_URI (e.g. bolt://localhost:7687) to run cleanup."
        )

    scope = "ALL orphans (incl. rel-targets)" if args.include_rel_targets \
        else "SAFE set (dangling + stranded-with-file_path)"

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            cq = _count_query(args.include_rel_targets)
            result = await session.run(cq)
            row = (await result.fetch(1))[0]
            n = row["n"]
            print(f"Cleanup scope: {scope}")
            print(f"Count query:\n  {cq}")
            print(f"Orphans matching: {n}")

            if not args.apply:
                print("\nDRY RUN — nothing deleted. Re-run with --apply to delete.")
                print("Delete query that WOULD run:\n  "
                      + _delete_query(args.include_rel_targets, args.batch))
                return

            if n == 0:
                print("Nothing to delete.")
                return

            dq = _delete_query(args.include_rel_targets, args.batch)
            print(f"\nAPPLYING delete (batched {args.batch}/tx):\n  {dq}")
            await session.run(dq)

            result = await session.run(cq)
            remaining = (await result.fetch(1))[0]["n"]
            print(f"Done. Remaining orphans in scope: {remaining}")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
