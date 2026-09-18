"""Embedded SQLite graph store (GRAPH_STORE=sqlite).

Single-file replacement for the Neo4j adapter in the simple deployment
profile — no server, no credentials. Implements the full module-level
function contract in `treeloom/graph_store.py:_EXPORTED`.

The graph Treeloom actually needs is modest: entity/relationship upserts,
depth-1 traversal, name/file lookups, and community storage — Louvain and
centrality already run in-process over networkx (community_adapter), so
plain tables + JOINs cover everything.

Semantics intentionally mirror the Neo4j adapter's load-bearing MERGEs
(see CLAUDE.md):
- `store_graph` creates the bare Source row mid-job (`INSERT OR IGNORE`) so
  entities are never dropped before `upsert_source` runs in `_finalize_job`.
- The relationship pass auto-creates missing endpoints and types bare
  TARGETS as `ExternalModule` (only when type is still NULL).
- `delete_source` removes only entities exclusive to that source.

Community embeddings are stored as JSON text — they're read back into
Python lists, never queried numerically.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiosqlite

from treeloom.domain.graph import is_valid_rel_type, require_valid_rel_type

logger = logging.getLogger(__name__)

GRAPH_DB_PATH = os.environ.get(
    "GRAPH_DB_PATH", str(Path.home() / ".treeloom" / "graph.db")
)

_conn: aiosqlite.Connection | None = None
_conn_lock: asyncio.Lock | None = None

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        type TEXT,
        name TEXT,
        file_path TEXT,
        start_line INTEGER,
        end_line INTEGER,
        signature TEXT,
        language TEXT,
        community_id TEXT,
        centrality REAL,
        updated_at INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS entities_file_path ON entities(file_path)",
    "CREATE INDEX IF NOT EXISTS entities_name ON entities(name)",
    "CREATE INDEX IF NOT EXISTS entities_community ON entities(community_id)",
    """
    CREATE TABLE IF NOT EXISTS relationships (
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        type TEXT NOT NULL,
        updated_at INTEGER,
        PRIMARY KEY (source_id, target_id, type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS relationships_target ON relationships(target_id)",
    """
    CREATE TABLE IF NOT EXISTS sources (
        id TEXT PRIMARY KEY,
        path TEXT,
        url TEXT,
        branch TEXT,
        indexed_at INTEGER,
        file_count INTEGER,
        chunk_count INTEGER,
        commit_sha TEXT,
        updated_at INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_entities (
        source_id TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        PRIMARY KEY (source_id, entity_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS source_entities_entity ON source_entities(entity_id)",
    """
    CREATE TABLE IF NOT EXISTS communities (
        id TEXT PRIMARY KEY,
        summary TEXT,
        embedding TEXT
    )
    """,
)

_ENTITY_COLS = (
    "id", "type", "name", "file_path", "start_line", "end_line",
    "signature", "language", "community_id", "centrality",
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _lock() -> asyncio.Lock:
    global _conn_lock
    if _conn_lock is None:
        _conn_lock = asyncio.Lock()
    return _conn_lock


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    if _conn is None:
        async with _lock():
            if _conn is None:
                Path(GRAPH_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
                conn = await aiosqlite.connect(GRAPH_DB_PATH)
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=5000")
                await conn.execute("PRAGMA foreign_keys=ON")
                _conn = conn
    return _conn


def _row_to_entity(row: aiosqlite.Row | dict) -> dict:
    """Entity dict shaped like a Neo4j node's properties.

    Unset (NULL) columns are omitted — Neo4j nodes simply lack unset
    properties, and callers use `.get()` throughout.
    """
    d = dict(row)
    return {k: v for k, v in d.items() if v is not None and k != "updated_at"}


async def close():
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None


async def ensure_schema():
    """Create tables + indexes (idempotent, cheap on every startup)."""
    conn = await _get_conn()
    async with _lock():
        for stmt in _SCHEMA_STATEMENTS:
            await conn.execute(stmt)
        await conn.commit()
    logger.info("sqlite graph schema ensured at %s", GRAPH_DB_PATH)


_ENTITY_UPSERT = """
INSERT INTO entities (id, type, name, file_path, start_line, end_line,
                      signature, language, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    type = excluded.type,
    name = excluded.name,
    file_path = excluded.file_path,
    start_line = excluded.start_line,
    end_line = excluded.end_line,
    signature = excluded.signature,
    language = excluded.language,
    updated_at = excluded.updated_at
"""


def _entity_params(e: dict) -> tuple:
    return (
        e["id"], e["type"], e["name"], e["file_path"],
        e.get("start_line", 0), e.get("end_line", 0),
        e.get("signature", ""), e.get("language", ""), _now_ms(),
    )


async def upsert_entity(entity: dict):
    conn = await _get_conn()
    async with _lock():
        await conn.execute(_ENTITY_UPSERT, _entity_params(entity))
        await conn.commit()


async def upsert_relationship(source_id: str, target_id: str, rel_type: str):
    """Create the edge iff BOTH endpoints exist (Neo4j MATCH...MATCH parity).

    Validated even though it is bound as a column value here and cannot inject:
    a type that is not a Cypher identifier has no representation in the Neo4j
    backend, so accepting it would mean the same source produced a different
    graph depending on which store was configured. Parity is the point of the
    docstring above.
    """
    require_valid_rel_type(rel_type)
    conn = await _get_conn()
    async with _lock():
        await conn.execute(
            """
            INSERT OR REPLACE INTO relationships (source_id, target_id, type, updated_at)
            SELECT ?, ?, ?, ?
            WHERE EXISTS (SELECT 1 FROM entities WHERE id = ?)
              AND EXISTS (SELECT 1 FROM entities WHERE id = ?)
            """,
            (source_id, target_id, rel_type, _now_ms(), source_id, target_id),
        )
        await conn.commit()


async def store_graph(
    entities: list[dict], relationships: list[dict], source_id: str | None = None
):
    """Persist a file's extracted graph in one transaction.

    Mirrors the Neo4j adapter's MERGE semantics exactly — see module
    docstring for the two load-bearing behaviors (mid-job Source creation,
    ExternalModule auto-typing of bare relationship targets).
    """
    conn = await _get_conn()
    now = _now_ms()
    async with _lock():
        if entities:
            if source_id:
                await conn.execute(
                    "INSERT OR IGNORE INTO sources (id, updated_at) VALUES (?, ?)",
                    (source_id, now),
                )
            await conn.executemany(
                _ENTITY_UPSERT, [_entity_params(e) for e in entities]
            )
            if source_id:
                await conn.executemany(
                    "INSERT OR IGNORE INTO source_entities (source_id, entity_id) "
                    "VALUES (?, ?)",
                    [(source_id, e["id"]) for e in entities],
                )

        valid_rels = []
        for r in relationships:
            if not is_valid_rel_type(r["type"]):
                logger.warning(
                    "store_graph: skipping relationship with invalid type %r",
                    r["type"],
                )
                continue
            valid_rels.append(r)

        if valid_rels:
            # MERGE both endpoints as bare entities (id only)...
            endpoint_ids = {r["source_id"] for r in valid_rels} | {
                r["target_id"] for r in valid_rels
            }
            await conn.executemany(
                "INSERT OR IGNORE INTO entities (id, updated_at) VALUES (?, ?)",
                [(eid, now) for eid in endpoint_ids],
            )
            # ...but only TARGETS with no type yet become ExternalModule.
            await conn.executemany(
                "UPDATE entities SET type = 'ExternalModule' "
                "WHERE id = ? AND type IS NULL",
                [(r["target_id"],) for r in valid_rels],
            )
            await conn.executemany(
                "INSERT OR REPLACE INTO relationships "
                "(source_id, target_id, type, updated_at) VALUES (?, ?, ?, ?)",
                [(r["source_id"], r["target_id"], r["type"], now)
                 for r in valid_rels],
            )
        await conn.commit()


async def upsert_source(source: dict):
    conn = await _get_conn()
    async with _lock():
        await conn.execute(
            """
            INSERT INTO sources (id, path, url, branch, indexed_at,
                                 file_count, chunk_count, commit_sha, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                path = excluded.path,
                url = excluded.url,
                branch = excluded.branch,
                indexed_at = excluded.indexed_at,
                file_count = excluded.file_count,
                chunk_count = excluded.chunk_count,
                commit_sha = excluded.commit_sha,
                updated_at = excluded.updated_at
            """,
            (
                source["id"], source.get("path", ""), source.get("url", ""),
                source.get("branch", ""), source.get("indexed_at", 0),
                source.get("file_count", 0), source.get("chunk_count", 0),
                source.get("commit_sha", ""), _now_ms(),
            ),
        )
        await conn.commit()


async def list_sources() -> list[dict]:
    conn = await _get_conn()
    cur = await conn.execute(
        "SELECT * FROM sources ORDER BY indexed_at DESC LIMIT 1000"
    )
    rows = await cur.fetchall()
    return [
        {k: v for k, v in dict(r).items() if v is not None and k != "updated_at"}
        for r in rows
    ]


async def entity_counts_by_source() -> dict[str, int]:
    """Return {source_id: entity_count} for every source in one query.

    Cheap fleet-rollup helper: a single GROUP BY over the
    source→entity join, not one query per source.
    """
    conn = await _get_conn()
    cur = await conn.execute(
        "SELECT source_id, COUNT(*) AS n FROM source_entities GROUP BY source_id"
    )
    rows = await cur.fetchall()
    return {r["source_id"]: r["n"] for r in rows}


async def delete_source(source_id: str):
    """Delete the source and entities owned EXCLUSIVELY by it."""
    conn = await _get_conn()
    async with _lock():
        # DROP first. `CREATE TEMP TABLE IF NOT EXISTS ... AS SELECT` does not
        # refresh an existing table — the IF NOT EXISTS short-circuits the whole
        # statement, SELECT included. The connection is a reused singleton, so
        # from the second delete_source() onward _doomed still held the PREVIOUS
        # source's entity ids: that source's rows were re-deleted (a no-op) while
        # the one actually being deleted kept all of its entities. Verified
        # against sqlite directly: deleting A then B left _doomed = A's ids.
        await conn.execute("DROP TABLE IF EXISTS _doomed")
        await conn.execute(
            """
            CREATE TEMP TABLE _doomed AS
            SELECT se.entity_id AS id FROM source_entities se
            WHERE se.source_id = ?1
              AND NOT EXISTS (
                  SELECT 1 FROM source_entities o
                  WHERE o.entity_id = se.entity_id AND o.source_id <> ?1
              )
            """,
            (source_id,),
        )
        await conn.execute(
            "DELETE FROM relationships WHERE source_id IN (SELECT id FROM _doomed) "
            "OR target_id IN (SELECT id FROM _doomed)"
        )
        await conn.execute(
            "DELETE FROM entities WHERE id IN (SELECT id FROM _doomed)"
        )
        await conn.execute("DROP TABLE _doomed")
        await conn.execute(
            "DELETE FROM source_entities WHERE source_id = ?", (source_id,)
        )
        await conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        await conn.commit()


def _scope_sql(
    source_id: str | None,
    exclude_source_ids: list[str] | None,
    entity_col: str = "e.id",
) -> tuple[list[str], list[Any]]:
    """Build WHERE clauses scoping an entity to authorized sources.

    Source membership lives in the source_entities(source_id, entity_id) join.
    Single-source: require the entity be in that source. Shared (all_access):
    exclude any entity present in a denied source. Subquery alias `se` is local
    so it won't collide with an outer `se`. Returns (clauses, params).
    """
    clauses: list[str] = []
    params: list[Any] = []
    if source_id:
        clauses.append(
            f"EXISTS (SELECT 1 FROM source_entities se "
            f"WHERE se.entity_id = {entity_col} AND se.source_id = ?)"
        )
        params.append(source_id)
    if exclude_source_ids:
        ph = ",".join("?" for _ in exclude_source_ids)
        clauses.append(
            f"NOT EXISTS (SELECT 1 FROM source_entities se "
            f"WHERE se.entity_id = {entity_col} AND se.source_id IN ({ph}))"
        )
        params.extend(exclude_source_ids)
    return clauses, params


async def get_entities_by_file(
    file_path: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    conn = await _get_conn()
    clauses, scope_params = _scope_sql(source_id, exclude_source_ids)
    sql = "SELECT e.* FROM entities e WHERE e.file_path = ?"
    if clauses:
        sql += " AND " + " AND ".join(clauses)
    sql += " ORDER BY e.start_line"
    cur = await conn.execute(sql, [file_path, *scope_params])
    return [_row_to_entity(r) for r in await cur.fetchall()]


async def get_entities_by_files_batch(
    file_paths: list[str],
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    if not file_paths:
        return {}
    out: dict[str, list[dict]] = {fp: [] for fp in file_paths}
    for fp in file_paths:
        out[fp] = await get_entities_by_file(
            fp, source_id=source_id, exclude_source_ids=exclude_source_ids
        )
    return out


async def list_entities(
    types: list[str],
    *,
    source_id: str | None = None,
    path_prefix: str | None = None,
    exclude: list[str] | None = None,
    limit: int = 500,
) -> list[dict]:
    conn = await _get_conn()
    params: list[Any] = []
    sql = "SELECT e.* FROM entities e"
    if source_id:
        sql += " JOIN source_entities se ON se.entity_id = e.id AND se.source_id = ?"
        params.append(source_id)
    placeholders = ",".join("?" for _ in types)
    sql += (
        f" WHERE e.type IN ({placeholders})"
        " AND e.name IS NOT NULL AND e.file_path IS NOT NULL"
    )
    params.extend(types)
    if path_prefix:
        sql += " AND e.file_path LIKE ? || '%'"
        params.append(path_prefix)
    for sub in exclude or []:
        sql += " AND e.file_path NOT LIKE '%' || ? || '%'"
        params.append(sub)
    sql += " ORDER BY e.file_path, e.start_line LIMIT ?"
    params.append(limit)
    cur = await conn.execute(sql, params)
    return [_row_to_entity(r) for r in await cur.fetchall()]


async def find_entities_by_name(
    name: str,
    entity_type: str | None = None,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    conn = await _get_conn()
    sql = "SELECT e.* FROM entities e WHERE e.name = ?"
    params: list[Any] = [name]
    if entity_type is not None:
        sql += " AND e.type = ?"
        params.append(entity_type)
    clauses, scope_params = _scope_sql(source_id, exclude_source_ids)
    if clauses:
        sql += " AND " + " AND ".join(clauses)
        params.extend(scope_params)
    sql += " ORDER BY e.file_path, e.start_line"
    cur = await conn.execute(sql, params)
    return [_row_to_entity(r) for r in await cur.fetchall()]


async def get_entity_by_id(entity_id: str) -> dict | None:
    conn = await _get_conn()
    cur = await conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
    row = await cur.fetchone()
    return _row_to_entity(row) if row else None


async def _related(
    entity_id: str,
    rel_type: str | None,
    direction: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Entities related to entity_id; direction is from the OTHER node's view.

    direction="in": others pointing AT entity_id (callers/references).
    direction="out": others entity_id points at (callees).

    The related entity `e` is scoped to authorized sources so a caller/reference
    from a source the requester can't see is never returned.
    """
    conn = await _get_conn()
    if direction == "in":
        join = "JOIN relationships r ON r.source_id = e.id AND r.target_id = ?"
    else:
        join = "JOIN relationships r ON r.target_id = e.id AND r.source_id = ?"
    sql = f"SELECT e.*, r.type AS _rel FROM entities e {join}"
    params: list[Any] = [entity_id]
    where: list[str] = []
    if rel_type:
        where.append("r.type = ?")
        params.append(rel_type)
    clauses, scope_params = _scope_sql(source_id, exclude_source_ids)
    where.extend(clauses)
    params.extend(scope_params)
    if where:
        sql += " WHERE " + " AND ".join(where)
    cur = await conn.execute(sql, params)
    out = []
    for row in await cur.fetchall():
        d = dict(row)
        rel = d.pop("_rel")
        ent = {k: v for k, v in d.items() if v is not None and k != "updated_at"}
        ent["_rel_type"] = rel
        ent["_rel_direction"] = direction
        ent["_target_id"] = entity_id
        out.append(ent)
    return out


async def find_callers(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    return await _related(
        entity_id, "CALLS", "in",
        source_id=source_id, exclude_source_ids=exclude_source_ids,
    )


async def find_callees(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    return await _related(
        entity_id, "CALLS", "out",
        source_id=source_id, exclude_source_ids=exclude_source_ids,
    )


async def find_references(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Any incoming relationship — CALLS, INHERITS, IMPORTS, DEFINES."""
    return await _related(
        entity_id, None, "in",
        source_id=source_id, exclude_source_ids=exclude_source_ids,
    )


async def find_entity(file_path: str, line: int) -> dict | None:
    conn = await _get_conn()
    cur = await conn.execute(
        """
        SELECT * FROM entities
        WHERE file_path = ? AND start_line <= ? AND end_line >= ?
        ORDER BY (end_line - start_line) ASC LIMIT 1
        """,
        (file_path, line, line),
    )
    row = await cur.fetchone()
    return _row_to_entity(row) if row else None


async def _neighbors_of(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Depth-1 neighbors in both directions with _rel_type/_rel_direction.

    The neighbor `e` is scoped to authorized sources (positional params, so the
    scope clause + its params are repeated for each UNION leg).
    """
    conn = await _get_conn()
    clauses, scope_params = _scope_sql(source_id, exclude_source_ids)
    scope = (" AND " + " AND ".join(clauses)) if clauses else ""
    cur = await conn.execute(
        f"""
        SELECT e.*, r.type AS _rel, 'out' AS _dir FROM entities e
        JOIN relationships r ON r.target_id = e.id AND r.source_id = ?{scope}
        UNION ALL
        SELECT e.*, r.type AS _rel, 'in' AS _dir FROM entities e
        JOIN relationships r ON r.source_id = e.id AND r.target_id = ?{scope}
        """,
        [entity_id, *scope_params, entity_id, *scope_params],
    )
    out = []
    seen: set[tuple[str, str, str]] = set()
    for row in await cur.fetchall():
        d = dict(row)
        rel, direction = d.pop("_rel"), d.pop("_dir")
        key = (d["id"], rel, direction)
        if key in seen:
            continue
        seen.add(key)
        ent = {k: v for k, v in d.items() if v is not None and k != "updated_at"}
        ent["_rel_type"] = rel
        ent["_rel_direction"] = direction
        out.append(ent)
    return out


async def get_neighbors_batch(
    entity_ids: list[str],
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Depth-1 neighbors for many entities; keys only for entities that exist."""
    if not entity_ids:
        return {}
    conn = await _get_conn()
    out: dict[str, list[dict]] = {}
    for eid in entity_ids:
        cur = await conn.execute("SELECT 1 FROM entities WHERE id = ?", (eid,))
        if await cur.fetchone() is None:
            continue
        out[eid] = await _neighbors_of(
            eid, source_id=source_id, exclude_source_ids=exclude_source_ids
        )
    return out


async def traverse(entity_id: str, depth: int = 1, max_nodes: int = 50) -> list[dict]:
    """BFS to `depth`, same visit/queue semantics as the Neo4j adapter."""
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(entity_id, 0)]
    results: list[dict] = []

    while queue:
        current_id, d = queue.pop(0)
        if current_id in visited or d > depth:
            continue
        visited.add(current_id)

        entity = await get_entity_by_id(current_id)
        if entity is None:
            continue
        entity["_neighbors"] = []
        for ne in await _neighbors_of(current_id):
            entity["_neighbors"].append(ne)
            if ne["id"] not in visited:
                queue.append((ne["id"], d + 1))
                if len(queue) + len(visited) > max_nodes:
                    break
        results.append(entity)
        if len(results) >= max_nodes:
            break

    return results


async def set_community_id(entity_id: str, community_id):
    conn = await _get_conn()
    async with _lock():
        await conn.execute(
            "UPDATE entities SET community_id = ? WHERE id = ?",
            (community_id, entity_id),
        )
        await conn.commit()


async def get_community_ids(entity_ids: list[str]) -> set:
    if not entity_ids:
        return set()
    conn = await _get_conn()
    placeholders = ",".join("?" for _ in entity_ids)
    cur = await conn.execute(
        f"SELECT DISTINCT community_id FROM entities "
        f"WHERE id IN ({placeholders}) AND community_id IS NOT NULL",
        entity_ids,
    )
    return {r["community_id"] for r in await cur.fetchall()}


async def get_entities_by_community(community_id) -> list[dict]:
    conn = await _get_conn()
    cur = await conn.execute(
        "SELECT * FROM entities WHERE community_id = ?", (community_id,)
    )
    return [_row_to_entity(r) for r in await cur.fetchall()]


async def clear_all():
    conn = await _get_conn()
    async with _lock():
        for table in ("relationships", "source_entities", "entities",
                      "sources", "communities"):
            await conn.execute(f"DELETE FROM {table}")
        await conn.commit()


async def delete_file_entities(file_path: str):
    conn = await _get_conn()
    async with _lock():
        await conn.execute(
            "DELETE FROM relationships WHERE source_id IN "
            "(SELECT id FROM entities WHERE file_path = ?) "
            "OR target_id IN (SELECT id FROM entities WHERE file_path = ?)",
            (file_path, file_path),
        )
        await conn.execute(
            "DELETE FROM source_entities WHERE entity_id IN "
            "(SELECT id FROM entities WHERE file_path = ?)",
            (file_path,),
        )
        await conn.execute(
            "DELETE FROM entities WHERE file_path = ?", (file_path,)
        )
        await conn.commit()


async def delete_entities_by_file(source_id: str, file_path: str) -> int:
    """Delete a file's entities within a source; returns the node count."""
    conn = await _get_conn()
    async with _lock():
        cur = await conn.execute(
            "SELECT e.id FROM entities e "
            "JOIN source_entities se ON se.entity_id = e.id "
            "WHERE se.source_id = ? AND e.file_path = ?",
            (source_id, file_path),
        )
        ids = [r["id"] for r in await cur.fetchall()]
        if not ids:
            await conn.commit()
            return 0
        placeholders = ",".join("?" for _ in ids)
        await conn.execute(
            f"DELETE FROM relationships WHERE source_id IN ({placeholders}) "
            f"OR target_id IN ({placeholders})",
            ids + ids,
        )
        await conn.execute(
            f"DELETE FROM source_entities WHERE entity_id IN ({placeholders})",
            ids,
        )
        await conn.execute(
            f"DELETE FROM entities WHERE id IN ({placeholders})", ids
        )
        await conn.commit()
        return len(ids)


async def pull_graph(source_id: str | None = None) -> list[dict]:
    """Entities (id/name/type/file_path) with outgoing `_rels`, for Louvain."""
    conn = await _get_conn()
    if source_id:
        cur = await conn.execute(
            "SELECT e.id, e.name, e.type, e.file_path FROM entities e "
            "JOIN source_entities se ON se.entity_id = e.id "
            "WHERE se.source_id = ?",
            (source_id,),
        )
    else:
        cur = await conn.execute(
            "SELECT id, name, type, file_path FROM entities"
        )
    nodes = [dict(r) for r in await cur.fetchall()]
    if not nodes:
        return []
    ids = [n["id"] for n in nodes]
    rels_by_source: dict[str, list[dict]] = {}
    CHUNK = 500
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        cur = await conn.execute(
            f"SELECT source_id, target_id, type FROM relationships "
            f"WHERE source_id IN ({placeholders})",
            chunk,
        )
        for r in await cur.fetchall():
            rels_by_source.setdefault(r["source_id"], []).append(
                {"type": r["type"], "target_id": r["target_id"]}
            )
    for n in nodes:
        n["_rels"] = rels_by_source.get(n["id"], [])
    return nodes


async def set_community_ids_batch(mapping: dict[str, str]):
    if not mapping:
        return
    conn = await _get_conn()
    async with _lock():
        await conn.executemany(
            "UPDATE entities SET community_id = ? WHERE id = ?",
            [(cid, eid) for eid, cid in mapping.items()],
        )
        await conn.commit()


async def set_centralities_batch(mapping: dict[str, float]):
    if not mapping:
        return
    conn = await _get_conn()
    async with _lock():
        await conn.executemany(
            "UPDATE entities SET centrality = ? WHERE id = ?",
            [(float(c), eid) for eid, c in mapping.items()],
        )
        await conn.commit()


async def upsert_community_summaries_batch(summaries: dict[str, str]):
    if not summaries:
        return
    conn = await _get_conn()
    async with _lock():
        await conn.executemany(
            "INSERT INTO communities (id, summary) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET summary = excluded.summary",
            list(summaries.items()),
        )
        await conn.commit()


async def set_community_embeddings_batch(mapping: dict[str, list[float]]):
    """Set embeddings on EXISTING community rows (Neo4j MATCH parity)."""
    if not mapping:
        return
    conn = await _get_conn()
    async with _lock():
        await conn.executemany(
            "UPDATE communities SET embedding = ? WHERE id = ?",
            [(json.dumps(emb), cid) for cid, emb in mapping.items()],
        )
        await conn.commit()


async def delete_source_communities(source_id: str):
    conn = await _get_conn()
    async with _lock():
        await conn.execute(
            "DELETE FROM communities WHERE id LIKE ? || '#%'", (source_id,)
        )
        await conn.commit()


async def get_community_summaries_map(community_ids: list[str]) -> dict[str, str]:
    if not community_ids:
        return {}
    conn = await _get_conn()
    ids = list(community_ids)
    placeholders = ",".join("?" for _ in ids)
    cur = await conn.execute(
        f"SELECT id, summary FROM communities WHERE id IN ({placeholders})",
        ids,
    )
    return {r["id"]: r["summary"] for r in await cur.fetchall()}


async def get_all_community_summaries() -> dict[str, str]:
    conn = await _get_conn()
    cur = await conn.execute(
        "SELECT id, summary FROM communities WHERE summary IS NOT NULL"
    )
    return {r["id"]: r["summary"] for r in await cur.fetchall()}


async def load_community_embeddings() -> dict[str, list[float]]:
    conn = await _get_conn()
    cur = await conn.execute(
        "SELECT id, embedding FROM communities WHERE embedding IS NOT NULL"
    )
    return {r["id"]: json.loads(r["embedding"]) for r in await cur.fetchall()}
