import asyncio
import functools
import logging
import os
import socket
import time
from collections import defaultdict
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncSession, AsyncDriver
from neo4j.addressing import Address
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from treeloom.config import require_env
from treeloom.infrastructure.tracing import get_tracer

from treeloom.domain.graph import is_valid_rel_type, require_valid_rel_type

logger = logging.getLogger(__name__)
_tracer = get_tracer("treeloom.neo4j")

NEO4J_URI = require_env("NEO4J_URI")
NEO4J_USER = require_env("NEO4J_USER")
NEO4J_PASSWORD = require_env("NEO4J_PASSWORD")

_driver: AsyncDriver | None = None

# ── DNS resolver cache ────────────────────────────────────────────────
# Without a resolver, the Neo4j driver calls getaddrinfo() for *every*
# new connection — under high concurrency (8 workers × graph reindexing)
# this triggers rate-limit throttles (e.g. 1,000 DNS/min).  We resolve
# once per TTL window and hand the driver a stable Address iterator.
_DNS_CACHE_TTL = float(os.environ.get("NEO4J_DNS_CACHE_TTL", "300"))
_dns_cache: dict[tuple[str, int], tuple[list[Address], float]] = {}


def _resolver(address: Address):
    """Caching DNS resolver for the Neo4j Bolt driver.

    Called every time the driver creates a TCP connection.  We resolve
    the host once and cache the result for `NEO4J_DNS_CACHE_TTL` seconds.
    """
    key = (address.host, address.port)
    now = time.monotonic()
    cached = _dns_cache.get(key)
    if cached is not None:
        addrs, expires = cached
        if now < expires:
            return addrs
    # Sync getaddrinfo is fine — Neo4j calls resolver in a thread-pool.
    infos = socket.getaddrinfo(
        address.host, address.port,
        type=socket.SOCK_STREAM,
    )
    resolved = [Address(info[4][:2]) for info in infos]
    _dns_cache[key] = (resolved, now + _DNS_CACHE_TTL)
    logger.debug("resolved %s:%d → %s (ttl=%ds)",
                 address.host, address.port, resolved, _DNS_CACHE_TTL)
    return resolved

# Reconnect-and-retry policy for Neo4j calls. The Bolt driver's built-in
# retry only kicks in for `session.execute_read/write(...)` callable form;
# our code uses `session.run()` directly, so we wrap the public functions
# in our own retry that also nukes the shared driver on disconnect — the
# `ServiceUnavailable: defunct connection` errors we see in production come
# from a stale TCP socket the driver clings to after the server drops it.
_NEO4J_RETRYABLE = (ServiceUnavailable, SessionExpired, TransientError)
_NEO4J_RETRY_ATTEMPTS = int(os.environ.get("NEO4J_RETRY_ATTEMPTS", "3"))
_NEO4J_RETRY_BASE_DELAY = float(os.environ.get("NEO4J_RETRY_BASE_DELAY", "1.0"))
# Hard ceiling per Neo4j operation. Without this, a half-dead Bolt
# connection (TCP alive, server unresponsive) hangs forever — the driver
# raises no exception, asyncio.wait_for at the outer file level appears
# unable to cancel the cleanup, and the indexer stalls indefinitely.
# Treat operations exceeding this budget the same as ServiceUnavailable:
# close the driver, reconnect, and retry.
_NEO4J_OP_TIMEOUT = float(os.environ.get("NEO4J_OP_TIMEOUT_SECONDS", "30"))


def _retry_on_disconnect(fn):
    """Retry a Neo4j-using coroutine on transient driver errors or hangs.

    On a retryable exception OR an operation that exceeds
    `NEO4J_OP_TIMEOUT_SECONDS`, sleeps with exponential backoff and
    retries.  The connection pool (`max_connection_lifetime=300`, keep-
    alive probes) handles stale connections on its own — we no longer
    close the shared driver on every transient error, because that
    triggers a fresh DNS resolution per recreated driver and cascades
    across all concurrent workers under load (8 workers × graph
    reindexing → DNS rate-limit throttles).

    Caps at `NEO4J_RETRY_ATTEMPTS` attempts before re-raising.
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        last_exc: BaseException | None = None
        with _tracer.start_as_current_span(
            f"neo4j.{fn.__name__}",
            attributes={"db.system": "neo4j"},
        ) as _span:
         for attempt in range(_NEO4J_RETRY_ATTEMPTS):
            try:
                return await asyncio.wait_for(
                    fn(*args, **kwargs),
                    timeout=_NEO4J_OP_TIMEOUT,
                )
            except (*_NEO4J_RETRYABLE, asyncio.TimeoutError) as exc:
                last_exc = exc
                kind = "hang" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
                try:
                    _span.add_event("retry", {"attempt": attempt + 1, "kind": kind})
                except Exception:
                    pass
                logger.warning(
                    "neo4j %s on %s (attempt %d/%d): %s",
                    kind, fn.__name__,
                    attempt + 1, _NEO4J_RETRY_ATTEMPTS, exc,
                )
                if attempt < _NEO4J_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(_NEO4J_RETRY_BASE_DELAY * (2 ** attempt))
         assert last_exc is not None
         try:
            from opentelemetry.trace import Status, StatusCode
            _span.set_status(Status(StatusCode.ERROR, str(last_exc)))
            _span.record_exception(last_exc)
         except Exception:
            pass
         raise last_exc
    return wrapper


def _get_session() -> AsyncSession:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            # Connection timeouts. Without these, a network blip causes
            # the driver to wait minutes before noticing — see the
            # `_retry_on_disconnect` doc for context.
            connection_timeout=15.0,
            connection_acquisition_timeout=30.0,
            keep_alive=True,
            max_connection_lifetime=300,
            # Pool sizing — prevents unbounded connection creation (each
            # connection = a DNS resolution via the resolver).  Default is
            # 400; under 8 concurrent workers each doing graph indexing, a
            # smaller pool is safer.  Keep enough headroom that transient
            # bursts don't exhaust the pool.
            max_connection_pool_size=50,
            # Caching DNS resolver — avoids a getaddrinfo() call per new
            # connection.  Under graph reindexing (100 sources × thousands
            # of files), the default behaviour hits DNS rate-limits.
            resolver=_resolver,
        )
    return _driver.session()


async def close():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


# Schema (constraints + indexes) the graph relies on. All are
# `IF NOT EXISTS`, so `ensure_schema()` is idempotent and cheap to run on
# every startup. Without the `Entity(file_path)` / `Entity(name)` indexes,
# `get_entities_by_file` / `find_entities_by_name` full-scan every Entity
# node — fast on a small graph, but a multi-second hang once the graph
# spans millions of nodes, which silently breaks `search_code`.
_SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT source_id_unique IF NOT EXISTS "
    "FOR (s:Source) REQUIRE s.id IS UNIQUE",
    # Without this, `upsert_community_summaries_batch` label-scans every
    # Community node per MERGE row — ~1k communities against a ~30k-node
    # label blew NEO4J_OP_TIMEOUT and silently left a source with zero
    # community summaries (post-index signals are enrichment-only, so the
    # index job still finalized `done`).
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS "
    "FOR (c:Community) REQUIRE c.id IS UNIQUE",
    "CREATE INDEX entity_file_path IF NOT EXISTS FOR (e:Entity) ON (e.file_path)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
)


@_retry_on_disconnect
async def ensure_schema():
    """Create the constraints and indexes the graph relies on (idempotent).

    Index population is online/non-blocking on existing data, so this is
    safe to call at startup even against a large, already-populated graph.
    """
    async with _get_session() as session:
        for stmt in _SCHEMA_STATEMENTS:
            await (await session.run(stmt)).consume()
    logger.info("neo4j schema ensured (%d statements)", len(_SCHEMA_STATEMENTS))


@_retry_on_disconnect
async def upsert_entity(entity: dict):
    async with _get_session() as session:
        await session.run(
            """
            MERGE (e:Entity {id: $id})
            SET e += $props, e.updated_at = timestamp()
            """,
            id=entity["id"],
            props={
                "type": entity["type"],
                "name": entity["name"],
                "file_path": entity["file_path"],
                "start_line": entity.get("start_line", 0),
                "end_line": entity.get("end_line", 0),
                "signature": entity.get("signature", ""),
                "language": entity.get("language", ""),
            },
        )


@_retry_on_disconnect
async def upsert_relationship(source_id: str, target_id: str, rel_type: str):
    """Create one CALLS/IMPORTS/... edge between two existing entities.

    `rel_type` is interpolated into the Cypher below because Neo4j has no
    placeholder for a relationship type — it cannot be a query parameter. So
    it is validated first: unvalidated, a caller supplying

        FOO]->(x) DETACH DELETE x //

    closes the MERGE pattern and appends arbitrary Cypher (CWE-943).

    `store_graph` has validated its types since it was written; this function
    did not, and the asymmetry is the whole bug. It has no callers in the tree
    today — it is reachable only through the `treeloom.graph_store` re-export —
    so this was not exploitable, but it is one caller away from being so, and
    relationship types ultimately derive from tree-sitter output over indexed
    source, which is attacker-influenced for any repo the operator indexes.
    """
    require_valid_rel_type(rel_type)
    async with _get_session() as session:
        await session.run(
            f"""
            MATCH (s:Entity {{id: $source_id}})
            MATCH (t:Entity {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.updated_at = timestamp()
            """,
            source_id=source_id,
            target_id=target_id,
        )


# Cypher relationship-type names must be a valid identifier. The graph
# extractor emits a small fixed set (CALLS, IMPORTS, INHERITS, ...) but we
# validate defensively before interpolating into the Cypher string. The rule
# itself lives in the domain so the SQLite backend applies the same one.
def _is_valid_rel_type(t: str) -> bool:
    return is_valid_rel_type(t)


@_retry_on_disconnect
async def store_graph(entities: list[dict], relationships: list[dict], source_id: str | None = None):
    """Persist a file's extracted graph.

    Batched via UNWIND: one round-trip for all entities (plus their
    CONTAINS link to the Source if `source_id` is provided), and one
    round-trip per distinct relationship type — instead of one
    round-trip per entity + per relationship as the old loop did.

    For files with hundreds of entities and relationships this collapses
    300+ round-trips down to ~5, which is the practical bottleneck for
    large repos when Neo4j is under any kind of resource pressure.
    """
    async with _get_session() as session:
        if entities:
            entity_params = [
                {
                    "id": e["id"],
                    "type": e["type"],
                    "name": e["name"],
                    "file_path": e["file_path"],
                    "start_line": e.get("start_line", 0),
                    "end_line": e.get("end_line", 0),
                    "signature": e.get("signature", ""),
                    "language": e.get("language", ""),
                }
                for e in entities
            ]
            if source_id:
                await session.run(
                    """
                    MERGE (s:Source {id: $source_id})
                    WITH s
                    UNWIND $entities AS e
                    MERGE (entity:Entity {id: e.id})
                    SET entity.type = e.type,
                        entity.name = e.name,
                        entity.file_path = e.file_path,
                        entity.start_line = e.start_line,
                        entity.end_line = e.end_line,
                        entity.signature = e.signature,
                        entity.language = e.language,
                        entity.updated_at = timestamp()
                    MERGE (s)-[:CONTAINS]->(entity)
                    """,
                    source_id=source_id,
                    entities=entity_params,
                )
            else:
                await session.run(
                    """
                    UNWIND $entities AS e
                    MERGE (entity:Entity {id: e.id})
                    SET entity.type = e.type,
                        entity.name = e.name,
                        entity.file_path = e.file_path,
                        entity.start_line = e.start_line,
                        entity.end_line = e.end_line,
                        entity.signature = e.signature,
                        entity.language = e.language,
                        entity.updated_at = timestamp()
                    """,
                    entities=entity_params,
                )

        # Group by relationship type so each query uses a static, validated
        # rel-type literal (Cypher requires this; you can't parameterize a
        # relationship type without APOC).
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in relationships:
            t = r["type"]
            if not _is_valid_rel_type(t):
                logger.warning(
                    "store_graph: skipping relationship with invalid type %r", t,
                )
                continue
            by_type[t].append({"source_id": r["source_id"], "target_id": r["target_id"]})

        for rel_type, rels in by_type.items():
            await session.run(
                f"""
                UNWIND $rels AS rel
                MERGE (s:Entity {{id: rel.source_id}})
                MERGE (t:Entity {{id: rel.target_id}})
                SET t.type = CASE WHEN t.type IS NULL THEN 'ExternalModule' ELSE t.type END
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.updated_at = timestamp()
                """,
                rels=rels,
            )


@_retry_on_disconnect
async def upsert_source(source: dict):
    async with _get_session() as session:
        await session.run(
            """
            MERGE (s:Source {id: $id})
            SET s += $props, s.updated_at = timestamp()
            """,
            id=source["id"],
            props={
                "path": source.get("path", ""),
                "url": source.get("url", ""),
                "branch": source.get("branch", ""),
                "indexed_at": source.get("indexed_at", 0),
                "file_count": source.get("file_count", 0),
                "chunk_count": source.get("chunk_count", 0),
                "commit_sha": source.get("commit_sha", ""),
            },
        )


@_retry_on_disconnect
async def list_sources() -> list[dict]:
    async with _get_session() as session:
        result = await session.run(
            "MATCH (s:Source) RETURN s ORDER BY s.indexed_at DESC"
        )
        records = await result.fetch(1000)
        return [dict(r["s"]) for r in records]


@_retry_on_disconnect
async def entity_counts_by_source() -> dict[str, int]:
    """Return {source_id: entity_count} for every source in one query.

    Cheap fleet-rollup helper: a single grouped traversal over
    (:Source)-[:CONTAINS]->(:Entity), not one query per source.
    """
    async with _get_session() as session:
        result = await session.run(
            "MATCH (s:Source)-[:CONTAINS]->(e:Entity) "
            "RETURN s.id AS id, count(e) AS n"
        )
        records = await result.fetch(10000)
        return {r["id"]: r["n"] for r in records if r["id"]}


@_retry_on_disconnect
async def delete_source(source_id: str):
    async with _get_session() as session:
        # Delete entities owned exclusively by this source.
        # Skip entities still CONTAINED by another Source.
        await session.run(
            """
            MATCH (s:Source {id: $source_id})-[:CONTAINS]->(e:Entity)
            WHERE NOT EXISTS {
                MATCH (other:Source)-[:CONTAINS]->(e)
                WHERE other.id <> $source_id
            }
            DETACH DELETE e
            """,
            source_id=source_id,
        )
        # Drop the Source node itself.
        await session.run(
            "MATCH (s:Source {id: $source_id}) DETACH DELETE s",
            source_id=source_id,
        )


def _scope_predicates(
    var: str,
    source_id: str | None,
    exclude_source_ids: list[str] | None,
) -> tuple[list[str], dict]:
    """Build WHERE predicates that scope a node var to authorized sources.

    Entities carry no source_id property — the link is (:Source)-[:CONTAINS]->.
    For a single-source query we require the node be CONTAINED by that source;
    for a shared (all_access) query we exclude any node CONTAINED by a denied
    source (conservatively: if it lives in *any* excluded source it's hidden,
    even if also present in an allowed one). Returns (clauses, params); params
    use distinct keys (`scope_*`) so they never collide with the caller's.
    """
    clauses: list[str] = []
    params: dict = {}
    if source_id:
        clauses.append(
            f"EXISTS {{ MATCH (:Source {{id: $scope_source_id}})-[:CONTAINS]->({var}) }}"
        )
        params["scope_source_id"] = source_id
    if exclude_source_ids:
        clauses.append(
            f"NOT EXISTS {{ MATCH (xs:Source)-[:CONTAINS]->({var}) "
            f"WHERE xs.id IN $scope_exclude }}"
        )
        params["scope_exclude"] = list(exclude_source_ids)
    return clauses, params


async def get_entities_by_files_batch(
    file_paths: list[str],
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Entities for many files in one round-trip, keyed by file_path.

    Replaces calling `get_entities_by_file` once per file (one Cypher query
    each) when the caller has a batch of hit files. `Entity(file_path)` is
    covered by the `entity_file_path` index, so the UNWIND stays index-backed.
    Files with no entities map to an empty list.
    """
    if not file_paths:
        return {}
    clauses, scope_params = _scope_predicates("e", source_id, exclude_source_ids)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with _get_session() as session:
        result = await session.run(
            f"""
            UNWIND $file_paths AS fp
            MATCH (e:Entity {{file_path: fp}})
            {where}
            WITH fp, e ORDER BY e.start_line
            RETURN fp, collect(e) AS entities
            """,
            file_paths=list(file_paths),
            **scope_params,
        )
        records = await result.fetch(len(file_paths) + 1)
        out: dict[str, list[dict]] = {fp: [] for fp in file_paths}
        for r in records:
            out[r["fp"]] = [dict(e) for e in r["entities"]]
        return out


@_retry_on_disconnect
async def get_entities_by_file(file_path: str, source_id: str | None = None) -> list[dict]:
    async with _get_session() as session:
        if source_id:
            result = await session.run(
                """
                MATCH (s:Source {id: $source_id})-[:CONTAINS]->(e:Entity {file_path: $file_path})
                RETURN e ORDER BY e.start_line
                """,
                file_path=file_path,
                source_id=source_id,
            )
        else:
            result = await session.run(
                "MATCH (e:Entity {file_path: $file_path}) RETURN e ORDER BY e.start_line",
                file_path=file_path,
            )
        records = await result.fetch(1000)
        return [dict(r["e"]) for r in records]


@_retry_on_disconnect
async def list_entities(
    types: list[str],
    *,
    source_id: str | None = None,
    path_prefix: str | None = None,
    exclude: list[str] | None = None,
    limit: int = 500,
) -> list[dict]:
    """List entities of the given types, optionally scoped to a source or a
    file-path prefix. Used by benchmark query generation to pull candidate
    Class/Function/Method entities (with name/signature/lines/file_path).

    `exclude` drops entities whose file_path CONTAINS any of the given
    substrings (e.g. "/docs/", "/tests/") — filtered in-query so `limit`
    applies after exclusion, not before.
    """
    params: dict[str, Any] = {"types": types, "limit": limit, "excludes": exclude or []}
    match = "(s:Source {id: $source_id})-[:CONTAINS]->(e:Entity)" if source_id else "(e:Entity)"
    if source_id:
        params["source_id"] = source_id
    where = ["e.type IN $types", "e.name IS NOT NULL", "e.file_path IS NOT NULL"]
    if path_prefix:
        where.append("e.file_path STARTS WITH $prefix")
        params["prefix"] = path_prefix
    # none(...) over an empty list is true, so this is a no-op when excludes=[].
    where.append("none(p IN $excludes WHERE e.file_path CONTAINS p)")
    cypher = (
        f"MATCH {match} WHERE {' AND '.join(where)} "
        "RETURN e ORDER BY e.file_path, e.start_line LIMIT $limit"
    )
    async with _get_session() as session:
        result = await session.run(cypher, **params)
        records = await result.fetch(limit)
        return [dict(r["e"]) for r in records]


@_retry_on_disconnect
async def find_entities_by_name(
    name: str,
    entity_type: str | None = None,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    clauses, scope_params = _scope_predicates("e", source_id, exclude_source_ids)
    where = ["($type IS NULL OR e.type = $type)"] + clauses
    params = {"name": name, "type": entity_type, **scope_params}
    query = f"""
        MATCH (e:Entity {{name: $name}})
        WHERE {' AND '.join(where)}
        RETURN e ORDER BY e.file_path, e.start_line
    """
    async with _get_session() as session:
        result = await session.run(query, **params)
        records = await result.fetch(1000)
        return [dict(r["e"]) for r in records]


@_retry_on_disconnect
async def find_callers(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    clauses, scope_params = _scope_predicates(
        "caller", source_id, exclude_source_ids
    )
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with _get_session() as session:
        result = await session.run(
            f"""
            MATCH (caller:Entity)-[:CALLS]->(e:Entity {{id: $id}})
            {where}
            RETURN caller, "CALLS" AS rel_type
            """,
            id=entity_id,
            **scope_params,
        )
        records = await result.fetch(1000)
        out = []
        for r in records:
            ent = dict(r["caller"])
            ent["_rel_type"] = r["rel_type"]
            ent["_rel_direction"] = "in"
            ent["_target_id"] = entity_id
            out.append(ent)
        return out


@_retry_on_disconnect
async def find_callees(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    clauses, scope_params = _scope_predicates(
        "callee", source_id, exclude_source_ids
    )
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with _get_session() as session:
        result = await session.run(
            f"""
            MATCH (e:Entity {{id: $id}})-[:CALLS]->(callee:Entity)
            {where}
            RETURN callee, "CALLS" AS rel_type
            """,
            id=entity_id,
            **scope_params,
        )
        records = await result.fetch(1000)
        out = []
        for r in records:
            ent = dict(r["callee"])
            ent["_rel_type"] = r["rel_type"]
            ent["_rel_direction"] = "out"
            ent["_target_id"] = entity_id
            out.append(ent)
        return out


@_retry_on_disconnect
async def find_references(
    entity_id: str,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Any incoming relationship — CALLS, INHERITS, IMPORTS, DEFINES."""
    clauses, scope_params = _scope_predicates(
        "src", source_id, exclude_source_ids
    )
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with _get_session() as session:
        result = await session.run(
            f"""
            MATCH (src:Entity)-[r]->(e:Entity {{id: $id}})
            {where}
            RETURN src, type(r) AS rel_type
            """,
            id=entity_id,
            **scope_params,
        )
        records = await result.fetch(1000)
        out = []
        for r in records:
            ent = dict(r["src"])
            ent["_rel_type"] = r["rel_type"]
            ent["_rel_direction"] = "in"
            ent["_target_id"] = entity_id
            out.append(ent)
        return out


@_retry_on_disconnect
async def find_entity(file_path: str, line: int) -> dict | None:
    async with _get_session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {file_path: $file_path})
            WHERE e.start_line <= $line AND e.end_line >= $line
            RETURN e
            ORDER BY (e.end_line - e.start_line) ASC
            LIMIT 1
            """,
            file_path=file_path,
            line=line,
        )
        records = await result.fetch(1000)
        if records:
            return dict(records[0]["e"])
        return None


@_retry_on_disconnect
async def get_neighbors_batch(
    entity_ids: list[str],
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Return depth-1 neighbors for many entities in a single round-trip.

    Replaces calling `traverse()` once per entity. The per-entity loop
    issued one Cypher query per node (and recursed into each neighbor),
    which becomes thousands of sequential round-trips when a hit file
    holds hundreds of entities (e.g. vendored deps) — the dominant cost
    in `graph_search` once the graph spans many large sources.

    Each neighbor dict carries `_rel_type`/`_rel_direction` exactly as
    `traverse()` populated them. `MATCH (e:Entity {id})` is index-backed
    by `entity_id_unique`, so the single `UNWIND` query stays fast.
    """
    if not entity_ids:
        return {}
    clauses, scope_params = _scope_predicates(
        "related", source_id, exclude_source_ids
    )
    extra = (" AND " + " AND ".join(clauses)) if clauses else ""
    async with _get_session() as session:
        result = await session.run(
            f"""
            UNWIND $ids AS eid
            MATCH (e:Entity {{id: eid}})
            OPTIONAL MATCH (e)-[r]-(related:Entity)
            WHERE related.id IS NOT NULL{extra}
            RETURN eid, collect(DISTINCT {{
                rel: type(r),
                direction: CASE WHEN startNode(r).id = eid THEN 'out' ELSE 'in' END,
                entity: related
            }}) AS neighbors
            """,
            ids=list(entity_ids),
            **scope_params,
        )
        records = await result.fetch(len(entity_ids) + 1)
        out: dict[str, list[dict]] = {}
        for rec in records:
            nbs: list[dict] = []
            for nb in rec.get("neighbors", []):
                if nb and nb["entity"]:
                    ne = dict(nb["entity"])
                    ne["_rel_type"] = nb["rel"]
                    ne["_rel_direction"] = nb["direction"]
                    nbs.append(ne)
            out[rec["eid"]] = nbs
        return out


@_retry_on_disconnect
async def traverse(entity_id: str, depth: int = 1, max_nodes: int = 50) -> list[dict]:
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(entity_id, 0)]
    results: list[dict] = []

    while queue:
        current_id, d = queue.pop(0)
        if current_id in visited or d > depth:
            continue
        visited.add(current_id)

        async with _get_session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id})
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                WHERE related.id IS NOT NULL
                RETURN e, collect(DISTINCT {rel: type(r), direction: CASE WHEN startNode(r).id = $id THEN 'out' ELSE 'in' END, entity: related}) AS neighbors
                """,
                id=current_id,
            )
            records = await result.fetch(1000)
            if records:
                rec = records[0]
                entity = dict(rec["e"])
                entity["_neighbors"] = []
                for nb in rec.get("neighbors", []):
                    if nb and nb["entity"]:
                        ne = dict(nb["entity"])
                        ne["_rel_type"] = nb["rel"]
                        ne["_rel_direction"] = nb["direction"]
                        entity["_neighbors"].append(ne)
                        if ne["id"] not in visited:
                            queue.append((ne["id"], d + 1))
                            if len(queue) + len(visited) > max_nodes:
                                break
                results.append(entity)
        if len(results) >= max_nodes:
            break

    return results


@_retry_on_disconnect
async def set_community_id(entity_id: str, community_id: int):
    async with _get_session() as session:
        await session.run(
            "MATCH (e:Entity {id: $id}) SET e.community_id = $community_id",
            id=entity_id,
            community_id=community_id,
        )


@_retry_on_disconnect
async def get_community_ids(entity_ids: list[str]) -> set[int]:
    async with _get_session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)
            WHERE e.id IN $ids AND e.community_id IS NOT NULL
            RETURN DISTINCT e.community_id AS community_id
            """,
            ids=entity_ids,
        )
        records = await result.fetch(1000)
        return {r["community_id"] for r in records}


@_retry_on_disconnect
async def get_entities_by_community(community_id: int) -> list[dict]:
    async with _get_session() as session:
        result = await session.run(
            "MATCH (e:Entity {community_id: $community_id}) RETURN e",
            community_id=community_id,
        )
        records = await result.fetch(1000)
        return [dict(r["e"]) for r in records]


@_retry_on_disconnect
async def clear_all():
    async with _get_session() as session:
        await session.run("MATCH (n) DETACH DELETE n")


@_retry_on_disconnect
async def delete_file_entities(file_path: str):
    async with _get_session() as session:
        await session.run(
            """
            MATCH (e:Entity {file_path: $file_path})
            DETACH DELETE e
            """,
            file_path=file_path,
        )


@_retry_on_disconnect
async def delete_entities_by_file(source_id: str, file_path: str) -> int:
    """Delete all entities for a specific file within a source.

    Scoped through the (:Source)-[:CONTAINS]->(:Entity) relationship, which is
    how a source owns its entities. This previously matched on an
    `e.source_id` PROPERTY that store_graph never writes — the entity upsert
    sets type/name/file_path/start_line/end_line/signature/language and then
    MERGEs the CONTAINS edge, and that edge is the whole association. So the
    query matched nothing, every time, silently.

    The effect was a graph that only ever grew: a file deleted through the
    incremental webhook path kept its Module/Class/Function nodes forever, so
    find_definition returned symbols that no longer exist, traversal returned
    stale neighbours, and community detection ran over phantom entities. It
    also left the structure of deleted files queryable.

    Args:
        source_id: The source identifier.
        file_path: The file path to delete entities for.

    Returns:
        Number of deleted entities (nodes).
    """
    async with _get_session() as session:
        result = await session.run(
            """
            MATCH (s:Source {id: $source_id})-[:CONTAINS]->(e:Entity)
            WHERE e.file_path = $file_path
            DETACH DELETE e
            """,
            source_id=source_id,
            file_path=file_path,
        )
        summary = await result.consume()
        return summary.counters.nodes_deleted


@_retry_on_disconnect
async def pull_graph(source_id: str | None = None) -> list[dict]:
    """Pull entities (with their outgoing rels) into memory.

    When `source_id` is given, scope to that source's entities — community
    detection runs per-source, not over the whole merged graph. Drains the
    full result set: the old `fetch(1000)` cap silently dropped all but the
    first 1000 nodes, so detection only ever saw a sliver of the graph.
    """
    # Project only the fields community detection / centrality / summaries
    # need (id, name, type, file_path) plus outgoing rel targets. Returning
    # full `e` nodes shipped every property (signatures, summaries, …) and
    # pushed large sources past NEO4J_OP_TIMEOUT.
    async with _get_session() as session:
        if source_id:
            result = await session.run(
                """
                MATCH (s:Source {id: $source_id})-[:CONTAINS]->(e:Entity)
                OPTIONAL MATCH (e)-[r]->(t:Entity)
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       e.file_path AS file_path,
                       collect(DISTINCT {type: type(r), target_id: t.id}) AS rels
                """,
                source_id=source_id,
            )
        else:
            result = await session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]->(t:Entity)
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       e.file_path AS file_path,
                       collect(DISTINCT {type: type(r), target_id: t.id}) AS rels
                """
            )
        nodes = []
        while True:
            batch = await result.fetch(1000)
            if not batch:
                break
            for rec in batch:
                nodes.append({
                    "id": rec["id"],
                    "name": rec["name"],
                    "type": rec["type"],
                    "file_path": rec["file_path"],
                    "_rels": [
                        {"type": r["type"], "target_id": r["target_id"]}
                        for r in rec.get("rels", [])
                        if r["type"] is not None
                    ],
                })
        return nodes


@_retry_on_disconnect
async def set_community_ids_batch(mapping: dict[str, str]):
    """Set `community_id` on many entities in one round-trip (UNWIND)."""
    if not mapping:
        return
    rows = [{"id": k, "cid": v} for k, v in mapping.items()]
    async with _get_session() as session:
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {id: row.id})
            SET e.community_id = row.cid
            """,
            rows=rows,
        )


@_retry_on_disconnect
async def set_centralities_batch(mapping: dict[str, float]):
    """Set `centrality` on many entities in one round-trip (UNWIND)."""
    if not mapping:
        return
    rows = [{"id": k, "c": float(v)} for k, v in mapping.items()]
    async with _get_session() as session:
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {id: row.id})
            SET e.centrality = row.c
            """,
            rows=rows,
        )


@_retry_on_disconnect
async def upsert_community_summaries_batch(summaries: dict[str, str]):
    """MERGE Community nodes and set their summaries in one round-trip."""
    if not summaries:
        return
    rows = [{"id": k, "summary": v} for k, v in summaries.items()]
    async with _get_session() as session:
        await session.run(
            """
            UNWIND $rows AS row
            MERGE (c:Community {id: row.id})
            SET c.summary = row.summary
            """,
            rows=rows,
        )


@_retry_on_disconnect
async def set_community_embeddings_batch(mapping: dict[str, list[float]]):
    """Set `embedding` on many Community nodes in one round-trip (UNWIND)."""
    if not mapping:
        return
    rows = [{"id": k, "emb": v} for k, v in mapping.items()]
    async with _get_session() as session:
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Community {id: row.id})
            SET c.embedding = row.emb
            """,
            rows=rows,
        )


async def _drain(result) -> list:
    """Fetch all records from an async result (no arbitrary cap)."""
    out: list = []
    while True:
        batch = await result.fetch(1000)
        if not batch:
            break
        out.extend(batch)
    return out


@_retry_on_disconnect
async def get_entity_by_id(entity_id: str) -> dict | None:
    async with _get_session() as session:
        result = await session.run(
            "MATCH (e:Entity {id: $id}) RETURN e", id=entity_id
        )
        records = await result.fetch(1)
        return dict(records[0]["e"]) if records else None


@_retry_on_disconnect
async def get_community_summaries_map(community_ids: list[str]) -> dict[str, str]:
    if not community_ids:
        return {}
    async with _get_session() as session:
        result = await session.run(
            """
            MATCH (c:Community)
            WHERE c.id IN $ids
            RETURN c.id AS id, c.summary AS summary
            """,
            ids=list(community_ids),
        )
        records = await result.fetch(1000)
        return {r["id"]: r["summary"] for r in records}


@_retry_on_disconnect
async def get_all_community_summaries() -> dict[str, str]:
    async with _get_session() as session:
        result = await session.run(
            "MATCH (c:Community) WHERE c.summary IS NOT NULL "
            "RETURN c.id AS id, c.summary AS summary"
        )
        return {r["id"]: r["summary"] for r in await _drain(result)}


@_retry_on_disconnect
async def load_community_embeddings() -> dict[str, list[float]]:
    # Drain ALL embedded communities — a fetch cap here silently zeroes the
    # community-cosine graph signal (GRAPH_BETA) for anything past the cap.
    async with _get_session() as session:
        result = await session.run(
            "MATCH (c:Community) WHERE c.embedding IS NOT NULL "
            "RETURN c.id AS id, c.embedding AS emb"
        )
        return {r["id"]: list(r["emb"]) for r in await _drain(result)}


@_retry_on_disconnect
async def delete_source_communities(source_id: str):
    """Remove Community nodes owned by a source before a re-run.

    Community ids are namespaced `"<source_id>#<n>"`, so re-detecting a
    source can leave stale higher-numbered communities behind if the count
    shrinks. Clear them first to keep the set exact.
    """
    async with _get_session() as session:
        await session.run(
            "MATCH (c:Community) WHERE c.id STARTS WITH $prefix DETACH DELETE c",
            prefix=f"{source_id}#",
        )