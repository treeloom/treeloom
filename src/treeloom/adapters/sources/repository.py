"""PostgreSQL adapter for SourceRepositoryPort.

Stores SourceRecord metadata in the source_records table.
When the pool is unavailable (no DATABASE_URL), save/delete are
no-ops and queries return empty results — graceful degradation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from treeloom.domain.sources import SourceRecord, SourceRepositoryPort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)

# Ceiling on a single source listing. Well above any realistic install — the
# fleet docs target hundreds — so it truncates nothing in practice.
SOURCE_LIST_MAX = int(os.environ.get("TREELOOM_SOURCE_LIST_MAX", "10000"))


def _row_to_record(row: asyncpg.Record) -> SourceRecord:
    """Convert an asyncpg.Record to a SourceRecord."""
    # asyncpg's Record.keys() is a single-use iterator; materialize it so the
    # repeated `"col" in keys` membership checks below don't consume it. Left as
    # the bare iterator, each `in` advanced past earlier columns, so a column
    # (e.g. commit_sha) checked after one positioned later in table order read
    # as "missing" and silently fell back to its default — making every source
    # report an empty commit_sha regardless of what was stored.
    keys = set(row.keys())
    graph_indexed = bool(row["graph_indexed"]) if "graph_indexed" in keys else True
    graph_indexed_at = row["graph_indexed_at"] if "graph_indexed_at" in keys else None
    if isinstance(graph_indexed_at, datetime):
        graph_indexed_at = graph_indexed_at.replace(tzinfo=timezone.utc)
    return SourceRecord(
        id=row["id"],
        path=row["path"] or "",
        url=row["url"] or "",
        branch=row["branch"] or "",
        indexed_at=row["indexed_at"].replace(tzinfo=timezone.utc)
        if isinstance(row["indexed_at"], datetime)
        else row["indexed_at"],
        file_count=int(row["file_count"] or 0),
        chunk_count=int(row["chunk_count"] or 0),
        commit_sha=(row["commit_sha"] if "commit_sha" in keys else "") or "",
        created_by=row.get("created_by"),
        kind=(row["kind"] if "kind" in keys else "repo") or "repo",
        graph_indexed=graph_indexed,
        graph_indexed_at=graph_indexed_at,
    )


class PostgreSourceRepository(SourceRepositoryPort):
    """Persist source metadata via asyncpg.

    Gracefully degrades when no pool is available:
    save/delete become no-ops, get_by_id returns None,
    list_all returns [].
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """Initialize with an optional pool override (for testing).

        When pool is None, get_pool() is called at method invocation time
        to pick up the global pool if one was initialized.
        """
        self._pool = pool

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        """Resolve pool: constructor override takes precedence."""
        if self._pool is not None:
            return self._pool
        return await get_pool()

    # ------------------------------------------------------------------
    # SourceRepositoryPort implementation
    # ------------------------------------------------------------------

    async def save(self, record: SourceRecord) -> None:
        """Insert or update a source record (upsert by id)."""
        pool = await self._get_pool()
        if pool is None:
            return

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO source_records
                           (id, path, url, branch, indexed_at,
                            file_count, chunk_count, commit_sha, created_by, kind)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                       ON CONFLICT (id) DO UPDATE SET
                           path       = EXCLUDED.path,
                           url        = EXCLUDED.url,
                           branch     = EXCLUDED.branch,
                           indexed_at = EXCLUDED.indexed_at,
                           file_count = EXCLUDED.file_count,
                           chunk_count = EXCLUDED.chunk_count,
                           commit_sha = EXCLUDED.commit_sha,
                           created_by = EXCLUDED.created_by,
                           kind       = EXCLUDED.kind""",
                    record.id,
                    record.path,
                    record.url,
                    record.branch,
                    record.indexed_at,
                    record.file_count,
                    record.chunk_count,
                    record.commit_sha,
                    record.created_by,
                    record.kind,
                )
        except Exception:
            logger.exception("PostgreSourceRepository.save: database insert failed")

    async def get_by_id(self, source_id: str) -> Optional[SourceRecord]:
        """Retrieve a single source by its deterministic id."""
        pool = await self._get_pool()
        if pool is None:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM source_records WHERE id = $1", source_id
                )
                if row is None:
                    return None
                return _row_to_record(row)
        except Exception:
            logger.exception("PostgreSourceRepository.get_by_id: database query failed")
            return None

    async def list_all(self, limit: int | None = None) -> list[SourceRecord]:
        """Return persisted sources, most recently indexed first.

        Bounded. The unlimited form materialised every row into memory on a
        request path (GET /sources, the fleet rollup), so the cost of listing
        grew without limit as an install onboarded repositories — the fleet
        documentation targets hundreds, and nothing here capped it (CWE-400).
        The default ceiling is far above any real install, so it changes no
        behaviour today; it exists so that "unusually many sources" degrades
        into a truncated list rather than into memory pressure.
        """
        pool = await self._get_pool()
        if pool is None:
            return []

        cap = SOURCE_LIST_MAX if limit is None else max(1, int(limit))
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM source_records ORDER BY indexed_at DESC "
                    "LIMIT $1",
                    cap,
                )
                if len(rows) == cap:
                    logger.warning(
                        "source_records listing hit the %d-row cap; results are "
                        "truncated (raise TREELOOM_SOURCE_LIST_MAX if intended)",
                        cap,
                    )
                return [_row_to_record(r) for r in rows]
        except Exception:
            logger.exception("PostgreSourceRepository.list_all: database query failed")
            return []

    async def delete(self, source_id: str) -> bool:
        """Remove a source record. Returns True if a record was deleted."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM source_records WHERE id = $1", source_id
                )
                # asyncpg execute returns a string like "DELETE 1" or "DELETE 0"
                return "DELETE 1" in (result or "")
        except Exception:
            logger.exception("PostgreSourceRepository.delete: database delete failed")
            return False

    async def set_graph_indexed(
        self,
        source_id: str,
        indexed: bool,
        when: Optional[datetime] = None,
    ) -> bool:
        """Mark a source's graph-indexing state. Used by the two-pass flow:
        chunks-only `/index-repo` sets False; `/index-graph` sets True.
        Returns True if a row was updated."""
        pool = await self._get_pool()
        if pool is None:
            return False
        ts = when or datetime.now(timezone.utc)
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """UPDATE source_records
                          SET graph_indexed = $2,
                              graph_indexed_at = $3
                        WHERE id = $1""",
                    source_id, indexed, ts if indexed else None,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("set_graph_indexed: database update failed")
            return False
