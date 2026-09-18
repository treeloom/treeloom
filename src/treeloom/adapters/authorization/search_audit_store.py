"""PostgreSQL adapter for the search audit trail.

Best-effort writes: a DB failure while auditing is swallowed so the search
path never breaks (same discipline as job_file_errors). Reads are for the
admin audit view.
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import Optional

import asyncpg

from treeloom.domain.search_audit import SearchAuditPort, SearchAuditRecord
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)


class PostgreSQLSearchAuditStore(SearchAuditPort):
    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self._pool = pool

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        if self._pool is not None:
            return self._pool
        return await get_pool()

    async def record(self, entry: SearchAuditRecord) -> None:
        try:
            pool = await self._get_pool()
            if pool is None:
                return
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO search_audit
                           (occurred_at, user_id, principal_groups, action,
                            scope, source_id, decision)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    entry.occurred_at,
                    entry.user_id,
                    list(entry.principal_groups),
                    entry.action,
                    entry.scope,
                    entry.source_id,
                    entry.decision,
                )
        except Exception:
            # Best-effort: auditing must never break the search path.
            logger.warning("search_audit write failed", exc_info=True)

    async def recent(
        self,
        user_id: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        pool = await self._get_pool()
        if pool is None:
            return []
        conds: list[str] = []
        params: list = []
        if user_id is not None:
            params.append(user_id)
            conds.append(f"user_id = ${len(params)}")
        if source_id is not None:
            params.append(source_id)
            conds.append(f"source_id = ${len(params)}")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        params.append(limit)
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT occurred_at, user_id, principal_groups, action,
                               scope, source_id, decision
                        FROM search_audit {where}
                        ORDER BY occurred_at DESC LIMIT ${len(params)}""",
                    *params,
                )
                out = []
                for r in rows:
                    d = dict(r)
                    oa = d.get("occurred_at")
                    if oa is not None:
                        d["occurred_at"] = oa.replace(tzinfo=timezone.utc).isoformat() \
                            if oa.tzinfo is None else oa.isoformat()
                    d["principal_groups"] = list(d.get("principal_groups") or [])
                    out.append(d)
                return out
        except Exception:
            logger.warning("search_audit read failed", exc_info=True)
            return []
