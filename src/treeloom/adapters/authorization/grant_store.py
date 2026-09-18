"""PostgreSQL adapter for GrantStorePort.

Stores per-source access grants (principal_type, principal_id, source_id,
effect). Mirrors the other authorization stores: pool override for tests,
graceful degradation when the pool is unavailable.

The hot path is the read side — effects_for_source / denied_sources — called on
every authorized search. Both take the caller's principals (user + groups) and
resolve against the indexed (principal_type, principal_id) / (source_id) columns.

Those two FAIL LOUD; the rest of this module degrades gracefully. The
difference is that their result is an authorization input, and both previously
swallowed database errors into an empty result — which does not read as "deny",
it reads as "this principal has no deny grants". For a caller with all_access
that is the difference between refusal and access: the deny grant is the only
thing standing between them and the source, so losing it during a database
blip grants exactly the access it was created to prevent (CWE-636). An empty
set is a real answer about a real caller, so it cannot double as an error
signal — the callers turn a raise into 503.
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from treeloom.domain.authorization import GrantStorePort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)


class GrantLookupUnavailable(RuntimeError):
    """The grant store could not answer an authorization question.

    Distinct from "the answer is no grants": callers must refuse the request
    rather than proceed on an empty result. See the module docstring.
    """


def _principal_arrays(
    principals: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Split principals into parallel (types, ids) arrays for an UNNEST join."""
    types = [p[0] for p in principals]
    ids = [p[1] for p in principals]
    return types, ids


class PostgreSQLGrantStore(GrantStorePort):
    """Persist per-source grants via asyncpg."""

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self._pool = pool

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        if self._pool is not None:
            return self._pool
        return await get_pool()

    async def grant(
        self,
        principal_type: str,
        principal_id: str,
        source_id: str,
        effect: str,
    ) -> bool:
        if principal_type not in ("user", "group") or effect not in (
            "allow",
            "deny",
        ):
            raise ValueError(
                f"invalid grant: principal_type={principal_type!r} effect={effect!r}"
            )
        pool = await self._get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO source_grants
                           (principal_type, principal_id, source_id, effect)
                       VALUES ($1, $2, $3, $4)
                       ON CONFLICT (principal_type, principal_id, source_id)
                       DO UPDATE SET effect = EXCLUDED.effect""",
                    principal_type,
                    principal_id,
                    source_id,
                    effect,
                )
                return True
        except Exception:
            logger.exception("grant: database upsert failed")
            return False

    async def revoke(
        self, principal_type: str, principal_id: str, source_id: str
    ) -> bool:
        pool = await self._get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """DELETE FROM source_grants
                       WHERE principal_type = $1 AND principal_id = $2
                         AND source_id = $3""",
                    principal_type,
                    principal_id,
                    source_id,
                )
                return "DELETE 1" in (result or "")
        except Exception:
            logger.exception("revoke: database delete failed")
            return False

    async def list_for_source(self, source_id: str) -> list[dict]:
        pool = await self._get_pool()
        if pool is None:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT principal_type, principal_id, source_id, effect, created_at
                       FROM source_grants WHERE source_id = $1
                       ORDER BY principal_type, principal_id""",
                    source_id,
                )
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("list_for_source: database query failed")
            return []

    async def effects_for_source(
        self, principals: list[tuple[str, str]], source_id: str
    ) -> set[str]:
        if not principals:
            return set()
        pool = await self._get_pool()
        if pool is None:
            # Enforcement is only reached with AUTH_ENABLED=true, where a
            # missing pool is a misconfiguration, not a mode.
            raise GrantLookupUnavailable(
                "grant store unavailable: cannot authorize without it"
            )
        types, ids = _principal_arrays(principals)
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT DISTINCT g.effect
                       FROM source_grants g
                       JOIN UNNEST($1::TEXT[], $2::TEXT[]) AS p(ptype, pid)
                         ON g.principal_type = p.ptype AND g.principal_id = p.pid
                       WHERE g.source_id = $3""",
                    types,
                    ids,
                    source_id,
                )
                return {r["effect"] for r in rows}
        except Exception as exc:
            logger.exception("effects_for_source: database query failed")
            raise GrantLookupUnavailable(
                f"grant lookup failed for source {source_id!r}"
            ) from exc

    async def denied_sources(
        self, principals: list[tuple[str, str]]
    ) -> list[str]:
        if not principals:
            return []
        pool = await self._get_pool()
        if pool is None:
            raise GrantLookupUnavailable(
                "grant store unavailable: cannot authorize without it"
            )
        types, ids = _principal_arrays(principals)
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT DISTINCT g.source_id
                       FROM source_grants g
                       JOIN UNNEST($1::TEXT[], $2::TEXT[]) AS p(ptype, pid)
                         ON g.principal_type = p.ptype AND g.principal_id = p.pid
                       WHERE g.effect = 'deny'""",
                    types,
                    ids,
                )
                return [r["source_id"] for r in rows]
        except Exception as exc:
            logger.exception("denied_sources: database query failed")
            raise GrantLookupUnavailable("denied-source lookup failed") from exc
