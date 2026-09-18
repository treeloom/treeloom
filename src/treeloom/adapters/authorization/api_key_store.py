"""PostgreSQL adapter for ApiKeyStorePort (service/admin API keys).

Stores API keys in the api_keys table.
When the pool is unavailable, operations become no-ops — graceful degradation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from treeloom.domain.authorization import ApiKey, ApiKeyStorePort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)

class RevocationUnavailable(RuntimeError):
    """A revocation could not be carried out because the store failed.

    Distinct from "no such credential". Collapsing the two is what made this
    dangerous: a database error returned False, the endpoint turned False into
    404 "not found", and an operator revoking a key they believe is
    compromised reads that as "already gone" and stops. The credential keeps
    working, and nothing in the response said otherwise (CWE-636).
    """



def _row_to_api_key(row: asyncpg.Record) -> ApiKey:
    """Convert an asyncpg.Record to an ApiKey entity."""

    def _tz(dt):
        if dt is None:
            return None
        if isinstance(dt, datetime) and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    return ApiKey(
        id=row["id"],
        name=row["name"],
        token_hash=row["token_hash"],
        scopes=list(row["scopes"] or []),
        all_access=bool(row["all_access"]),
        created_by=row.get("created_by"),
        created_at=_tz(row["created_at"]) or datetime.now(timezone.utc),
        expires_at=_tz(row.get("expires_at")),
        last_used_at=_tz(row.get("last_used_at")),
        revoked=bool(row["revoked"]),
    )


class PostgreSQLApiKeyStore(ApiKeyStorePort):
    """Persist API keys via asyncpg to the api_keys table.

    Gracefully degrades when no pool is available: all methods return
    None/False/[] as appropriate.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """Initialize with an optional pool override (for testing).

        When pool is None, get_pool() is called at method invocation time.
        """
        self._pool = pool

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        """Resolve pool: constructor override takes precedence."""
        if self._pool is not None:
            return self._pool
        return await get_pool()

    async def create(
        self,
        name: str,
        token_hash: str,
        scopes: list[str],
        all_access: bool = False,
        created_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[ApiKey]:
        """Create a new API key and return it, or None on failure."""
        pool = await self._get_pool()
        if pool is None:
            logger.warning("api_key_store.create: no database pool available")
            return None

        key_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO api_keys
                           (id, name, token_hash, scopes, all_access,
                            created_by, created_at, expires_at, revoked)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, FALSE)""",
                    key_id,
                    name,
                    token_hash,
                    scopes,
                    all_access,
                    created_by,
                    now,
                    expires_at,
                )
        except Exception:
            logger.exception("api_key_store.create: database insert failed")
            return None

        return ApiKey(
            id=key_id,
            name=name,
            token_hash=token_hash,
            scopes=list(scopes),
            all_access=all_access,
            created_by=created_by,
            created_at=now,
            expires_at=expires_at,
        )

    async def get_by_token_hash(self, token_hash: str) -> Optional[ApiKey]:
        """Look up an active, non-expired, non-revoked API key by its hash."""
        pool = await self._get_pool()
        if pool is None:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT * FROM api_keys
                       WHERE token_hash = $1
                         AND revoked = FALSE
                         AND (expires_at IS NULL OR expires_at > NOW())""",
                    token_hash,
                )
                return _row_to_api_key(row) if row is not None else None
        except Exception:
            logger.exception("api_key_store.get_by_token_hash: database query failed")
            return None

    async def list_keys(self) -> list[ApiKey]:
        """Return all API keys (admin view)."""
        pool = await self._get_pool()
        if pool is None:
            return []

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM api_keys ORDER BY created_at"
                )
                return [_row_to_api_key(r) for r in rows]
        except Exception:
            logger.exception("api_key_store.list_keys: database query failed")
            return []

    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key by id. Returns True if found.

        Raises RevocationUnavailable if the store cannot be reached or the
        update fails — never False, which the caller reports as 404.
        """
        pool = await self._get_pool()
        if pool is None:
            raise RevocationUnavailable("api_key store unavailable")

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE api_keys SET revoked = TRUE WHERE id = $1",
                    key_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception as exc:
            logger.exception("api_key_store.revoke: database update failed")
            raise RevocationUnavailable(f"could not revoke api key {key_id!r}") from exc

    async def touch_last_used(self, key_id: str) -> None:
        """Update last_used_at to NOW() — best-effort, never raises."""
        pool = await self._get_pool()
        if pool is None:
            return

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
                    key_id,
                )
        except Exception:
            logger.exception("api_key_store.touch_last_used: database update failed")
