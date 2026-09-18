"""PostgreSQL adapter for TokenStorePort (Personal Access Tokens).

Stores PATs in the personal_access_tokens table.
When the pool is unavailable, operations become no-ops — graceful degradation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from treeloom.domain.authorization import PersonalAccessToken, TokenStorePort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)


def _row_to_pat(row: asyncpg.Record) -> PersonalAccessToken:
    """Convert an asyncpg.Record to a PersonalAccessToken entity."""

    def _tz(dt):
        if dt is None:
            return None
        if isinstance(dt, datetime) and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    return PersonalAccessToken(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        token_hash=row["token_hash"],
        scopes=list(row["scopes"] or []),
        created_at=_tz(row["created_at"]) or datetime.now(timezone.utc),
        expires_at=_tz(row.get("expires_at")),
        last_used_at=_tz(row.get("last_used_at")),
        revoked=bool(row["revoked"]),
    )


class PostgreSQLPersonalAccessTokenStore(TokenStorePort):
    """Persist PATs via asyncpg to the personal_access_tokens table.

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
        user_id: str,
        name: str,
        token_hash: str,
        scopes: list[str],
        expires_at: Optional[datetime] = None,
    ) -> Optional[PersonalAccessToken]:
        """Create a new PAT and return it, or None on failure."""
        pool = await self._get_pool()
        if pool is None:
            logger.warning("token_store.create: no database pool available")
            return None

        token_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO personal_access_tokens
                           (id, user_id, name, token_hash, scopes,
                            created_at, expires_at, revoked)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, FALSE)""",
                    token_id,
                    user_id,
                    name,
                    token_hash,
                    scopes,
                    now,
                    expires_at,
                )
        except Exception:
            logger.exception("token_store.create: database insert failed")
            return None

        return PersonalAccessToken(
            id=token_id,
            user_id=user_id,
            name=name,
            token_hash=token_hash,
            scopes=list(scopes),
            created_at=now,
            expires_at=expires_at,
        )

    async def get_by_token_hash(
        self, token_hash: str
    ) -> Optional[PersonalAccessToken]:
        """Look up an active, non-expired, non-revoked PAT by its hash."""
        pool = await self._get_pool()
        if pool is None:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT * FROM personal_access_tokens
                       WHERE token_hash = $1
                         AND revoked = FALSE
                         AND (expires_at IS NULL OR expires_at > NOW())""",
                    token_hash,
                )
                return _row_to_pat(row) if row is not None else None
        except Exception:
            logger.exception("token_store.get_by_token_hash: database query failed")
            return None

    async def list_for_user(self, user_id: str) -> list[PersonalAccessToken]:
        """Return all PATs for a user (including revoked)."""
        pool = await self._get_pool()
        if pool is None:
            return []

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM personal_access_tokens
                       WHERE user_id = $1
                       ORDER BY created_at""",
                    user_id,
                )
                return [_row_to_pat(r) for r in rows]
        except Exception:
            logger.exception("token_store.list_for_user: database query failed")
            return []

    async def revoke(self, token_id: str, user_id: str) -> bool:
        """Revoke a PAT scoped to its owner. Returns True if found.

        Raises RevocationUnavailable rather than returning False on a store
        failure — see that exception for why the distinction matters.
        """
        from treeloom.adapters.authorization.api_key_store import (
            RevocationUnavailable,
        )

        pool = await self._get_pool()
        if pool is None:
            raise RevocationUnavailable("token store unavailable")

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """UPDATE personal_access_tokens
                       SET revoked = TRUE
                       WHERE id = $1 AND user_id = $2""",
                    token_id,
                    user_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception as exc:
            logger.exception("token_store.revoke: database update failed")
            raise RevocationUnavailable(f"could not revoke token {token_id!r}") from exc

    async def touch_last_used(self, token_id: str) -> None:
        """Update last_used_at to NOW() — best-effort, never raises."""
        pool = await self._get_pool()
        if pool is None:
            return

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE personal_access_tokens SET last_used_at = NOW() WHERE id = $1",
                    token_id,
                )
        except Exception:
            logger.exception("token_store.touch_last_used: database update failed")
