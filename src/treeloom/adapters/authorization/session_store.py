"""PostgreSQL adapter for SessionStorePort (browser sessions).

Stores DB-backed sessions in the sessions table.
When the pool is unavailable, operations become no-ops — graceful degradation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from treeloom.domain.authorization import Session, SessionStorePort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)


def _row_to_session(row: asyncpg.Record) -> Session:
    """Convert an asyncpg.Record to a Session entity."""

    def _tz(dt):
        if dt is None:
            return None
        if isinstance(dt, datetime) and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    return Session(
        id=row["id"],
        user_id=row["user_id"],
        token_hash=row["token_hash"],
        created_at=_tz(row["created_at"]) or datetime.now(timezone.utc),
        expires_at=_tz(row["expires_at"]) or datetime.now(timezone.utc),
        last_seen_at=_tz(row.get("last_seen_at")),
    )


class PostgreSQLSessionStore(SessionStorePort):
    """Persist browser sessions via asyncpg to the sessions table.

    Gracefully degrades when no pool is available: all methods return
    None/False/0 as appropriate.
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
        token_hash: str,
        expires_at: datetime,
    ) -> Optional[Session]:
        """Create a new session and return it, or None on failure."""
        pool = await self._get_pool()
        if pool is None:
            logger.warning("session_store.create: no database pool available")
            return None

        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO sessions (id, user_id, token_hash, created_at, expires_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    session_id,
                    user_id,
                    token_hash,
                    now,
                    expires_at,
                )
        except Exception:
            logger.exception("session_store.create: database insert failed")
            return None

        return Session(
            id=session_id,
            user_id=user_id,
            token_hash=token_hash,
            created_at=now,
            expires_at=expires_at,
        )

    async def get_by_token_hash(self, token_hash: str) -> Optional[Session]:
        """Look up a non-expired session by its hash."""
        pool = await self._get_pool()
        if pool is None:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT * FROM sessions
                       WHERE token_hash = $1
                         AND expires_at > NOW()""",
                    token_hash,
                )
                return _row_to_session(row) if row is not None else None
        except Exception:
            logger.exception("session_store.get_by_token_hash: database query failed")
            return None

    async def delete(self, token_hash: str) -> bool:
        """Delete a session (logout). Returns True if a row was removed."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM sessions WHERE token_hash = $1",
                    token_hash,
                )
                return "DELETE 1" in (result or "")
        except Exception:
            logger.exception("session_store.delete: database delete failed")
            return False

    async def touch(self, token_hash: str) -> None:
        """Update last_seen_at to NOW() — best-effort, never raises."""
        pool = await self._get_pool()
        if pool is None:
            return

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE sessions SET last_seen_at = NOW() WHERE token_hash = $1",
                    token_hash,
                )
        except Exception:
            logger.exception("session_store.touch: database update failed")

    async def delete_expired(self) -> int:
        """Delete all expired sessions and return the count removed."""
        pool = await self._get_pool()
        if pool is None:
            return 0

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM sessions WHERE expires_at <= NOW()"
                )
                # Result looks like "DELETE N"
                parts = (result or "").split()
                if len(parts) >= 2 and parts[0] == "DELETE":
                    return int(parts[1])
                return 0
        except Exception:
            logger.exception("session_store.delete_expired: database delete failed")
            return 0

    async def delete_for_user(self, user_id: str) -> int:
        """Delete every session for a user (e.g. after a password change).
        Returns the count removed."""
        pool = await self._get_pool()
        if pool is None:
            return 0

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM sessions WHERE user_id = $1",
                    user_id,
                )
                # Result looks like "DELETE N"
                parts = (result or "").split()
                if len(parts) >= 2 and parts[0] == "DELETE":
                    return int(parts[1])
                return 0
        except Exception:
            logger.exception("session_store.delete_for_user: database delete failed")
            return 0
