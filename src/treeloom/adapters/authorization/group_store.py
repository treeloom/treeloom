"""PostgreSQL adapter for GroupStorePort.

Stores authorization groups and membership. Mirrors PostgreSQLUserStore:
constructor pool override for tests, graceful degradation to no-ops when the
pool is unavailable.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from treeloom.domain.authorization import Group, GroupStorePort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)


def _row_to_group(row: asyncpg.Record) -> Group:
    return Group(
        id=row["id"],
        name=row["name"],
        all_access=bool(row["all_access"]),
        created_at=row["created_at"].replace(tzinfo=timezone.utc)
        if isinstance(row["created_at"], datetime)
        else row["created_at"],
    )


class PostgreSQLGroupStore(GroupStorePort):
    """Persist groups + membership via asyncpg."""

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self._pool = pool

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        if self._pool is not None:
            return self._pool
        return await get_pool()

    async def create_group(
        self, name: str, all_access: bool = False
    ) -> Optional[Group]:
        group = Group(
            id=uuid.uuid4().hex,
            name=name,
            all_access=all_access,
            created_at=datetime.now(timezone.utc),
        )
        pool = await self._get_pool()
        if pool is None:
            logger.warning("create_group: no database pool available")
            return None
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO groups (id, name, all_access, created_at)
                       VALUES ($1, $2, $3, $4)""",
                    group.id,
                    group.name,
                    group.all_access,
                    group.created_at,
                )
        except Exception:
            logger.exception("create_group: database insert failed")
            return None
        return group

    async def get_by_id(self, group_id: str) -> Optional[Group]:
        pool = await self._get_pool()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM groups WHERE id = $1", group_id
                )
                return _row_to_group(row) if row is not None else None
        except Exception:
            logger.exception("get_by_id: database query failed")
            return None

    async def get_by_name(self, name: str) -> Optional[Group]:
        pool = await self._get_pool()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM groups WHERE name = $1", name
                )
                return _row_to_group(row) if row is not None else None
        except Exception:
            logger.exception("get_by_name: database query failed")
            return None

    async def list_groups(self) -> list[Group]:
        pool = await self._get_pool()
        if pool is None:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM groups ORDER BY name")
                return [_row_to_group(r) for r in rows]
        except Exception:
            logger.exception("list_groups: database query failed")
            return []

    async def delete_group(self, group_id: str) -> bool:
        # group_members rows cascade via the FK in migration 012.
        pool = await self._get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM groups WHERE id = $1", group_id
                )
                return "DELETE 1" in (result or "")
        except Exception:
            logger.exception("delete_group: database delete failed")
            return False

    async def set_all_access(self, group_id: str, all_access: bool) -> bool:
        pool = await self._get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE groups SET all_access = $1 WHERE id = $2",
                    all_access,
                    group_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("set_all_access: database update failed")
            return False

    async def add_member(self, group_id: str, user_id: str) -> bool:
        pool = await self._get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO group_members (group_id, user_id)
                       VALUES ($1, $2)
                       ON CONFLICT (group_id, user_id) DO NOTHING""",
                    group_id,
                    user_id,
                )
                return True
        except Exception:
            logger.exception("add_member: database insert failed")
            return False

    async def remove_member(self, group_id: str, user_id: str) -> bool:
        pool = await self._get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM group_members WHERE group_id = $1 AND user_id = $2",
                    group_id,
                    user_id,
                )
                return "DELETE 1" in (result or "")
        except Exception:
            logger.exception("remove_member: database delete failed")
            return False

    async def list_members(self, group_id: str) -> list[str]:
        pool = await self._get_pool()
        if pool is None:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT user_id FROM group_members WHERE group_id = $1",
                    group_id,
                )
                return [r["user_id"] for r in rows]
        except Exception:
            logger.exception("list_members: database query failed")
            return []
