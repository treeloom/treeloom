"""PostgreSQL-backed JobGroupStore.

A job_group is a single submission of 1..N index Tasks -- a "Job" in the
operator UI v2. Schema lives in `migrations/020_job_groups.sql`; each `jobs`
row (a "Task") links to its parent via `jobs.group_id`. The physical `jobs`
table is deliberately NOT renamed (see the migration header).

Like JobStore, this fails loud (RuntimeError) when the Postgres pool is
unavailable rather than silently degrading.
"""

from __future__ import annotations

import uuid

from treeloom.adapters.postgresql.connection import get_pool
from treeloom.domain.jobs import JobGroup


class JobGroupStore:
    """asyncpg-backed job_groups table."""

    async def init(self) -> None:
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(
                "JobGroupStore requires DATABASE_URL to be set and reachable. "
                "Set DATABASE_URL=postgresql://... in your environment."
            )

    @staticmethod
    def _row_to_group(row) -> JobGroup:
        return JobGroup(
            id=row["id"],
            label=row["label"],
            kind=row["kind"] or "repo",
            created_at=row["created_at"],
            created_by=row["created_by"],
            task_count=row["task_count"],
        )

    async def create(
        self,
        label: str,
        kind: str,
        created_by: str | None = None,
        task_count: int = 1,
        created_at: float | None = None,
    ) -> str:
        """Insert a new job_group and return its id.

        `created_at` defaults to the DB clock (epoch seconds) when not given.
        """
        group = JobGroup(
            id="grp_" + uuid.uuid4().hex,
            label=label or "(unnamed)",
            kind=kind or "repo",
            created_by=created_by,
            task_count=task_count,
        )
        if created_at is not None:
            group.created_at = created_at
        pool = await get_pool()
        if pool is None:
            raise RuntimeError("JobGroupStore.create: Postgres pool unavailable")
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO job_groups (id, label, kind, created_at, created_by, task_count)
                VALUES ($1, $2, $3, COALESCE($4, extract(epoch from now())), $5, $6)
                """,
                group.id, group.label, group.kind,
                created_at, group.created_by, group.task_count,
            )
        return group.id

    async def get(self, group_id: str) -> JobGroup | None:
        pool = await get_pool()
        if pool is None:
            raise RuntimeError("JobGroupStore.get: Postgres pool unavailable")
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM job_groups WHERE id = $1", group_id
            )
            return self._row_to_group(row) if row else None

    async def list_all(self) -> list[JobGroup]:
        """All groups, newest first (by created_at)."""
        pool = await get_pool()
        if pool is None:
            raise RuntimeError("JobGroupStore.list_all: Postgres pool unavailable")
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM job_groups ORDER BY created_at DESC"
            )
            return [self._row_to_group(r) for r in rows]

    async def increment_task_count(self, group_id: str, by: int = 1) -> int:
        """Atomically add `by` to a group's task_count; return the new value.

        Used when more Tasks are attached to an existing group (e.g. fleet
        onboarding, follow-up). Returns 0 if the group is missing.
        """
        pool = await get_pool()
        if pool is None:
            raise RuntimeError("JobGroupStore.increment_task_count: Postgres pool unavailable")
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE job_groups SET task_count = task_count + $2 "
                "WHERE id = $1 RETURNING task_count",
                group_id, by,
            )
            return row["task_count"] if row else 0
