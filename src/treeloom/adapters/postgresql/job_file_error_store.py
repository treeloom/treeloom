"""Persist per-file indexing errors so operators can investigate
patterns and (eventually) retry just the failed files.

Schema lives in migration `011_job_file_errors.sql`. The `jobs.errors`
counter is the cheap aggregate; this table is the queryable detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobFileError:
    job_id: str
    file_path: str
    error_kind: str  # "timeout" | "exception"
    error_message: str
    elapsed_s: float | None
    occurred_at: datetime

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "file_path": self.file_path,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "elapsed_s": self.elapsed_s,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
        }


class JobFileErrorStore:
    """asyncpg-backed CRUD over `job_file_errors`.

    Best-effort: `record()` swallows DB errors with a log so error-
    accounting never breaks the indexer's own error-handling path.
    Read methods raise on misconfiguration (no pool).
    """

    async def record(
        self,
        job_id: str,
        file_path: str,
        error_kind: str,
        error_message: str = "",
        elapsed_s: float | None = None,
    ) -> None:
        if not job_id or not file_path:
            return
        pool = await get_pool()
        if pool is None:
            return
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO job_file_errors
                        (job_id, file_path, error_kind, error_message, elapsed_s)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (job_id, file_path)
                    DO UPDATE SET error_kind = EXCLUDED.error_kind,
                                  error_message = EXCLUDED.error_message,
                                  elapsed_s = EXCLUDED.elapsed_s,
                                  occurred_at = NOW()
                    """,
                    job_id, file_path, error_kind,
                    error_message or "", elapsed_s,
                )
        except Exception:
            logger.exception("JobFileErrorStore.record failed (best-effort, ignored)")

    async def list_for_job(
        self,
        job_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[JobFileError]:
        pool = await get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id, file_path, error_kind, error_message,
                       elapsed_s, occurred_at
                  FROM job_file_errors
                 WHERE job_id = $1
              ORDER BY occurred_at DESC
                 LIMIT $2 OFFSET $3
                """,
                job_id, int(limit), int(offset),
            )
        return [
            JobFileError(
                job_id=r["job_id"],
                file_path=r["file_path"],
                error_kind=r["error_kind"],
                error_message=r["error_message"] or "",
                elapsed_s=r["elapsed_s"],
                occurred_at=r["occurred_at"],
            )
            for r in rows
        ]

    async def count_for_job(self, job_id: str) -> int:
        pool = await get_pool()
        if pool is None:
            return 0
        async with pool.acquire() as conn:
            v = await conn.fetchval(
                "SELECT COUNT(*) FROM job_file_errors WHERE job_id = $1",
                job_id,
            )
        return int(v or 0)
