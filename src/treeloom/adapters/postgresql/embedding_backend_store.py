"""Postgres-backed registry for TEI embedding backends.

Replaces the previous `.env`-only config (EMBEDDING_URLS,
EMBEDDING_FALLBACK_URLS) with a live, mutable list shared across all
indexer worker processes. Mutating methods emit `NOTIFY
embedding_backends_changed` in the same transaction as the write.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)

NOTIFY_CHANNEL = "embedding_backends_changed"

_POOL_REQUIRED = (
    "EmbeddingBackendStore requires DATABASE_URL to be set and reachable. "
    "Set DATABASE_URL=postgresql://... in your environment."
)


@dataclass
class BackendRow:
    id: UUID
    url: str
    klass: str  # 'gpu' | 'cpu'
    enabled: bool
    created_at: datetime
    created_by: UUID | None

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "url": self.url,
            "klass": self.klass,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "created_by": str(self.created_by) if self.created_by else None,
        }


def _row_to_backend(row) -> BackendRow:
    return BackendRow(
        id=row["id"],
        url=row["url"],
        klass=row["klass"],
        enabled=row["enabled"],
        created_at=row["created_at"],
        created_by=row["created_by"],
    )


class EmbeddingBackendStore:
    """Async CRUD over the `embedding_backends` table."""

    async def list_all(self) -> list[BackendRow]:
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM embedding_backends ORDER BY created_at"
            )
        return [_row_to_backend(r) for r in rows]

    async def list_enabled(self) -> list[BackendRow]:
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM embedding_backends WHERE enabled = TRUE ORDER BY created_at"
            )
        return [_row_to_backend(r) for r in rows]

    async def add(
        self,
        url: str,
        klass: str,
        created_by: UUID | None = None,
    ) -> BackendRow:
        if klass not in ("gpu", "cpu"):
            raise ValueError(f"klass must be 'gpu' or 'cpu', got {klass!r}")
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO embedding_backends (url, klass, created_by)
                    VALUES ($1, $2, $3)
                    RETURNING *
                    """,
                    url, klass, created_by,
                )
                # NOTIFY in the same transaction as the write — rollback
                # of the txn discards the notification.
                await conn.execute(f"NOTIFY {NOTIFY_CHANNEL}")
        return _row_to_backend(row)

    async def delete(self, backend_id: UUID) -> bool:
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM embedding_backends WHERE id = $1",
                    backend_id,
                )
                deleted = result.endswith(" 1")
                if deleted:
                    await conn.execute(f"NOTIFY {NOTIFY_CHANNEL}")
        return deleted

    async def seed_from_env_if_empty(self) -> int:
        """One-time migration helper. Inserts URLs from env into the registry
        ONLY when the table is empty. Returns the number of rows inserted."""
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)

        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM embedding_backends")
            if count > 0:
                return 0

            gpu_urls = _split_env("EMBEDDING_URLS") or _split_env("EMBEDDING_URL")
            cpu_urls = _split_env("EMBEDDING_FALLBACK_URLS") or _split_env("EMBEDDING_FALLBACK_URL")

            if not gpu_urls and not cpu_urls:
                logger.info("No EMBEDDING_URLS or EMBEDDING_FALLBACK_URLS in env — nothing to seed")
                return 0

            inserted = 0
            async with conn.transaction():
                for url in gpu_urls:
                    await conn.execute(
                        "INSERT INTO embedding_backends (url, klass) VALUES ($1, 'gpu') "
                        "ON CONFLICT (url) DO NOTHING",
                        url,
                    )
                    inserted += 1
                for url in cpu_urls:
                    await conn.execute(
                        "INSERT INTO embedding_backends (url, klass) VALUES ($1, 'cpu') "
                        "ON CONFLICT (url) DO NOTHING",
                        url,
                    )
                    inserted += 1
                # No NOTIFY needed: nothing is LISTENing yet at seed time
                # (startup hook runs before EmbeddingProxy.start_listener).
            logger.info("Seeded embedding_backends with %d row(s) from env", inserted)
            return inserted


def _split_env(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [u.strip() for u in raw.split(",") if u.strip()]
