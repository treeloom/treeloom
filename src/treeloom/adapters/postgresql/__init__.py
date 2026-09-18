"""PostgreSQL adapter package for treeloom.

Provides connection pooling and migration support. All database interaction
goes through asyncpg pools managed by connection.py.
"""

import logging
import os
from pathlib import Path

from treeloom.adapters.postgresql.connection import init_pool, get_pool, close_pool  # noqa: F401

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATIONS_TABLE = "treeloom_migrations"


async def run_migrations() -> list[str]:
    """Run pending SQL migrations against the PostgreSQL database.

    Creates a tracking table (treeloom_migrations) to ensure each migration
    runs exactly once. Migration files are loaded in name order from the
    migrations/ directory.

    Returns:
        List of migration filenames that were applied. Empty list if no
        migrations were pending or the pool is unavailable.
    """
    pool = await get_pool()
    if pool is None:
        return []

    applied: list[str] = []

    async with pool.acquire() as conn:
        try:
            # Ensure tracking table exists
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_MIGRATIONS_TABLE} (name TEXT PRIMARY KEY)"
            )

            # Find pending migrations
            migration_files = sorted(
                f for f in os.listdir(_MIGRATIONS_DIR) if f.endswith(".sql")
            )
            for fname in migration_files:
                record = await conn.fetchrow(
                    f"SELECT name FROM {_MIGRATIONS_TABLE} WHERE name = $1", fname
                )
                if record is not None:
                    continue  # Already applied

                sql_path = _MIGRATIONS_DIR / fname
                sql = sql_path.read_text()
                await conn.execute(sql)
                await conn.execute(
                    f"INSERT INTO {_MIGRATIONS_TABLE} (name) VALUES ($1)", fname
                )
                logger.info("Applied migration: %s", fname)
                applied.append(fname)
        except Exception:
            logger.exception("Migration execution failed")

    return applied
