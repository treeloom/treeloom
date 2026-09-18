"""asyncpg connection pool manager for treeloom.

Provides a connection pool that reads DATABASE_URL from the environment.
When no DATABASE_URL is set or the pool cannot connect, pool() returns None
and callers should fall back to in-memory/degraded mode.
"""

import logging
import os
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_pool(database_url: Optional[str] = None) -> Optional[asyncpg.Pool]:
    """Initialize the global asyncpg connection pool.

    Args:
        database_url: PostgreSQL connection string. If None, reads DATABASE_URL
                     from environment.

    Returns:
        An asyncpg.Pool on success, or None if the database is unreachable
        or DATABASE_URL is not configured.
    """
    global _pool

    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        logger.warning("DATABASE_URL not configured — PostgreSQL features disabled")
        return None

    try:
        _pool = await asyncpg.create_pool(
            url,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        logger.info("PostgreSQL connection pool initialized")
        return _pool
    except Exception as exc:
        logger.warning("Failed to connect to PostgreSQL: %s — falling back to in-memory mode", exc)
        _pool = None
        return None


async def get_pool() -> Optional[asyncpg.Pool]:
    """Return the global connection pool, or None if uninitialized/unavailable."""
    return _pool


async def close_pool() -> None:
    """Close the connection pool and release resources."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")
