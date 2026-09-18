"""PostgreSQL adapter for UserStorePort.

Stores user accounts in the users table with HMAC-SHA256-hashed API keys.
When the pool is unavailable, operations become no-ops — graceful degradation.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import bcrypt

from treeloom.domain.authorization import User, Role, UserStorePort
from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)

# ── HMAC secret for deterministic API key hashing (FR-007) ─────────
_HMAC_SECRET = os.environ.get("TREELOOM_HMAC_SECRET", "")
if not _HMAC_SECRET:
    _HMAC_SECRET = secrets.token_hex(32)
    logger.warning(
        "TREELOOM_HMAC_SECRET not set — generated random secret. "
        "Set TREELOOM_HMAC_SECRET for persistent API keys across restarts."
    )


def hash_api_key(raw_key: str, secret: str = _HMAC_SECRET) -> str:
    """Hash an API key deterministically using HMAC-SHA256.

    Allows exact-match lookups against stored hashes (unlike bcrypt which
    generates a random salt on every call).

    Args:
        raw_key: The raw API key string.
        secret: HMAC secret key (defaults to _HMAC_SECRET).

    Returns:
        Hex-encoded SHA256 HMAC of the raw key.
    """
    return hmac.new(
        secret.encode(), raw_key.encode(), hashlib.sha256
    ).hexdigest()


# bcrypt hashes at most 72 bytes of input and DISCARDS the rest without
# complaint. That is the algorithm, not a bug here — but silently accepting a
# longer password and then authenticating on a 72-byte prefix is: a
# passphrase-manager secret that shares its first 72 bytes with another is the
# same credential as far as this system is concerned, and nothing told the
# user (CWE-916). Note the limit is in BYTES, so a non-ASCII passphrase hits
# it sooner than its character count suggests.
BCRYPT_MAX_PASSWORD_BYTES = 72


class PasswordTooLongError(ValueError):
    """Raised instead of silently hashing a truncated password."""


def hash_password(raw: str) -> str:
    """Hash a plaintext password using bcrypt (per-row salt).

    Use for local-login passwords. Do NOT use for token lookup — tokens
    use the deterministic HMAC hash_api_key so they can be looked up
    by exact-match without iterating every row.

    Raises PasswordTooLongError above bcrypt's 72-byte input limit rather
    than hashing a prefix and reporting success.
    """
    encoded = raw.encode()
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"password is {len(encoded)} bytes; bcrypt hashes at most "
            f"{BCRYPT_MAX_PASSWORD_BYTES} and would silently ignore the rest"
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode()


# A well-formed bcrypt hash: $2<variant>$<cost>$<53 chars of salt+digest>,
# e.g. $2b$12$C6UzMDM.H6dfI/f/IKcEeO... (60 chars total). Validating this
# shape BEFORE calling bcrypt.checkpw is deliberate, not cosmetic: newer
# bcrypt wheels (4.0.1) panic in their Rust layer on malformed/truncated
# hashes (`pyo3_runtime.PanicException`), which derives from BaseException
# and is NOT caught by `except (ValueError, TypeError)` — or even a bare
# `except Exception`. Rejecting non-bcrypt-shaped input up front means we
# never hand bcrypt something it can panic on.
_BCRYPT_HASH_RE = re.compile(r"^\$2[abxy]\$\d{2}\$[./A-Za-z0-9]{53}$")


def verify_password(raw: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.

    Returns False on any error (bad hash format, type mismatch, etc.)
    without raising — callers should treat False as 'wrong password'.
    A corrupt/truncated stored hash must never crash a login request.
    """
    if not _BCRYPT_HASH_RE.match(hashed):
        return False
    encoded = raw.encode()
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        # Refuse rather than compare a prefix. Accepting it here would let an
        # over-long password authenticate against a hash made from its first
        # 72 bytes — which is the truncation hazard, just on the read side.
        # hash_password rejects these, so no stored hash can have come from
        # one; anything reaching here is an attempt to exploit the prefix.
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode())
    except (ValueError, TypeError):
        return False


# Selects a user row plus the authorization read-model: the array of group ids
# they belong to, and effective all_access (own flag OR any group's flag).
# Used by the identity lookups so the middleware gets a fully-hydrated principal
# in one round trip.
_HYDRATED_SELECT = """
SELECT u.*,
       COALESCE(
           ARRAY_AGG(gm.group_id) FILTER (WHERE gm.group_id IS NOT NULL),
           ARRAY[]::TEXT[]
       ) AS group_ids,
       (u.all_access OR BOOL_OR(COALESCE(g.all_access, FALSE))) AS effective_all_access
FROM users u
LEFT JOIN group_members gm ON gm.user_id = u.id
LEFT JOIN groups g ON g.id = gm.group_id
WHERE {where}
GROUP BY u.id
"""


def _row_to_user(row: asyncpg.Record) -> User:
    """Convert an asyncpg.Record to a User entity.

    Handles both hydrated rows (with group_ids / effective_all_access from the
    LEFT JOIN) and plain ``SELECT *`` rows (list_users, etc.), defaulting the
    authorization read-model to empty when those columns are absent.
    """
    keys = set(row.keys())
    if "effective_all_access" in keys:
        all_access = bool(row["effective_all_access"])
    elif "all_access" in keys:
        all_access = bool(row["all_access"])
    else:
        all_access = False
    group_ids = (
        list(row["group_ids"])
        if "group_ids" in keys and row["group_ids"] is not None
        else []
    )
    return User(
        id=row["id"],
        username=row["username"],
        email=row.get("email") or "",
        api_key_hash=row["api_key_hash"],
        role=Role(row["role"]),
        active=bool(row["active"]),
        created_at=row["created_at"].replace(tzinfo=timezone.utc)
        if isinstance(row["created_at"], datetime)
        else row["created_at"],
        all_access=all_access,
        group_ids=group_ids,
        password_hash=row.get("password_hash") or "",
    )


class PostgreSQLUserStore(UserStorePort):
    """Persist user accounts via asyncpg to the users table.

    Gracefully degrades when no pool is available: all methods return
    None/False/0/[] as appropriate.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """Initialize with an optional pool override (for testing).

        When pool is None, get_pool() is called at method invocation time
        to pick up the global pool if one was initialized.
        """
        self._pool = pool

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        """Resolve pool: constructor override takes precedence."""
        if self._pool is not None:
            return self._pool
        return await get_pool()

    # ------------------------------------------------------------------
    # UserStorePort implementation
    # ------------------------------------------------------------------

    async def create_user(
        self,
        username: str,
        email: str = "",
        role: Role = Role.USER,
        api_key_hash: str = "",
    ) -> Optional[User]:
        """Create a new user and return the User entity.

        Returns None if the database is unavailable (pool is None or
        the insert fails).
        """
        import uuid

        user_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        user = User(
            id=user_id,
            username=username,
            email=email,
            api_key_hash=api_key_hash,
            role=role,
            active=True,
            created_at=now,
        )

        pool = await self._get_pool()
        if pool is None:
            logger.warning("create_user: no database pool available")
            return None

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO users
                           (id, username, email, api_key_hash, role, active, created_at)
                       VALUES ($1, $2, $3, $4, $5, TRUE, $6)""",
                    user.id,
                    user.username,
                    user.email,
                    user.api_key_hash,
                    user.role.value,
                    user.created_at,
                )
        except Exception:
            logger.exception("create_user: database insert failed")
            return None

        return user

    async def get_by_api_key_hash(self, key_hash: str) -> Optional[User]:
        """Look up a user by their hashed API key."""
        pool = await self._get_pool()
        if pool is None:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    _HYDRATED_SELECT.format(
                        where="u.api_key_hash = $1 AND u.active = TRUE"
                    ),
                    key_hash,
                )
                if row is None:
                    return None
                return _row_to_user(row)
        except Exception:
            logger.exception("get_by_api_key_hash: database query failed")
            return None

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Look up a user by their id."""
        pool = await self._get_pool()
        if pool is None:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    _HYDRATED_SELECT.format(where="u.id = $1"), user_id
                )
                if row is None:
                    return None
                return _row_to_user(row)
        except Exception:
            logger.exception("get_by_id: database query failed")
            return None

    async def get_by_username(self, username: str) -> Optional[User]:
        """Look up a user by username (hydrated)."""
        pool = await self._get_pool()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    _HYDRATED_SELECT.format(where="u.username = $1"), username
                )
                return _row_to_user(row) if row is not None else None
        except Exception:
            logger.exception("get_by_username: database query failed")
            return None

    async def list_users(self) -> list[User]:
        """Return all active users."""
        pool = await self._get_pool()
        if pool is None:
            return []

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM users WHERE active = TRUE ORDER BY created_at"
                )
                return [_row_to_user(r) for r in rows]
        except Exception:
            logger.exception("list_users: database query failed")
            return []

    async def update_role(self, user_id: str, role: Role) -> bool:
        """Update a user's role. Returns True if the user was found."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE users SET role = $1 WHERE id = $2",
                    role.value,
                    user_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("update_role: database update failed")
            return False

    async def update_api_key_hash(self, user_id: str, new_hash: str) -> bool:
        """Rotate a user's API key hash. Returns True if user found."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE users SET api_key_hash = $1 WHERE id = $2",
                    new_hash,
                    user_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("update_api_key_hash: database update failed")
            return False

    async def update_all_access(self, user_id: str, all_access: bool) -> bool:
        """Set a user's all_access flag. Returns True if the user was found."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE users SET all_access = $1 WHERE id = $2",
                    all_access,
                    user_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("update_all_access: database update failed")
            return False

    async def delete_user(self, user_id: str) -> bool:
        """Soft-delete a user (set active=False). Returns True if found."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE users SET active = FALSE WHERE id = $1",
                    user_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("delete_user: database update failed")
            return False

    async def count_users(self) -> int:
        """Return the total number of active users."""
        pool = await self._get_pool()
        if pool is None:
            return 0

        try:
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM users WHERE active = TRUE"
                )
                return int(count or 0)
        except Exception:
            logger.exception("count_users: database query failed")
            return 0

    async def update_password(self, user_id: str, password_hash: str) -> bool:
        """Update a user's bcrypt password hash. Returns True if user found."""
        pool = await self._get_pool()
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE users SET password_hash = $1 WHERE id = $2",
                    password_hash,
                    user_id,
                )
                return "UPDATE 1" in (result or "")
        except Exception:
            logger.exception("update_password: database update failed")
            return False
