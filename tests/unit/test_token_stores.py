"""Detroit-style unit tests for the three new token/session stores.

Mock only the asyncpg pool (out-of-process). The real store adapters are
exercised against a simulated in-memory DB row structure.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared pool/row mock helpers
# ---------------------------------------------------------------------------


def _make_mock_row(data: dict) -> MagicMock:
    """asyncpg.Record-like mock from a dict."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, k: data[k]
    mock.get = lambda k, default=None: data.get(k, default)
    mock.keys.return_value = list(data.keys())
    return mock


def _make_pool(store: dict, table: str) -> AsyncMock:
    """Build a minimal AsyncMock pool that simulates one table.

    Supports fetchrow, fetch, execute for INSERT/UPDATE/DELETE.
    """
    pool = AsyncMock()
    conn = AsyncMock()

    async def _fetchrow(query: str, *args):
        q = query.upper()
        if "SELECT" in q and "WHERE" in q and args:
            token_hash = args[0]
            for row in store.values():
                if row.get("token_hash") == token_hash:
                    return _make_mock_row(row)
        return None

    async def _fetch(query: str, *args):
        q = query.upper()
        if "SELECT" in q and args:
            uid = args[0]
            return [_make_mock_row(r) for r in store.values() if r.get("user_id") == uid]
        if "SELECT" in q:
            return [_make_mock_row(r) for r in store.values()]
        return []

    async def _execute(query: str, *args):
        q = query.upper()
        if "INSERT INTO" in q:
            # positional args map to columns by INSERT order
            row_id = args[0]
            row: dict = {"id": row_id}
            # Copy all positional args by index
            store[row_id] = {k: (args[i] if i < len(args) else None)
                             for i, k in enumerate(["id"] + list(store.get("_cols", {}).get(table, [])))}
            # Simpler: just store by id with a dict built from args
            if table == "personal_access_tokens":
                row = {
                    "id": args[0], "user_id": args[1], "name": args[2],
                    "token_hash": args[3], "scopes": list(args[4] or []),
                    "created_at": args[5], "expires_at": args[6] if len(args) > 6 else None,
                    "revoked": False, "last_used_at": None,
                }
            elif table == "api_keys":
                row = {
                    "id": args[0], "name": args[1], "token_hash": args[2],
                    "scopes": list(args[3] or []), "all_access": bool(args[4]),
                    "created_by": args[5] if len(args) > 5 else None,
                    "created_at": args[6] if len(args) > 6 else None,
                    "expires_at": args[7] if len(args) > 7 else None,
                    "revoked": False, "last_used_at": None,
                }
            elif table == "sessions":
                row = {
                    "id": args[0], "user_id": args[1], "token_hash": args[2],
                    "created_at": args[3], "expires_at": args[4],
                    "last_seen_at": None,
                }
            store[row_id] = row
            return "INSERT 1"
        if "UPDATE" in q and "SET REVOKED = TRUE" in q:
            uid = args[0]
            if uid in store:
                store[uid]["revoked"] = True
                return "UPDATE 1"
            return "UPDATE 0"
        if "UPDATE" in q and "SET LAST_USED_AT" in q:
            uid = args[0]
            if uid in store:
                store[uid]["last_used_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        if "UPDATE" in q and "SET LAST_SEEN_AT" in q:
            # token_hash-based
            token_hash = args[0]
            for row in store.values():
                if row.get("token_hash") == token_hash:
                    row["last_seen_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        if "DELETE FROM" in q:
            if args:
                token_hash = args[0]
                to_del = [k for k, v in store.items() if v.get("token_hash") == token_hash]
                for k in to_del:
                    del store[k]
                return f"DELETE {len(to_del)}"
            return "DELETE 0"
        return "OK"

    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.fetch = AsyncMock(side_effect=_fetch)
    conn.execute = AsyncMock(side_effect=_execute)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=ctx)
    return pool


# ---------------------------------------------------------------------------
# PersonalAccessToken store tests
# ---------------------------------------------------------------------------


class TestTokenStoreCreate:
    """token_store.create returns a PersonalAccessToken dataclass."""

    @pytest.mark.asyncio
    async def test_create_returns_dataclass(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )
        from treeloom.domain.authorization import PersonalAccessToken

        store_data: dict = {}
        pool = _make_pool(store_data, "personal_access_tokens")
        store = PostgreSQLPersonalAccessTokenStore(pool=pool)

        pat = await store.create(
            user_id="user-1",
            name="my-token",
            token_hash="hmac-abc",
            scopes=["search", "index"],
        )
        assert pat is not None
        assert isinstance(pat, PersonalAccessToken)
        assert pat.user_id == "user-1"
        assert pat.name == "my-token"
        assert pat.scopes == ["search", "index"]
        assert pat.revoked is False


class TestTokenStoreGetByTokenHash:
    """token_store.get_by_token_hash hit and miss."""

    @pytest.mark.asyncio
    async def test_get_by_hash_hit(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        store_data: dict = {}
        pool = _make_pool(store_data, "personal_access_tokens")
        store = PostgreSQLPersonalAccessTokenStore(pool=pool)

        await store.create("user-1", "tok", "hash-xyz", ["search"])
        result = await store.get_by_token_hash("hash-xyz")
        assert result is not None
        assert result.token_hash == "hash-xyz"

    @pytest.mark.asyncio
    async def test_get_by_hash_miss(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        store_data: dict = {}
        pool = _make_pool(store_data, "personal_access_tokens")
        store = PostgreSQLPersonalAccessTokenStore(pool=pool)
        result = await store.get_by_token_hash("no-such-hash")
        assert result is None


class TestTokenStoreRevoke:
    """token_store.revoke marks a PAT as revoked."""

    @pytest.mark.asyncio
    async def test_revoke_returns_true(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        store_data: dict = {}
        pool = _make_pool(store_data, "personal_access_tokens")

        conn = AsyncMock()

        async def _execute(query: str, *args):
            if "UPDATE" in query.upper() and "REVOKED" in query.upper():
                return "UPDATE 1"
            return "OK"

        conn.execute = AsyncMock(side_effect=_execute)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLPersonalAccessTokenStore(pool=pool)
        result = await store.revoke("token-id-1", "user-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_not_found_returns_false(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        store_data: dict = {}
        pool = _make_pool(store_data, "personal_access_tokens")

        conn = AsyncMock()

        async def _execute(query: str, *args):
            return "UPDATE 0"

        conn.execute = AsyncMock(side_effect=_execute)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLPersonalAccessTokenStore(pool=pool)
        result = await store.revoke("nonexistent", "user-1")
        assert result is False


class TestTokenStoreNoPool:
    """token_store degrades gracefully when pool is unavailable."""

    @pytest.mark.asyncio
    async def test_create_returns_none_without_pool(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        store = PostgreSQLPersonalAccessTokenStore(pool=None)
        # Override _get_pool to return None
        store._pool = None
        store._get_pool = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await store.create("u", "n", "h", [])
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_none_without_pool(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        store = PostgreSQLPersonalAccessTokenStore(pool=None)
        store._get_pool = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await store.get_by_token_hash("h")
        assert result is None


# ---------------------------------------------------------------------------
# ApiKey store tests
# ---------------------------------------------------------------------------


class TestApiKeyStoreCreate:
    """api_key_store.create returns an ApiKey dataclass."""

    @pytest.mark.asyncio
    async def test_create_returns_dataclass(self):
        from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore
        from treeloom.domain.authorization import ApiKey

        store_data: dict = {}
        pool = _make_pool(store_data, "api_keys")
        store = PostgreSQLApiKeyStore(pool=pool)

        key = await store.create(
            name="ci-key",
            token_hash="hmac-ci",
            scopes=["search", "index"],
            all_access=False,
            created_by="admin-1",
        )
        assert key is not None
        assert isinstance(key, ApiKey)
        assert key.name == "ci-key"
        assert key.scopes == ["search", "index"]
        assert key.all_access is False
        assert key.created_by == "admin-1"


class TestApiKeyStoreGetByTokenHash:
    """api_key_store.get_by_token_hash hit/miss."""

    @pytest.mark.asyncio
    async def test_hit(self):
        from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore

        store_data: dict = {}
        pool = _make_pool(store_data, "api_keys")
        store = PostgreSQLApiKeyStore(pool=pool)

        await store.create("svc-key", "hash-svc", ["index"])
        result = await store.get_by_token_hash("hash-svc")
        assert result is not None
        assert result.token_hash == "hash-svc"

    @pytest.mark.asyncio
    async def test_miss(self):
        from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore

        store_data: dict = {}
        pool = _make_pool(store_data, "api_keys")
        store = PostgreSQLApiKeyStore(pool=pool)
        result = await store.get_by_token_hash("no-such")
        assert result is None


class TestApiKeyStoreRevoke:
    """api_key_store.revoke marks a key as revoked."""

    @pytest.mark.asyncio
    async def test_revoke_found(self):
        from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool = AsyncMock()
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLApiKeyStore(pool=pool)
        result = await store.revoke("key-id-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_not_found(self):
        from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool = AsyncMock()
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLApiKeyStore(pool=pool)
        result = await store.revoke("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Session store tests
# ---------------------------------------------------------------------------


class TestSessionStoreCreate:
    """session_store.create returns a Session dataclass."""

    @pytest.mark.asyncio
    async def test_create_returns_dataclass(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore
        from treeloom.domain.authorization import Session

        store_data: dict = {}
        pool = _make_pool(store_data, "sessions")
        store = PostgreSQLSessionStore(pool=pool)

        expires = datetime.now(timezone.utc) + timedelta(hours=12)
        session = await store.create(
            user_id="user-1",
            token_hash="sess-hash",
            expires_at=expires,
        )
        assert session is not None
        assert isinstance(session, Session)
        assert session.user_id == "user-1"
        assert session.token_hash == "sess-hash"


class TestSessionStoreGetByTokenHash:
    """session_store.get_by_token_hash hit/miss."""

    @pytest.mark.asyncio
    async def test_hit(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore

        store_data: dict = {}
        pool = _make_pool(store_data, "sessions")
        store = PostgreSQLSessionStore(pool=pool)

        expires = datetime.now(timezone.utc) + timedelta(hours=12)
        await store.create("user-1", "hash-sess", expires)
        result = await store.get_by_token_hash("hash-sess")
        assert result is not None
        assert result.token_hash == "hash-sess"

    @pytest.mark.asyncio
    async def test_miss(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore

        store_data: dict = {}
        pool = _make_pool(store_data, "sessions")
        store = PostgreSQLSessionStore(pool=pool)
        result = await store.get_by_token_hash("no-such-hash")
        assert result is None


class TestSessionStoreDeleteExpired:
    """session_store.delete_expired returns the count of deleted rows."""

    @pytest.mark.asyncio
    async def test_delete_expired_returns_count(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 3")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool = AsyncMock()
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLSessionStore(pool=pool)
        count = await store.delete_expired()
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_expired_zero(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 0")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool = AsyncMock()
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLSessionStore(pool=pool)
        count = await store.delete_expired()
        assert count == 0


class TestSessionStoreDelete:
    """session_store.delete removes a session."""

    @pytest.mark.asyncio
    async def test_delete_found(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 1")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool = AsyncMock()
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLSessionStore(pool=pool)
        result = await store.delete("some-hash")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 0")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)
        pool = AsyncMock()
        pool.acquire = MagicMock(return_value=ctx)

        store = PostgreSQLSessionStore(pool=pool)
        result = await store.delete("no-such")
        assert result is False
