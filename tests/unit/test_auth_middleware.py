"""Detroit-style unit tests for the FastAPI auth middleware (US2).

Mock only out-of-process dependencies (asyncpg pool for PostgreSQLUserStore,
hmac for API key hashing). Never mock internal treeloom classes.

Principle VIII — classical unit tests: the real middleware, real UserStorePort
adapter (with a mock pool), and real hmac behavior are exercised.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from treeloom.domain.authorization import User, Role, Permission, UserStorePort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_hmac(mocker):
    """Patch hash_api_key so we control hashing deterministically.

    HMAC is an out-of-process concern (crypto). We mock hash_api_key so we
    know exactly what hash to store and can do exact-match lookups.
    """
    def _mock_hash(raw_key: str) -> str:
        # Deterministic: prepend a marker so tests can predict
        return f"hmac-mock-{raw_key}"

    mocker.patch(
        "treeloom.adapters.authorization.user_store.hash_api_key",
        side_effect=_mock_hash,
    )

    # Also patch where it's imported in the middleware
    mocker.patch(
        "treeloom.application.auth_middleware.hash_api_key",
        side_effect=_mock_hash,
    )


# ---------------------------------------------------------------------------
# Mock UserStorePort
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_user_store() -> UserStorePort:
    """A mock UserStorePort that tracks API key hash lookups.

    This is the only 'internal' interface we mock because it's the port
    the middleware depends on. In Detroit-style, we mock the port interface
    to isolate the middleware from the real database.
    """
    store = AsyncMock(spec=UserStorePort)
    store._users_by_hash: dict[str, User] = {}  # type: ignore[attr-defined]

    async def _get_by_hash(key_hash: str):
        return store._users_by_hash.get(key_hash)

    store.get_by_api_key_hash = AsyncMock(side_effect=_get_by_hash)
    return store


def _make_user(username="dev1", role=Role.USER) -> User:
    return User(
        id="user-001",
        username=username,
        email=f"{username}@example.com",
        api_key_hash="hmac-mock-***",
        role=role,
        active=True,
    )


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------

def _build_test_app(mocker, mock_user_store, auth_enabled: str = "true"):
    """Build a FastAPI test app with the auth middleware wired in.

    Uses the real AuthMiddleware class, only mocking the UserStorePort
    and hash_api_key (out-of-process).
    """
    app = FastAPI()

    # Set env for AUTH_ENABLED check (properly isolated via mocker.patch.dict)
    mocker.patch.dict(os.environ, {"AUTH_ENABLED": auth_enabled})

    # We create the middleware here. The middleware needs UserStorePort and
    # hash_api_key. We patch hash_api_key at the module level where it's used.
    from treeloom.application.auth_middleware import AuthMiddleware

    app.add_middleware(
        AuthMiddleware,
        user_store=mock_user_store,
    )

    @app.get("/protected")
    async def protected_route():
        return {"status": "ok"}

    @app.get("/health")
    async def health_route():
        return {"status": "healthy"}

    @app.get("/search")
    async def search_route():
        return {"results": []}

    @app.get("/status")
    async def status_route():
        return {"status": "idle"}

    return app


# ===========================================================================
# T018: Auth Middleware Tests
# ===========================================================================


class TestCorsPreflightBypassesAuth:
    """CORS preflight (OPTIONS) must not be 401'd when auth is enabled, or the
    browser blocks every cross-origin SPA request before it starts (regression
    for the operator UI v2 'failed to load' bug)."""

    @pytest.mark.asyncio
    async def test_options_preflight_not_blocked_with_auth_enabled(
        self, mocker, mock_user_store
    ):
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)
        resp = client.options(
            "/protected",
            headers={
                "Origin": "http://localhost:3001",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        # Before the fix this was 401 (AuthMiddleware rejecting the credential-less
        # preflight). After: OPTIONS bypasses auth, so it is never 401.
        assert resp.status_code != 401


class TestAuthNoHeaderReturns401:
    """test_auth_no_header_returns_401: request without Auth header → 401."""

    @pytest.mark.asyncio
    async def test_no_auth_header_on_protected_route(
        self, mocker, mock_user_store
    ):
        """Request to a protected route without Authorization header → 401."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get("/protected")
        assert response.status_code == 401


class TestAuthValidKeyAttachesUser:
    """test_auth_valid_key_attaches_user: valid API key → 200, user in state."""

    @pytest.mark.asyncio
    async def test_valid_api_key_attaches_user_to_request(
        self, mocker, mock_user_store
    ):
        """Request with valid Authorization: Bearer *** → 200, request.state.user is set."""
        _patch_hmac(mocker)

        user = _make_user(username="dev1", role=Role.USER)
        # HMAC mock will hash "***" → "hmac-mock-***"
        mock_user_store._users_by_hash["hmac-mock-***"] = user

        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer ***"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_401(
        self, mocker, mock_user_store
    ):
        """Request with invalid API key → 401."""
        _patch_hmac(mocker)

        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer bad-key"},
        )
        assert response.status_code == 401


class TestAuthDisabledBypasses:
    """test_auth_disabled_bypasses: AUTH_ENABLED=false → bypass auth checks."""

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_request_without_header(
        self, mocker, mock_user_store
    ):
        """With AUTH_ENABLED=false, request without auth header → 200."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="false")
        client = TestClient(app)

        response = client.get("/protected")
        assert response.status_code == 200


class TestHealthSearchBypass:
    """Health and search endpoints always bypass auth, even when enabled."""

    @pytest.mark.asyncio
    async def test_health_endpoint_bypasses_auth(self, mocker, mock_user_store):
        """GET /health always returns 200 regardless of auth."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_endpoint_bypasses_auth(self, mocker, mock_user_store):
        """GET /search always returns 200 regardless of auth."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get("/search")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_status_endpoint_bypasses_auth(self, mocker, mock_user_store):
        """GET /status is public — returns 200 without an Auth header when auth is on."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get("/status")
        assert response.status_code == 200


class TestAuthMalformedHeader:
    """Edge cases for malformed Authorization headers."""

    @pytest.mark.asyncio
    async def test_bearer_without_key_returns_401(
        self, mocker, mock_user_store
    ):
        """Authorization header with 'Bearer' but no key → 401."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_auth_scheme_returns_401(
        self, mocker, mock_user_store
    ):
        """Authorization header with wrong scheme (e.g. Basic) → 401."""
        _patch_hmac(mocker)
        app = _build_test_app(mocker, mock_user_store, auth_enabled="true")
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Basic abc123"},
        )
        assert response.status_code == 401


# ===========================================================================
# T021: PostgreSQLUserStore Tests (via mock pool)
# ===========================================================================


@pytest.fixture
def mock_pool() -> AsyncMock:
    """A mock asyncpg.Pool that simulates the users table."""
    pool = AsyncMock()

    # Internal storage dict that simulates the database
    _store: dict[str, dict] = {}
    _next_id = 0

    conn = AsyncMock()

    async def _fetchrow(query, *args):
        """Simulate SELECT by api_key_hash or id."""
        if "SELECT" in query and "FROM users" in query:
            if "api_key_hash" in query and args:
                hash_val = args[0]
                for row in _store.values():
                    if row.get("api_key_hash") == hash_val:
                        # Respect the "active = TRUE" filter in the query
                        if "active = TRUE" in query and not row.get("active"):
                            return None
                        return _make_mock_row(row)
                return None
            if "id = $1" in query and args:
                uid = args[0]
                if uid in _store:
                    return _make_mock_row(_store[uid])
                return None
        return None

    async def _fetch(query, *args):
        """Simulate SELECT list_users."""
        if "SELECT" in query and "FROM users" in query and "active" in query:
            active = [r for r in _store.values() if r.get("active") is True]
            return [_make_mock_row(r) for r in active]
        return []

    async def _fetchval(query, *args):
        """Simulate SELECT COUNT(*)."""
        if "COUNT(*)" in query:
            active = [r for r in _store.values() if r.get("active") is True]
            return len(active)
        return 0

    async def _execute(query, *args):
        """Simulate INSERT/UPDATE/DELETE."""
        nonlocal _next_id
        if "INSERT" in query and "users" in query:
            uid = args[0]
            _store[uid] = {
                "id": uid,
                "username": args[1],
                "email": args[2] if len(args) > 2 else "",
                "api_key_hash": args[3] if len(args) > 3 else "",
                "role": args[4] if len(args) > 4 else "user",
                "active": True,
                "created_at": "2025-01-01T00:00:00+00:00",
            }
            return "INSERT 1"
        elif "UPDATE" in query and "SET role" in query:
            uid = args[1] if len(args) > 1 else None
            if uid and uid in _store:
                _store[uid]["role"] = args[0]
                return "UPDATE 1"
            return "UPDATE 0"
        elif "UPDATE" in query and "SET api_key_hash" in query:
            uid = args[1] if len(args) > 1 else None
            if uid and uid in _store:
                _store[uid]["api_key_hash"] = args[0]
                return "UPDATE 1"
            return "UPDATE 0"
        elif "UPDATE" in query and "SET active" in query:
            uid = args[0] if args else None
            if uid and uid in _store:
                _store[uid]["active"] = False
                return "UPDATE 1"
            return "UPDATE 0"
        return "OK"

    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.fetch = AsyncMock(side_effect=_fetch)
    conn.fetchval = AsyncMock(side_effect=_fetchval)
    conn.execute = AsyncMock(side_effect=_execute)

    _acquire_ctx = MagicMock()
    _acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    _acquire_ctx.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=_acquire_ctx)

    return pool


def _make_mock_row(data: dict) -> MagicMock:
    """Create an asyncpg.Record-like mock from a dict."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, k: data[k]
    mock.get = data.get
    mock.keys.return_value = list(data.keys())
    mock.values.return_value = list(data.values())
    return mock


class TestPostgreSQLUserStoreCreateUser:
    """T021: create_user inserts a user and returns a User entity."""

    @pytest.mark.asyncio
    async def test_create_user_returns_user(self, mock_pool):
        """create_user() should insert and return a User with correct fields."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        user = await store.create_user(
            username="testuser",
            email="test@example.com",
            role=Role.USER,
            api_key_hash="hmac-mock-hashedkey",
        )

        assert isinstance(user, User)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == Role.USER
        assert user.api_key_hash == "hmac-mock-hashedkey"
        assert user.active is True


class TestPostgreSQLUserStoreGetByApiKeyHash:
    """T021: get_by_api_key_hash retrieves a user by hash."""

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_user(self, mock_pool):
        """After creating a user, get_by_api_key_hash returns it."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        await store.create_user(
            username="findme",
            email="find@example.com",
            role=Role.USER,
            api_key_hash="hmac-mock-myhash",
        )
        user = await store.get_by_api_key_hash("hmac-mock-myhash")
        assert user is not None
        assert user.username == "findme"

    @pytest.mark.asyncio
    async def test_get_by_hash_returns_none(self, mock_pool):
        """Unknown hash returns None."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        user = await store.get_by_api_key_hash("nonexistent")
        assert user is None


class TestPostgreSQLUserStoreGetById:
    """T021: get_by_id retrieves a user by id."""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_user(self, mock_pool):
        """After creating a user, get_by_id returns it."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        created = await store.create_user(
            username="byid",
            email="byid@example.com",
            role=Role.ADMIN,
            api_key_hash="hmac-mock-byidhash",
        )
        user = await store.get_by_id(created.id)
        assert user is not None
        assert user.username == "byid"
        assert user.role == Role.ADMIN

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none(self, mock_pool):
        """Unknown id returns None."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        user = await store.get_by_id("nonexistent-id")
        assert user is None


class TestPostgreSQLUserStoreListUsers:
    """T021: list_users returns all active users."""

    @pytest.mark.asyncio
    async def test_list_users_returns_all(self, mock_pool):
        """list_users() returns all active users in creation order."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        await store.create_user(username="u1", email="u1@x.com", role=Role.USER, api_key_hash="h1")
        await store.create_user(username="u2", email="u2@x.com", role=Role.ADMIN, api_key_hash="h2")

        users = await store.list_users()
        assert len(users) == 2
        usernames = {u.username for u in users}
        assert usernames == {"u1", "u2"}


class TestPostgreSQLUserStoreCountUsers:
    """T021: count_users returns total active users."""

    @pytest.mark.asyncio
    async def test_count_users(self, mock_pool):
        """count_users() returns the number of active users."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        assert await store.count_users() == 0

        await store.create_user(username="c1", api_key_hash="h1")
        assert await store.count_users() == 1

        await store.create_user(username="c2", api_key_hash="h2")
        assert await store.count_users() == 2


class TestPostgreSQLUserStoreUpdateRole:
    """T021: update_role changes a user's role."""

    @pytest.mark.asyncio
    async def test_update_role(self, mock_pool):
        """update_role changes role and returns True."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        created = await store.create_user(
            username="roleuser",
            role=Role.USER,
            api_key_hash="h1",
        )
        assert created.role == Role.USER

        result = await store.update_role(created.id, Role.ADMIN)
        assert result is True

        updated = await store.get_by_id(created.id)
        assert updated is not None
        assert updated.role == Role.ADMIN


class TestPostgreSQLUserStoreUpdateApiKeyHash:
    """T021: update_api_key_hash rotates a user's API key hash."""

    @pytest.mark.asyncio
    async def test_update_api_key_hash(self, mock_pool):
        """update_api_key_hash updates the hash and returns True."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        created = await store.create_user(
            username="keyuser",
            api_key_hash="old-hash",
        )

        result = await store.update_api_key_hash(created.id, "new-hash")
        assert result is True

        # Old hash lookup should return None now
        old_result = await store.get_by_api_key_hash("old-hash")
        assert old_result is None

        # New hash lookup should return the user
        new_result = await store.get_by_api_key_hash("new-hash")
        assert new_result is not None
        assert new_result.id == created.id


class TestPostgreSQLUserStoreDeleteUser:
    """T021: delete_user soft-deletes a user."""

    @pytest.mark.asyncio
    async def test_delete_user(self, mock_pool):
        """delete_user sets active=False and returns True."""
        from treeloom.adapters.authorization.user_store import PostgreSQLUserStore

        store = PostgreSQLUserStore(pool=mock_pool)
        created = await store.create_user(
            username="deluser",
            api_key_hash="del-hash",
        )

        result = await store.delete_user(created.id)
        assert result is True

        # After soft-delete, get_by_api_key_hash should return None (active=False)
        user = await store.get_by_api_key_hash("del-hash")
        assert user is None
