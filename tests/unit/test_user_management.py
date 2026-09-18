"""Detroit-style unit tests for user management endpoints (US3).

Mock only out-of-process dependencies (hash_api_key, asyncpg pool via mocked
PostgreSQLUserStore methods). The real app, auth middleware, and domain
objects are exercised.

Principle VIII — classical unit tests: use the real FastAPI app with
the real AuthMiddleware, mocking only the UserStorePort persistence.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from treeloom.domain.authorization import User, Role, UserStorePort

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_hmac(mocker):
    """Patch hash_api_key so we control hashing deterministically.

    HMAC is an out-of-process concern (crypto). We mock hash_api_key
    to return a deterministic string based on the raw key.
    """
    def _mock_hash(raw_key: str) -> str:
        return f"hmac-mock-{raw_key}"

    mocker.patch(
        "treeloom.adapters.authorization.user_store.hash_api_key",
        side_effect=_mock_hash,
    )
    mocker.patch(
        "treeloom.application.auth_middleware.hash_api_key",
        side_effect=_mock_hash,
    )


def _make_user(
    user_id: str = "user-001",
    username: str = "dev1",
    role: Role = Role.USER,
    api_key_hash: str = "hmac-mock-secret-key",
    active: bool = True,
) -> User:
    """Create a User entity for test setup."""
    return User(
        id=user_id,
        username=username,
        email=f"{username}@example.com",
        api_key_hash=api_key_hash,
        role=role,
        active=active,
    )


def _mock_store_for_auth(mocker, mock_store, users_by_hash: dict[str, User]):
    """Configure mock_user_store.get_by_api_key_hash to return users by hash.

    The auth middleware calls _user_store.get_by_api_key_hash(key_hash).
    We mock this method to return the User matching the hash.
    """

    async def _get_by_hash(key_hash: str):
        return users_by_hash.get(key_hash)

    mock_store.get_by_api_key_hash = AsyncMock(side_effect=_get_by_hash)
    # Also mock other methods that endpoints may use
    mock_store.create_user = AsyncMock()
    mock_store.list_users = AsyncMock(return_value=[])
    mock_store.delete_user = AsyncMock(return_value=True)
    mock_store.get_by_id = AsyncMock(return_value=None)
    mock_store.update_api_key_hash = AsyncMock(return_value=True)
    mock_store.count_users = AsyncMock(return_value=0)


# ===========================================================================
# T025: User Management Endpoint Tests
# ===========================================================================


class TestCreateUserAsAdmin:
    """test_create_user_as_admin: admin creates user → 201 with user record."""

    @pytest.mark.asyncio
    async def test_create_user_as_admin(self, mocker):
        """Admin POST /users → 201 with id, username, role, api_key."""
        _patch_hmac(mocker)

        # Import app and _user_store after HMAC is patched
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        # Set up auth as admin
        admin_user = _make_user(
            user_id="admin-001",
            username="admin",
            role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )
        users_by_hash = {"hmac-mock-admin-key": admin_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)

        # Mock create_user to return a new user
        async def _create_user(username, email="", role=Role.USER, api_key_hash=""):
            return User(
                id="user-002",
                username=username,
                email=email,
                api_key_hash=api_key_hash,
                role=role,
                active=True,
            )

        _user_store.create_user = AsyncMock(side_effect=_create_user)

        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.post(
            "/users",
            json={"username": "newuser", "email": "new@example.com", "role": "user"},
            headers={"Authorization": "Bearer admin-key"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert "id" in data
        assert "role" in data
        assert "api_key" in data  # raw key shown once
        assert "api_key_hash" not in data  # never expose the hash

    @pytest.mark.asyncio
    async def test_create_user_default_role(self, mocker):
        """Admin creates user without role → defaults to 'user'."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        admin_user = _make_user(
            user_id="admin-001", username="admin", role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )
        users_by_hash = {"hmac-mock-admin-key": admin_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)

        created_users = []

        async def _create_user(username, email="", role=Role.USER, api_key_hash=""):
            u = User(
                id="user-003",
                username=username,
                email=email,
                api_key_hash=api_key_hash,
                role=role,
                active=True,
            )
            created_users.append(u)
            return u

        _user_store.create_user = AsyncMock(side_effect=_create_user)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.post(
            "/users",
            json={"username": "defaultrole"},
            headers={"Authorization": "Bearer admin-key"},
        )

        assert response.status_code == 201
        assert created_users[0].role == Role.USER


class TestCreateUserAsUserForbidden:
    """test_create_user_as_user_forbidden: non-admin user tries → 403."""

    @pytest.mark.asyncio
    async def test_create_user_as_regular_user_forbidden(self, mocker):
        """Regular user POST /users → 403."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        regular_user = _make_user(
            user_id="user-001",
            username="dev1",
            role=Role.USER,
            api_key_hash="hmac-mock-user-key",
        )
        users_by_hash = {"hmac-mock-user-key": regular_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.post(
            "/users",
            json={"username": "shouldfail"},
            headers={"Authorization": "Bearer user-key"},
        )

        assert response.status_code == 403


class TestListUsersAsAdmin:
    """test_list_users_as_admin: admin lists users → 200 with list."""

    @pytest.mark.asyncio
    async def test_list_users_as_admin(self, mocker):
        """Admin GET /users → 200 with list of users (no api_key_hash)."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        admin_user = _make_user(
            user_id="admin-001", username="admin", role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )
        users_by_hash = {"hmac-mock-admin-key": admin_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)

        store_users = [
            _make_user(user_id="u1", username="alice", role=Role.ADMIN),
            _make_user(user_id="u2", username="bob", role=Role.USER),
        ]
        _user_store.list_users = AsyncMock(return_value=store_users)

        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.get(
            "/users",
            headers={"Authorization": "Bearer admin-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["username"] == "alice"
        assert data[1]["username"] == "bob"
        # NEVER expose api_key_hash
        for user_data in data:
            assert "api_key_hash" not in user_data


class TestListUsersAsUserForbidden:
    """test_list_users_as_user_forbidden: non-admin tries → 403."""

    @pytest.mark.asyncio
    async def test_list_users_as_regular_user_forbidden(self, mocker):
        """Regular user GET /users → 403."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        regular_user = _make_user(
            user_id="user-001", username="dev1", role=Role.USER,
            api_key_hash="hmac-mock-user-key",
        )
        users_by_hash = {"hmac-mock-user-key": regular_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.get(
            "/users",
            headers={"Authorization": "Bearer user-key"},
        )

        assert response.status_code == 403


class TestDeleteUserAsAdmin:
    """test_delete_user_as_admin: admin deletes user → 204."""

    @pytest.mark.asyncio
    async def test_delete_user_as_admin(self, mocker):
        """Admin DELETE /users/{id} → 204."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        admin_user = _make_user(
            user_id="admin-001", username="admin", role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )
        users_by_hash = {"hmac-mock-admin-key": admin_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        _user_store.delete_user = AsyncMock(return_value=True)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.delete(
            "/users/user-to-delete",
            headers={"Authorization": "Bearer admin-key"},
        )

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_deleted_user_cannot_authenticate(self, mocker):
        """After deletion, deleted user's API key no longer works."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        admin_user = _make_user(
            user_id="admin-001", username="admin", role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )
        # The deleted user
        user_to_delete = _make_user(
            user_id="victim-id",
            username="victim",
            role=Role.USER,
            api_key_hash="hmac-mock-victim-key",
        )

        users_by_hash = {"hmac-mock-admin-key": admin_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        _user_store.delete_user = AsyncMock(return_value=True)

        # After deletion, get_by_api_key_hash returns None for victim
        deleted_users: set[str] = set()

        async def _get_by_hash(key_hash: str):
            if key_hash == "hmac-mock-admin-key":
                return admin_user
            if key_hash == "hmac-mock-victim-key" and "victim-id" in deleted_users:
                return None
            if key_hash == "hmac-mock-victim-key":
                return user_to_delete
            return None

        _user_store.get_by_api_key_hash = AsyncMock(side_effect=_get_by_hash)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)

        # Delete the victim user
        response = client.delete(
            "/users/victim-id",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert response.status_code == 204

        # Mark as deleted so subsequent auth fails
        deleted_users.add("victim-id")

        # Now try to authenticate as the deleted user
        response = client.get(
            "/users",
            headers={"Authorization": "Bearer victim-key"},
        )
        assert response.status_code == 401


class TestDeleteUserAsUserForbidden:
    """test_delete_user_as_user_forbidden: non-admin tries → 403."""

    @pytest.mark.asyncio
    async def test_delete_user_as_regular_user_forbidden(self, mocker):
        """Regular user DELETE /users/{id} → 403."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        regular_user = _make_user(
            user_id="user-001", username="dev1", role=Role.USER,
            api_key_hash="hmac-mock-user-key",
        )
        users_by_hash = {"hmac-mock-user-key": regular_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.delete(
            "/users/some-other-user",
            headers={"Authorization": "Bearer user-key"},
        )

        assert response.status_code == 403


class TestRotateKey:
    """test_rotate_key: admin rotates own key → 200 with new key."""

    @pytest.mark.asyncio
    async def test_rotate_own_key_as_admin(self, mocker):
        """Admin POST /users/{self_id}/rotate-key → 200 with new api_key."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        admin_user = _make_user(
            user_id="admin-001", username="admin", role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )

        # Track hash updates
        updated_hashes: dict[str, str] = {}

        async def _update_hash(user_id: str, new_hash: str) -> bool:
            updated_hashes[user_id] = new_hash
            return True

        _user_store.update_api_key_hash = AsyncMock(side_effect=_update_hash)

        users_by_hash = {"hmac-mock-admin-key": admin_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        _user_store.get_by_api_key_hash = AsyncMock(
            side_effect=lambda kh: admin_user if kh == "hmac-mock-admin-key" else None
        )

        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.post(
            "/users/admin-001/rotate-key",
            headers={"Authorization": "Bearer admin-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["api_key"] != ""  # new key is returned

    @pytest.mark.asyncio
    async def test_old_key_stops_working_after_rotation(self, mocker):
        """After key rotation, old API key returns 401."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        admin_user = _make_user(
            user_id="admin-001", username="admin", role=Role.ADMIN,
            api_key_hash="hmac-mock-admin-key",
        )

        # The current hash that gets looked up
        current_hash: str = "hmac-mock-admin-key"

        async def _get_by_hash(key_hash: str):
            if key_hash == current_hash:
                return admin_user
            return None

        _user_store.get_by_api_key_hash = AsyncMock(side_effect=_get_by_hash)

        async def _update_hash(user_id: str, new_hash: str) -> bool:
            nonlocal current_hash
            current_hash = new_hash
            return True

        _user_store.update_api_key_hash = AsyncMock(side_effect=_update_hash)
        _user_store.list_users = AsyncMock(return_value=[])
        _user_store.create_user = AsyncMock()
        _user_store.delete_user = AsyncMock(return_value=True)
        _user_store.get_by_id = AsyncMock(return_value=None)
        _user_store.count_users = AsyncMock(return_value=0)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)

        # Rotate key
        response = client.post(
            "/users/admin-001/rotate-key",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert response.status_code == 200

        # Now the old key should stop working
        # (current_hash was updated to the new hash by _update_hash)
        # Try auth with old key pattern
        response = client.get(
            "/users",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rotate_key_as_regular_user_self(self, mocker):
        """Regular user can rotate their own key."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        regular_user = _make_user(
            user_id="user-001", username="dev1", role=Role.USER,
            api_key_hash="hmac-mock-user-key",
        )
        users_by_hash = {"hmac-mock-user-key": regular_user}

        async def _update_hash(user_id: str, new_hash: str) -> bool:
            return True

        _user_store.update_api_key_hash = AsyncMock(side_effect=_update_hash)
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.post(
            "/users/user-001/rotate-key",
            headers={"Authorization": "Bearer user-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data

    @pytest.mark.asyncio
    async def test_rotate_key_as_user_for_other_user_forbidden(self, mocker):
        """Regular user cannot rotate another user's key."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        regular_user = _make_user(
            user_id="user-001", username="dev1", role=Role.USER,
            api_key_hash="hmac-mock-user-key",
        )
        users_by_hash = {"hmac-mock-user-key": regular_user}
        _mock_store_for_auth(mocker, _user_store, users_by_hash)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        client = TestClient(app)
        response = client.post(
            "/users/user-002/rotate-key",
            headers={"Authorization": "Bearer user-key"},
        )

        assert response.status_code == 403


class TestSeedAdminOnFirstStart:
    """test_seed_admin_on_first_start: count_users==0 seeds admin from env."""

    @pytest.mark.asyncio
    async def test_seed_admin_when_no_users(self, mocker):
        """When count_users returns 0 and TREELOOM_ADMIN_KEY is set,
        an admin user is seeded on startup."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        _user_store.count_users = AsyncMock(return_value=0)

        seeded_users: list[User] = []

        async def _create_user(username, email="", role=Role.USER, api_key_hash=""):
            u = User(
                id="seeded-admin-id",
                username=username,
                email=email,
                api_key_hash=api_key_hash,
                role=role,
                active=True,
            )
            seeded_users.append(u)
            return u

        _user_store.create_user = AsyncMock(side_effect=_create_user)
        _user_store.list_users = AsyncMock(return_value=[])
        _user_store.delete_user = AsyncMock(return_value=True)
        _user_store.get_by_id = AsyncMock(return_value=None)
        _user_store.update_api_key_hash = AsyncMock(return_value=True)
        _user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        admin_key = "initial-admin-key-42"
        mocker.patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "TREELOOM_ADMIN_KEY": admin_key},
        )

        # Exercise the extracted seed step directly — the full startup()
        # hook requires reachable Postgres (JobStore fail-loud) by design.
        from treeloom.application.lifecycle import _seed_admin_if_first_start

        await _seed_admin_if_first_start()

        assert len(seeded_users) == 1
        admin = seeded_users[0]
        assert admin.username == "admin"
        assert admin.role == Role.ADMIN
        assert admin.active is True

        # The hash should be HMAC-hashed admin_key
        # Since hash_api_key is mocked, hash_api_key("initial-admin-key-42") → "hmac-mock-initial-admin-key-42"
        assert admin.api_key_hash == "hmac-mock-initial-admin-key-42"

    @pytest.mark.asyncio
    async def test_no_seed_when_users_exist(self, mocker):
        """When count_users > 0, no admin is seeded."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        _user_store.count_users = AsyncMock(return_value=5)
        seeded_users: list[User] = []

        async def _create_user(username, email="", role=Role.USER, api_key_hash=""):
            u = User(
                id="should-not-exist",
                username=username,
                email=email,
                api_key_hash=api_key_hash,
                role=role,
                active=True,
            )
            seeded_users.append(u)
            return u

        _user_store.create_user = AsyncMock(side_effect=_create_user)

        mocker.patch.dict(os.environ, {"TREELOOM_ADMIN_KEY": "some-key"})

        from treeloom.application.lifecycle import _seed_admin_if_first_start
        await _seed_admin_if_first_start()

        assert len(seeded_users) == 0

    @pytest.mark.asyncio
    async def test_no_seed_when_no_env_key(self, mocker):
        """When count_users is 0 but TREELOOM_ADMIN_KEY is not set, no seed."""
        _patch_hmac(mocker)
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        _user_store.count_users = AsyncMock(return_value=0)
        seeded_users: list[User] = []

        async def _create_user(username, email="", role=Role.USER, api_key_hash=""):
            u = User(
                id="should-not-exist-2",
                username=username,
                email=email,
                api_key_hash=api_key_hash,
                role=role,
                active=True,
            )
            seeded_users.append(u)
            return u

        _user_store.create_user = AsyncMock(side_effect=_create_user)

        # No TREELOOM_ADMIN_KEY in env
        mocker.patch.dict(os.environ, {}, clear=True)

        from treeloom.application.lifecycle import _seed_admin_if_first_start
        await _seed_admin_if_first_start()

        assert len(seeded_users) == 0
