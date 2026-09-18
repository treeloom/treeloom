"""Detroit-style unit tests for the local-auth REST endpoints.

Uses FastAPI TestClient with AsyncMock stores injected via the real app.
Follows test_user_management.py for store assembly patterns.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from treeloom.application import routes_auth as _rauth
from treeloom.application import indexer_authz as _authz
from fastapi.testclient import TestClient

from treeloom.domain.authorization import (
    User, Role, UserStorePort,
    TokenStorePort, ApiKeyStorePort, SessionStorePort,
    PersonalAccessToken, ApiKey, Session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_hmac(mocker):
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
    mocker.patch(
        "treeloom.application.indexer_service.hash_api_key",
        side_effect=_mock_hash,
        create=True,
    )


def _patch_password(mocker, verify_result: bool = True):
    mocker.patch(
        "treeloom.adapters.authorization.user_store.verify_password",
        return_value=verify_result,
    )
    mocker.patch(
        "treeloom.adapters.authorization.user_store.hash_password",
        return_value="$2b$12$fakehash",
    )
    mocker.patch(
        "treeloom.application.indexer_service.hash_password",
        return_value="$2b$12$fakehash",
        create=True,
    )
    mocker.patch(
        "treeloom.application.indexer_service.verify_password",
        return_value=verify_result,
        create=True,
    )


def _make_user(
    user_id: str = "admin-001",
    username: str = "admin",
    role: Role = Role.ADMIN,
    api_key_hash: str = "hmac-mock-admin-key",
    password_hash: str = "$2b$12$fakehash",
) -> User:
    return User(
        id=user_id,
        username=username,
        email=f"{username}@example.com",
        api_key_hash=api_key_hash,
        role=role,
        active=True,
        password_hash=password_hash,
    )


def _setup_admin_auth(mocker, admin_user: User):
    """Wire admin bearer auth for the real app's _user_store."""
    from treeloom.application.indexer_state import _user_store

    async def _get_by_hash(key_hash: str):
        if key_hash == admin_user.api_key_hash:
            return admin_user
        return None

    _user_store.get_by_api_key_hash = AsyncMock(side_effect=_get_by_hash)
    _user_store.get_by_id = AsyncMock(return_value=admin_user)
    _user_store.list_users = AsyncMock(return_value=[])
    _user_store.create_user = AsyncMock(return_value=None)
    _user_store.delete_user = AsyncMock(return_value=True)
    _user_store.update_api_key_hash = AsyncMock(return_value=True)
    _user_store.count_users = AsyncMock(return_value=1)
    _user_store.update_role = AsyncMock(return_value=True)
    _user_store.update_password = AsyncMock(return_value=True)
    _user_store.get_by_username = AsyncMock(return_value=None)


def _setup_session_store(mocker, session_result=None):
    from treeloom.application.indexer_state import _session_store

    _session_store.create = AsyncMock(return_value=session_result)
    _session_store.get_by_token_hash = AsyncMock(return_value=None)
    _session_store.delete = AsyncMock(return_value=True)
    _session_store.touch = AsyncMock()


def _setup_token_store(mocker):
    from treeloom.application.indexer_state import _token_store

    _token_store.create = AsyncMock(return_value=None)
    _token_store.get_by_token_hash = AsyncMock(return_value=None)
    _token_store.list_for_user = AsyncMock(return_value=[])
    _token_store.revoke = AsyncMock(return_value=False)
    _token_store.touch_last_used = AsyncMock()


def _setup_api_key_store(mocker):
    from treeloom.application.indexer_state import _api_key_store

    _api_key_store.create = AsyncMock(return_value=None)
    _api_key_store.get_by_token_hash = AsyncMock(return_value=None)
    _api_key_store.list_keys = AsyncMock(return_value=[])
    _api_key_store.revoke = AsyncMock(return_value=False)
    _api_key_store.touch_last_used = AsyncMock()


# ---------------------------------------------------------------------------
# Login / logout tests
# ---------------------------------------------------------------------------


class TestAuthLogin:
    """POST /auth/login success and failure."""

    @pytest.mark.asyncio
    async def test_login_success_sets_cookie(self, mocker):
        """Valid credentials → 200 with set-cookie header."""
        _patch_hmac(mocker)

        user = _make_user()
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        from treeloom.application.indexer_service import (
            app,
        )
        from treeloom.application.indexer_state import (
            _user_store,
            _session_store,
            _login_attempt_store,
        )

        _user_store.get_by_username = AsyncMock(return_value=user)
        _user_store.get_by_api_key_hash = AsyncMock(return_value=user)

        # Login rate-limiter store: no failures, accept writes.
        # PRIOR failure counts (the store records this attempt itself, and
        # only when not already locked). 0 prior = proceed.
        _login_attempt_store.atomic_record_and_count = AsyncMock(
            return_value=(0, 0)
        )
        _login_attempt_store.record = AsyncMock()
        _login_attempt_store.clear = AsyncMock()

        fake_session = Session(
            id="sess-1", user_id=user.id, token_hash="hmac-mock-x",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
        )
        _session_store.create = AsyncMock(return_value=fake_session)
        _session_store.get_by_token_hash = AsyncMock(return_value=None)

        # Patch verify_password at the source module (lazy import)
        mocker.patch(
            "treeloom.adapters.authorization.user_store.verify_password",
            return_value=True,
        )

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": "correct"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "user" in data
        assert "scopes" in data
        assert "treeloom_session" in resp.cookies

    @pytest.mark.asyncio
    async def test_login_bad_password_returns_401(self, mocker):
        """Wrong password → 401."""
        _patch_hmac(mocker)
        user = _make_user()
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        from treeloom.application.indexer_service import (
            app,
        )
        from treeloom.application.indexer_state import (
            _user_store,
            _login_attempt_store,
        )

        _user_store.get_by_username = AsyncMock(return_value=user)
        _user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        # PRIOR failure counts (the store records this attempt itself, and
        # only when not already locked). 0 prior = proceed.
        _login_attempt_store.atomic_record_and_count = AsyncMock(
            return_value=(0, 0)
        )
        _login_attempt_store.record = AsyncMock()
        _login_attempt_store.clear = AsyncMock()

        mocker.patch(
            "treeloom.adapters.authorization.user_store.verify_password",
            return_value=False,
        )

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_login_unknown_user_returns_401(self, mocker):
        """Unknown username → 401."""
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        from treeloom.application.indexer_service import (
            app,
        )
        from treeloom.application.indexer_state import (
            _user_store,
            _login_attempt_store,
        )

        _user_store.get_by_username = AsyncMock(return_value=None)
        _user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        # PRIOR failure counts (the store records this attempt itself, and
        # only when not already locked). 0 prior = proceed.
        _login_attempt_store.atomic_record_and_count = AsyncMock(
            return_value=(0, 0)
        )
        _login_attempt_store.record = AsyncMock()
        _login_attempt_store.clear = AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/auth/login",
            json={"username": "nobody", "password": "pass"},
        )
        assert resp.status_code == 401


    @pytest.mark.asyncio
    async def test_login_rate_limited_returns_429(self, mocker):
        """Failures at/over the threshold → 429 with Retry-After, before
        bcrypt even runs (verify_password must not be consulted)."""
        _patch_hmac(mocker)
        mocker.patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "TREELOOM_LOGIN_MAX_ATTEMPTS": "5"},
        )

        from treeloom.application import indexer_service as svc
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _login_attempt_store

        # Force the module-level threshold the handler reads.
        mocker.patch.object(_rauth, "LOGIN_MAX_ATTEMPTS", 5)
        mocker.patch.object(_rauth, "LOGIN_WINDOW_SECONDS", 300)

        # 5 PRIOR failures == LOGIN_MAX_ATTEMPTS -> locked.
        _login_attempt_store.atomic_record_and_count = AsyncMock(
            return_value=(5, 5)
        )
        _login_attempt_store.record = AsyncMock()
        _login_attempt_store.clear = AsyncMock()

        verify = mocker.patch(
            "treeloom.adapters.authorization.user_store.verify_password",
            return_value=True,
        )

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": "whatever"},
        )
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        verify.assert_not_called()

    @pytest.mark.asyncio
    async def test_lock_contention_is_refused_not_500(self, mocker):
        """The per-username lock waits a bounded time and then gives up, so a
        flood cannot hold one pool connection per waiter. Reaching that bound
        means more simultaneous attempts on one account than the lock can
        drain — refuse it like any other throttle decision.

        Letting the exception escape would be a 500 that still ran no bcrypt
        but leaked the internal failure, and would leave the caller without a
        Retry-After to back off on.
        """
        _patch_hmac(mocker)
        from treeloom.application import indexer_service as svc
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _login_attempt_store
        from treeloom.adapters.postgresql.login_attempt_store import (
            LoginThrottleContention,
        )

        _login_attempt_store.atomic_record_and_count = AsyncMock(
            side_effect=LoginThrottleContention("admin")
        )
        _login_attempt_store.record = AsyncMock()
        _login_attempt_store.clear = AsyncMock()
        verify = mocker.patch(
            "treeloom.adapters.authorization.user_store.verify_password",
            return_value=True,
        )

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/auth/login", json={"username": "admin", "password": "whatever"}
        )
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        verify.assert_not_called()
        _login_attempt_store.record.assert_not_awaited()


class TestAuthLogout:
    """POST /auth/logout clears session cookie."""

    @pytest.mark.asyncio
    async def test_logout_clears_cookie(self, mocker):
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store, _session_store

        _user_store.get_by_api_key_hash = AsyncMock(return_value=None)
        _session_store.delete = AsyncMock(return_value=True)
        _session_store.get_by_token_hash = AsyncMock(return_value=None)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/auth/logout",
            cookies={"treeloom_session": "some-token"},
        )
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# PAT tests
# ---------------------------------------------------------------------------


class TestPATCreate:
    """POST /users/{id}/tokens rejects over-privileged scopes."""

    @pytest.mark.asyncio
    async def test_pat_create_rejects_over_privileged_scopes(self, mocker):
        """A user-role owner cannot create an admin-scoped PAT."""
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        admin = _make_user(user_id="admin-1", role=Role.ADMIN)
        owner = _make_user(
            user_id="user-1", username="alice", role=Role.USER,
            api_key_hash="hmac-mock-admin-key",
        )

        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store, _token_store

        _user_store.get_by_api_key_hash = AsyncMock(return_value=admin)
        _user_store.get_by_id = AsyncMock(return_value=owner)
        _token_store.create = AsyncMock(return_value=None)
        _token_store.get_by_token_hash = AsyncMock(return_value=None)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/users/user-1/tokens",
            json={"name": "over-privileged", "scopes": ["admin"]},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_pat_create_allowed_scopes_returns_201(self, mocker):
        """Admin creating a search-scoped PAT for a user → 201 + token shown once."""
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        admin = _make_user(user_id="admin-1", role=Role.ADMIN)
        owner = _make_user(user_id="user-2", username="bob", role=Role.USER,
                           api_key_hash="hmac-mock-admin-key")

        fake_pat = PersonalAccessToken(
            id="pat-new", user_id="user-2", name="ci",
            token_hash="hmac-mock-x", scopes=["search"],
            created_at=datetime.now(timezone.utc),
        )

        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store, _token_store

        _user_store.get_by_api_key_hash = AsyncMock(return_value=admin)
        _user_store.get_by_id = AsyncMock(return_value=owner)
        _token_store.create = AsyncMock(return_value=fake_pat)
        _token_store.get_by_token_hash = AsyncMock(return_value=None)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/users/user-2/tokens",
            json={"name": "ci", "scopes": ["search"]},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "token" in data  # raw token shown once
        assert "token_hash" not in data  # never expose the hash


# ---------------------------------------------------------------------------
# _require_scope
# ---------------------------------------------------------------------------


class TestRequireScope:
    """_require_scope blocks callers whose token lacks the required scope."""

    @pytest.mark.asyncio
    async def test_scope_gate_blocks_search_only_token_from_index(self, mocker):
        """A search-only PAT cannot hit an index-scoped endpoint.

        Tests _require_scope directly by injecting state into a mock request.
        """
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_authz import _require_scope
        from fastapi import HTTPException

        pat_user = _make_user(user_id="user-3", username="restricted", role=Role.USER)

        # Build a mock request with a search-only scope set
        mock_request = MagicMock()
        mock_request.state.user = pat_user
        mock_request.state.scopes = {"search"}  # no "index"

        try:
            _require_scope(mock_request, "index")
            assert False, "Expected HTTPException 403"
        except HTTPException as exc:
            assert exc.status_code == 403

    @pytest.mark.asyncio
    async def test_scope_gate_passes_for_full_role_admin(self, mocker):
        """An admin via a full-role auth path (session/legacy) carries
        every scope and passes."""
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_authz import _require_scope

        admin_user = _make_user(user_id="admin-1", role=Role.ADMIN)
        mock_request = MagicMock()
        mock_request.state.user = admin_user
        # Full-role paths populate scopes from scopes_for_role(ADMIN).
        mock_request.state.scopes = {"search", "index", "admin"}

        result = _require_scope(mock_request, "index")
        assert result == admin_user

    @pytest.mark.asyncio
    async def test_scope_gate_blocks_admin_with_scope_limited_token(self, mocker):
        """An admin-owned PAT/API key minted without the scope is blocked —
        no role bypass (the over-privileged-token fix)."""
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_authz import _require_scope
        from fastapi import HTTPException

        admin_user = _make_user(user_id="admin-1", role=Role.ADMIN)
        mock_request = MagicMock()
        mock_request.state.user = admin_user
        mock_request.state.scopes = {"search"}  # scope-limited token

        try:
            _require_scope(mock_request, "index")
            assert False, "Expected HTTPException 403"
        except HTTPException as exc:
            assert exc.status_code == 403

    @pytest.mark.asyncio
    async def test_scope_gate_passes_when_scope_present(self, mocker):
        """User with matching scope is allowed."""
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_authz import _require_scope

        user = _make_user(user_id="user-1", role=Role.USER)
        mock_request = MagicMock()
        mock_request.state.user = user
        mock_request.state.scopes = {"search", "index"}

        result = _require_scope(mock_request, "index")
        assert result == user

    def test_scope_gate_disabled_when_auth_off(self, mocker):
        """When AUTH_ENABLED=false, _require_scope always returns None."""
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "false"})
        from treeloom.application.indexer_authz import _require_scope

        mock_request = MagicMock()
        result = _require_scope(mock_request, "admin")
        assert result is None


# ---------------------------------------------------------------------------
# PUT /users/{id}/role
# ---------------------------------------------------------------------------


class TestSetUserRole:
    """PUT /users/{id}/role — admin only."""

    @pytest.mark.asyncio
    async def test_put_role_admin_only(self, mocker):
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        non_admin = _make_user(
            user_id="user-9", username="nonadmin",
            role=Role.USER, api_key_hash="hmac-mock-user-key",
        )

        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        _user_store.get_by_api_key_hash = AsyncMock(return_value=non_admin)
        _user_store.get_by_id = AsyncMock(return_value=non_admin)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.put(
            "/users/user-9/role",
            json={"role": "admin"},
            headers={"Authorization": "Bearer user-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_put_role_success_as_admin(self, mocker):
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})

        admin = _make_user(user_id="admin-1", role=Role.ADMIN)
        target = _make_user(user_id="user-10", username="target",
                            role=Role.USER, api_key_hash="hmac-mock-admin-key")

        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        _user_store.get_by_api_key_hash = AsyncMock(return_value=admin)
        _user_store.get_by_id = AsyncMock(return_value=target)
        _user_store.update_role = AsyncMock(return_value=True)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.put(
            "/users/user-10/role",
            json={"role": "admin"},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"


# ---------------------------------------------------------------------------
# Scoped-admin-token escalation (#2) and initial-password set (#3) fixes
# ---------------------------------------------------------------------------


def _wire_pat_principal(pat, principal):
    """Point the real app stores at a single PAT → principal resolution."""
    from treeloom.application.indexer_state import (
        _user_store,
        _token_store,
        _session_store,
        _api_key_store,
    )
    _user_store.get_by_id = AsyncMock(return_value=principal)
    _user_store.get_by_api_key_hash = AsyncMock(return_value=None)
    _user_store.get_by_username = AsyncMock(return_value=None)
    _token_store.get_by_token_hash = AsyncMock(return_value=pat)
    _token_store.touch_last_used = AsyncMock()
    _token_store.create = AsyncMock(return_value=None)
    _session_store.get_by_token_hash = AsyncMock(return_value=None)
    _session_store.delete_for_user = AsyncMock(return_value=0)
    _api_key_store.get_by_token_hash = AsyncMock(return_value=None)


def _pat(user_id, scopes, token_hash):
    return PersonalAccessToken(
        id="pat-x", user_id=user_id, name="t", token_hash=token_hash,
        scopes=scopes, created_at=datetime.now(timezone.utc),
    )


class TestScopedAdminTokenCannotActOnOthers:
    """A scope-limited PAT owned by an admin must not perform admin-on-others
    actions (the over-privileged-token fix extended to self-or-admin endpoints)."""

    @pytest.mark.asyncio
    async def test_search_only_admin_pat_cannot_create_others_token(self, mocker):
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_service import app

        admin = _make_user(user_id="admin-1", role=Role.ADMIN)
        _wire_pat_principal(_pat("admin-1", ["search"], "hmac-mock-ro-tok"), admin)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/users/victim/tokens",
            json={"name": "x", "scopes": ["search"]},
            headers={"Authorization": "Bearer ro-tok"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_search_only_admin_pat_cannot_reset_others_password(self, mocker):
        _patch_hmac(mocker)
        _patch_password(mocker, verify_result=False)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_service import app

        admin = _make_user(user_id="admin-1", role=Role.ADMIN)
        _wire_pat_principal(_pat("admin-1", ["search"], "hmac-mock-ro-tok"), admin)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/users/victim/set-password",
            json={"password": "pwned"},
            headers={"Authorization": "Bearer ro-tok"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_full_scope_admin_pat_can_act_on_others(self, mocker):
        """An admin PAT minted WITH the admin scope still works on others."""
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _token_store

        admin = _make_user(user_id="admin-1", role=Role.ADMIN)
        _wire_pat_principal(
            _pat("admin-1", ["search", "index", "admin"], "hmac-mock-rw-tok"), admin
        )
        created = PersonalAccessToken(
            id="pat-new", user_id="victim", name="x", token_hash="hmac-mock-new",
            scopes=["search"], created_at=datetime.now(timezone.utc),
        )
        _token_store.create = AsyncMock(return_value=created)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/users/victim/tokens",
            json={"name": "x", "scopes": ["search"]},
            headers={"Authorization": "Bearer rw-tok"},
        )
        assert resp.status_code == 201
        assert resp.json()["token"]  # raw token shown once


class TestInitialPasswordSet:
    """Self-service set-password: first-time set needs no current password,
    but changing an existing one does (#3)."""

    @pytest.mark.asyncio
    async def test_self_initial_set_without_current_password(self, mocker):
        _patch_hmac(mocker)
        # verify would FAIL — but it must not be called for an empty hash.
        _patch_password(mocker, verify_result=False)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        user = _make_user(user_id="u-1", username="bob", role=Role.USER, password_hash="")
        _wire_pat_principal(_pat("u-1", ["search"], "hmac-mock-u-tok"), user)
        _user_store.update_password = AsyncMock(return_value=True)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/users/u-1/set-password",
            json={"password": "firstpw"},
            headers={"Authorization": "Bearer u-tok"},
        )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_self_change_with_existing_password_rejects_wrong_current(self, mocker):
        _patch_hmac(mocker)
        _patch_password(mocker, verify_result=False)  # wrong current password
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _user_store

        user = _make_user(
            user_id="u-1", username="bob", role=Role.USER,
            password_hash="$2b$12$existing",
        )
        _wire_pat_principal(_pat("u-1", ["search"], "hmac-mock-u-tok"), user)
        _user_store.update_password = AsyncMock(return_value=True)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/users/u-1/set-password",
            json={"password": "newpw", "current_password": "wrong"},
            headers={"Authorization": "Bearer u-tok"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Session TTL parsing (#7) and admin bootstrap coexistence (#6)
# ---------------------------------------------------------------------------


class TestSessionTtlParsing:
    """_session_ttl_hours() degrades to the default instead of crashing."""

    def test_valid_value(self, mocker):
        from treeloom.application.routes_auth import _session_ttl_hours
        mocker.patch.dict(os.environ, {"TREELOOM_SESSION_TTL_HOURS": "24"})
        assert _session_ttl_hours() == 24

    def test_malformed_value_falls_back(self, mocker):
        from treeloom.application.routes_auth import _session_ttl_hours
        mocker.patch.dict(os.environ, {"TREELOOM_SESSION_TTL_HOURS": "12h"})
        assert _session_ttl_hours() == 12

    def test_nonpositive_falls_back(self, mocker):
        from treeloom.application.routes_auth import _session_ttl_hours
        mocker.patch.dict(os.environ, {"TREELOOM_SESSION_TTL_HOURS": "0"})
        assert _session_ttl_hours() == 12


class TestAdminBootstrapCoexistence:
    """_seed_admin_if_first_start seeds one admin carrying every configured
    auth method (#6 — key and password no longer race on an elif)."""

    @pytest.mark.asyncio
    async def test_key_and_password_seed_single_admin_with_both(self, mocker):
        _patch_hmac(mocker)
        _patch_password(mocker)
        mocker.patch.dict(os.environ, {
            "TREELOOM_ADMIN_KEY": "k123",
            "TREELOOM_BOOTSTRAP_ADMIN_USER": "root",
            "TREELOOM_BOOTSTRAP_ADMIN_PASSWORD": "pw",
        })
        from treeloom.application.lifecycle import (
            _seed_admin_if_first_start,
        )
        from treeloom.application.indexer_state import (
            _user_store,
        )
        seeded = _make_user(user_id="seed-1", username="root", role=Role.ADMIN)
        _user_store.count_users = AsyncMock(return_value=0)
        _user_store.create_user = AsyncMock(return_value=seeded)
        _user_store.update_password = AsyncMock(return_value=True)

        await _seed_admin_if_first_start()

        # One admin created, with the api-key hash AND a password applied.
        _user_store.create_user.assert_called_once()
        kwargs = _user_store.create_user.call_args.kwargs
        assert kwargs["username"] == "root"
        assert kwargs["api_key_hash"] == "hmac-mock-k123"
        _user_store.update_password.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_users_exist(self, mocker):
        mocker.patch.dict(os.environ, {"TREELOOM_ADMIN_KEY": "k"})
        from treeloom.application.lifecycle import (
            _seed_admin_if_first_start,
        )
        from treeloom.application.indexer_state import (
            _user_store,
        )
        _user_store.count_users = AsyncMock(return_value=3)
        _user_store.create_user = AsyncMock()

        await _seed_admin_if_first_start()
        _user_store.create_user.assert_not_called()


# ── Login throttle: the store owns the record-or-not decision ───────────────
# An earlier version recorded every attempt and refused afterwards, so each
# REJECTED request refreshed the sliding window and the lockout never drained —
# one request per window kept any known username, including the bootstrap
# admin, locked out permanently. The store now counts prior failures and
# records only when the attempt will actually be evaluated, which means the
# handler must hand it both thresholds.
#
# The conditional INSERT itself is SQL behaviour inside a transaction and
# belongs to the integration suite, which this unit suite deliberately
# excludes; what is pinned here is that the store is given what it needs to
# make that decision at all.


class TestThrottleThresholdsArePassedToTheStore:
    @pytest.mark.asyncio
    async def test_handler_passes_both_ceilings(self, mocker):
        _patch_hmac(mocker)
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application import indexer_service as svc
        from treeloom.application.indexer_service import app
        from treeloom.application.indexer_state import _login_attempt_store

        mocker.patch.object(_rauth, "LOGIN_MAX_ATTEMPTS", 5)
        mocker.patch.object(_rauth, "LOGIN_MAX_ATTEMPTS_PER_USERNAME", 50)
        mocker.patch.object(_rauth, "LOGIN_WINDOW_SECONDS", 300)

        arc = AsyncMock(return_value=(0, 0))
        _login_attempt_store.atomic_record_and_count = arc
        _login_attempt_store.record = AsyncMock()
        _login_attempt_store.clear = AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        client.post("/auth/login", json={"username": "u", "password": "p"})

        args = arc.await_args[0]
        assert args[2] == 300, "window must be forwarded"
        assert args[3] == 5, "per-IP ceiling must be forwarded"
        assert args[4] == 50, "per-username ceiling must be forwarded"
