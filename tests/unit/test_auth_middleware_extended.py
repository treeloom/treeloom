"""Detroit-style unit tests for the extended AuthMiddleware.

Tests session cookie, PAT, API-key, and legacy resolution order.
Uses AsyncMock(spec=<Port>) for each store, as the existing
test_auth_middleware.py does for UserStorePort.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from starlette.requests import Request as StarletteRequest

from treeloom.domain.authorization import (
    User, Role, UserStorePort,
    TokenStorePort, ApiKeyStorePort, SessionStorePort,
    PersonalAccessToken, ApiKey, Session,
    scopes_for_role,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_hmac(mocker):
    """Patch hash_api_key to a deterministic mock."""
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
    api_key_hash: str = "hmac-mock-***",
) -> User:
    return User(
        id=user_id,
        username=username,
        email=f"{username}@example.com",
        api_key_hash=api_key_hash,
        role=role,
        active=True,
    )


def _make_session(user_id: str, token_hash: str) -> Session:
    return Session(
        id="sess-001",
        user_id=user_id,
        token_hash=token_hash,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
    )


def _make_pat(user_id: str, token_hash: str, scopes: list[str]) -> PersonalAccessToken:
    return PersonalAccessToken(
        id="pat-001",
        user_id=user_id,
        name="test-pat",
        token_hash=token_hash,
        scopes=scopes,
        created_at=datetime.now(timezone.utc),
    )


def _make_api_key(
    key_id: str,
    name: str,
    token_hash: str,
    scopes: list[str],
    all_access: bool = False,
    created_by: str | None = None,
) -> ApiKey:
    return ApiKey(
        id=key_id,
        name=name,
        token_hash=token_hash,
        scopes=scopes,
        all_access=all_access,
        created_by=created_by,
        created_at=datetime.now(timezone.utc),
    )


def _build_app(
    mocker,
    mock_user_store,
    mock_session_store=None,
    mock_token_store=None,
    mock_api_key_store=None,
    auth_enabled: str = "true",
):
    """Build a minimal FastAPI app with the extended AuthMiddleware."""
    app = FastAPI()
    mocker.patch.dict(os.environ, {"AUTH_ENABLED": auth_enabled})
    from treeloom.application.auth_middleware import AuthMiddleware

    app.add_middleware(
        AuthMiddleware,
        user_store=mock_user_store,
        session_store=mock_session_store,
        token_store=mock_token_store,
        api_key_store=mock_api_key_store,
    )

    @app.get("/protected")
    async def protected(request: StarletteRequest):
        user = getattr(request.state, "user", None)
        scopes = getattr(request.state, "scopes", set())
        return {
            "status": "ok",
            "user": user.username if user else None,
            "role": user.role.value if user else None,
            "all_access": user.all_access if user else None,
            "scopes": sorted(scopes),
        }

    return app


# ---------------------------------------------------------------------------
# Session cookie tests
# ---------------------------------------------------------------------------


class TestSessionCookieAuth:
    """Session cookie is checked first and resolves to user + role scopes."""

    @pytest.mark.asyncio
    async def test_session_cookie_hit_returns_user_and_role_scopes(self, mocker):
        _patch_hmac(mocker)
        user = _make_user(user_id="u-1", username="alice", role=Role.USER)
        session = _make_session("u-1", "hmac-mock-my-session-token")

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=user)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_session_store = AsyncMock(spec=SessionStorePort)
        mock_session_store.get_by_token_hash = AsyncMock(return_value=session)
        mock_session_store.touch = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_session_store=mock_session_store)
        client = TestClient(app)

        resp = client.get("/protected", cookies={"treeloom_session": "my-session-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "alice"
        # User role scopes: search + index
        assert "search" in data["scopes"]
        assert "index" in data["scopes"]
        assert "admin" not in data["scopes"]

    @pytest.mark.asyncio
    async def test_expired_session_falls_through_to_401(self, mocker):
        """An expired/revoked session cookie → store returns None → 401."""
        _patch_hmac(mocker)

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_session_store = AsyncMock(spec=SessionStorePort)
        mock_session_store.get_by_token_hash = AsyncMock(return_value=None)

        app = _build_app(mocker, mock_user_store, mock_session_store=mock_session_store)
        client = TestClient(app)

        resp = client.get(
            "/protected", cookies={"treeloom_session": "expired-token"}
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_session_touch_called(self, mocker):
        """Session touch is called on a valid cookie."""
        _patch_hmac(mocker)
        user = _make_user(user_id="u-2", username="bob")
        session = _make_session("u-2", "hmac-mock-cookie-val")

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=user)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_session_store = AsyncMock(spec=SessionStorePort)
        mock_session_store.get_by_token_hash = AsyncMock(return_value=session)
        mock_session_store.touch = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_session_store=mock_session_store)
        client = TestClient(app)
        client.get("/protected", cookies={"treeloom_session": "cookie-val"})

        mock_session_store.touch.assert_called_once_with("hmac-mock-cookie-val")


# ---------------------------------------------------------------------------
# PAT (Personal Access Token) tests
# ---------------------------------------------------------------------------


class TestPATAuth:
    """PAT bearer resolution attaches owner + token's scopes."""

    @pytest.mark.asyncio
    async def test_pat_bearer_resolves_token_scopes(self, mocker):
        """PAT hit: returned scopes are the token's scopes, not role scopes."""
        _patch_hmac(mocker)
        user = _make_user(user_id="u-3", username="charlie", role=Role.ADMIN)
        pat = _make_pat("u-3", "hmac-mock-pat-raw", scopes=["search"])

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=user)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=pat)
        mock_token_store.touch_last_used = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_token_store=mock_token_store)
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer pat-raw"})
        assert resp.status_code == 200
        data = resp.json()
        # Token scopes: search only (even though owner is admin)
        assert data["scopes"] == ["search"]

    @pytest.mark.asyncio
    async def test_revoked_pat_falls_through_to_legacy(self, mocker):
        """Revoked/expired PAT (store returns None) → falls through to legacy."""
        _patch_hmac(mocker)
        legacy_user = _make_user(
            user_id="u-4", username="dana", api_key_hash="hmac-mock-token-val"
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=legacy_user)
        mock_user_store.get_by_id = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=None)  # miss

        app = _build_app(mocker, mock_user_store, mock_token_store=mock_token_store)
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer token-val"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "dana"

    @pytest.mark.asyncio
    async def test_pat_touch_called(self, mocker):
        """PAT touch_last_used is called on hit."""
        _patch_hmac(mocker)
        user = _make_user(user_id="u-5", username="eve")
        pat = _make_pat("u-5", "hmac-mock-pat-ev", scopes=["search"])

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=user)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=pat)
        mock_token_store.touch_last_used = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_token_store=mock_token_store)
        client = TestClient(app)
        client.get("/protected", headers={"Authorization": "Bearer pat-ev"})
        mock_token_store.touch_last_used.assert_called_once_with("pat-001")


# ---------------------------------------------------------------------------
# API key tests
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """API key resolution attaches principal + key scopes."""

    @pytest.mark.asyncio
    async def test_api_key_with_owner_resolves_owner(self, mocker):
        """API key with created_by → load real owner user."""
        _patch_hmac(mocker)
        owner = _make_user(user_id="u-6", username="frank", role=Role.USER)
        api_key = _make_api_key(
            "key-1", "ci-key", "hmac-mock-ci-token",
            scopes=["index"],
            created_by="u-6",
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=owner)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=None)

        mock_api_key_store = AsyncMock(spec=ApiKeyStorePort)
        mock_api_key_store.get_by_token_hash = AsyncMock(return_value=api_key)
        mock_api_key_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_token_store=mock_token_store,
            mock_api_key_store=mock_api_key_store,
        )
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer ci-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "frank"
        assert data["scopes"] == ["index"]

    @pytest.mark.asyncio
    async def test_api_key_without_owner_synthetic_principal(self, mocker):
        """API key without created_by → synthetic service principal."""
        _patch_hmac(mocker)
        api_key = _make_api_key(
            "key-2", "svc-key", "hmac-mock-svc-token",
            scopes=["search", "index"],
            all_access=True,
            created_by=None,
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=None)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=None)

        mock_api_key_store = AsyncMock(spec=ApiKeyStorePort)
        mock_api_key_store.get_by_token_hash = AsyncMock(return_value=api_key)
        mock_api_key_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_token_store=mock_token_store,
            mock_api_key_store=mock_api_key_store,
        )
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer svc-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "svc-key"
        assert sorted(data["scopes"]) == ["index", "search"]


# ---------------------------------------------------------------------------
# token data-access privilege follows the token's SCOPES, not the
# owner's role. An admin-owned search-only key must resolve to role USER.
# ---------------------------------------------------------------------------


class TestScopedKeyRoleDerivation:
    @pytest.mark.asyncio
    async def test_admin_owned_search_key_resolves_user_role(self, mocker):
        """Admin-owned key with scopes=['search'] → resolved role USER."""
        _patch_hmac(mocker)
        owner = _make_user(user_id="adm-1", username="adminowner", role=Role.ADMIN)
        api_key = _make_api_key(
            "key-s", "search-key", "hmac-mock-search-token",
            scopes=["search"],
            created_by="adm-1",
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=owner)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=None)

        mock_api_key_store = AsyncMock(spec=ApiKeyStorePort)
        mock_api_key_store.get_by_token_hash = AsyncMock(return_value=api_key)
        mock_api_key_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_token_store=mock_token_store,
            mock_api_key_store=mock_api_key_store,
        )
        client = TestClient(app)

        resp = client.get(
            "/protected", headers={"Authorization": "Bearer search-token"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "adminowner"
        assert data["role"] == "user"  # derived from scopes, NOT owner's admin
        assert data["scopes"] == ["search"]
        # The cached owner object must NOT be mutated.
        assert owner.role == Role.ADMIN

    @pytest.mark.asyncio
    async def test_admin_scoped_key_keeps_admin_role(self, mocker):
        """Admin-owned key WITH the admin scope → resolved role ADMIN."""
        _patch_hmac(mocker)
        owner = _make_user(user_id="adm-2", username="adminowner2", role=Role.ADMIN)
        api_key = _make_api_key(
            "key-a", "admin-key", "hmac-mock-admin-token",
            scopes=["admin", "search"],
            created_by="adm-2",
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=owner)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=None)

        mock_api_key_store = AsyncMock(spec=ApiKeyStorePort)
        mock_api_key_store.get_by_token_hash = AsyncMock(return_value=api_key)
        mock_api_key_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_token_store=mock_token_store,
            mock_api_key_store=mock_api_key_store,
        )
        client = TestClient(app)

        resp = client.get(
            "/protected", headers={"Authorization": "Bearer admin-token"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "admin"

    @pytest.mark.asyncio
    async def test_search_key_preserves_owner_all_access(self, mocker):
        """Derived non-admin principal keeps the owner's all_access flag."""
        _patch_hmac(mocker)
        owner = _make_user(user_id="adm-3", username="adminowner3", role=Role.ADMIN)
        owner.all_access = True
        api_key = _make_api_key(
            "key-aa", "search-aa", "hmac-mock-aa-token",
            scopes=["search"],
            created_by="adm-3",
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=owner)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=None)

        mock_api_key_store = AsyncMock(spec=ApiKeyStorePort)
        mock_api_key_store.get_by_token_hash = AsyncMock(return_value=api_key)
        mock_api_key_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_token_store=mock_token_store,
            mock_api_key_store=mock_api_key_store,
        )
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer aa-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "user"
        assert data["all_access"] is True

    @pytest.mark.asyncio
    async def test_admin_owned_search_pat_resolves_user_role(self, mocker):
        """Admin-owned PAT with scopes=['search'] → resolved role USER too."""
        _patch_hmac(mocker)
        owner = _make_user(user_id="adm-4", username="patadmin", role=Role.ADMIN)
        pat = _make_pat("adm-4", "hmac-mock-pat-token", scopes=["search"])

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=owner)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=pat)
        mock_token_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_token_store=mock_token_store,
        )
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer pat-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "patadmin"
        assert data["role"] == "user"
        assert owner.role == Role.ADMIN  # owner not mutated


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    """Legacy api_key_hash still works when no PAT/session matches."""

    @pytest.mark.asyncio
    async def test_legacy_key_still_authenticates(self, mocker):
        _patch_hmac(mocker)
        user = _make_user(user_id="u-7", username="grace", api_key_hash="hmac-mock-old-key")

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=user)
        mock_user_store.get_by_id = AsyncMock(return_value=None)

        # No PAT or session stores
        app = _build_app(mocker, mock_user_store)
        client = TestClient(app)

        resp = client.get("/protected", headers={"Authorization": "Bearer old-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "grace"
        # Legacy gets role-derived scopes
        assert "search" in data["scopes"]


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    """Session cookie takes precedence over bearer token."""

    @pytest.mark.asyncio
    async def test_session_cookie_wins_over_bearer_pat(self, mocker):
        """When both session cookie and PAT are present, session cookie wins."""
        _patch_hmac(mocker)
        session_user = _make_user(user_id="u-sess", username="session-user")
        pat_user = _make_user(user_id="u-pat", username="pat-user")

        session = _make_session("u-sess", "hmac-mock-session-val")
        pat = _make_pat("u-pat", "hmac-mock-bearer-val", scopes=["search"])

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(
            side_effect=lambda uid: session_user if uid == "u-sess" else pat_user
        )
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_session_store = AsyncMock(spec=SessionStorePort)
        mock_session_store.get_by_token_hash = AsyncMock(return_value=session)
        mock_session_store.touch = AsyncMock()

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=pat)
        mock_token_store.touch_last_used = AsyncMock()

        app = _build_app(
            mocker, mock_user_store,
            mock_session_store=mock_session_store,
            mock_token_store=mock_token_store,
        )
        client = TestClient(app)

        resp = client.get(
            "/protected",
            cookies={"treeloom_session": "session-val"},
            headers={"Authorization": "Bearer bearer-val"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"] == "session-user"


# ---------------------------------------------------------------------------
# Public paths
# ---------------------------------------------------------------------------


class TestPublicPaths:
    """/auth/login and /auth/logout are public (no auth required)."""

    @pytest.mark.asyncio
    async def test_auth_login_is_public(self, mocker):
        _patch_hmac(mocker)
        mock_user_store = AsyncMock(spec=UserStorePort)
        app = FastAPI()
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.auth_middleware import AuthMiddleware

        app.add_middleware(AuthMiddleware, user_store=mock_user_store)

        @app.post("/auth/login")
        async def login():
            return {"ok": True}

        client = TestClient(app)
        resp = client.post("/auth/login", json={"username": "a", "password": "b"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_logout_is_public(self, mocker):
        _patch_hmac(mocker)
        mock_user_store = AsyncMock(spec=UserStorePort)
        app = FastAPI()
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.auth_middleware import AuthMiddleware

        app.add_middleware(AuthMiddleware, user_store=mock_user_store)

        @app.post("/auth/logout")
        async def logout():
            return {"ok": True}

        client = TestClient(app)
        resp = client.post("/auth/logout")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Deactivated-principal rejection (#1 security fix)
# ---------------------------------------------------------------------------


class TestDeactivatedPrincipalRejected:
    """A soft-deleted user (active=False) must lose access via every auth
    path that resolves the principal with get_by_id (session, PAT, owner-bound
    API key) — get_by_id does not filter on active, so the middleware must."""

    @pytest.mark.asyncio
    async def test_inactive_user_session_rejected(self, mocker):
        _patch_hmac(mocker)
        user = _make_user(user_id="u-x", username="fired")
        user.active = False
        session = _make_session("u-x", "hmac-mock-stale-session")

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=user)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_session_store = AsyncMock(spec=SessionStorePort)
        mock_session_store.get_by_token_hash = AsyncMock(return_value=session)
        mock_session_store.touch = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_session_store=mock_session_store)
        client = TestClient(app)
        resp = client.get("/protected", cookies={"treeloom_session": "stale-session"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_inactive_user_pat_rejected(self, mocker):
        _patch_hmac(mocker)
        user = _make_user(user_id="u-y", username="fired")
        user.active = False
        pat = _make_pat("u-y", "hmac-mock-pat-tok", ["search", "index"])

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=user)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_token_store = AsyncMock(spec=TokenStorePort)
        mock_token_store.get_by_token_hash = AsyncMock(return_value=pat)
        mock_token_store.touch_last_used = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_token_store=mock_token_store)
        client = TestClient(app)
        resp = client.get(
            "/protected", headers={"Authorization": "Bearer pat-tok"}
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_owner_bound_api_key_with_inactive_owner_rejected(self, mocker):
        """An owner-bound key whose creator is deactivated must NOT fall back
        to a synthetic admin principal — it is rejected outright."""
        _patch_hmac(mocker)
        owner = _make_user(user_id="admin-z", username="exadmin", role=Role.ADMIN)
        owner.active = False
        key = _make_api_key(
            "k-1", "ci-key", "hmac-mock-ci-tok", ["admin"], created_by="admin-z"
        )

        mock_user_store = AsyncMock(spec=UserStorePort)
        mock_user_store.get_by_id = AsyncMock(return_value=owner)
        mock_user_store.get_by_api_key_hash = AsyncMock(return_value=None)

        mock_api_key_store = AsyncMock(spec=ApiKeyStorePort)
        mock_api_key_store.get_by_token_hash = AsyncMock(return_value=key)
        mock_api_key_store.touch_last_used = AsyncMock()

        app = _build_app(mocker, mock_user_store, mock_api_key_store=mock_api_key_store)
        client = TestClient(app)
        resp = client.get(
            "/protected", headers={"Authorization": "Bearer ci-tok"}
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_dashboard_is_public(self, mocker):
        """The static dashboard shell must load without auth so a logged-out
        user can reach the login form (data calls it makes are authed)."""
        _patch_hmac(mocker)
        mock_user_store = AsyncMock(spec=UserStorePort)
        app = FastAPI()
        mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})
        from treeloom.application.auth_middleware import AuthMiddleware

        app.add_middleware(AuthMiddleware, user_store=mock_user_store)

        @app.get("/dashboard")
        async def dashboard():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/dashboard")  # no Authorization header / cookie
        assert resp.status_code == 200
