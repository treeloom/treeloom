"""Regression: `_require_authenticated_user` (used by skip-route handlers like
GET /sources) must resolve the `treeloom_session` cookie, not only a Bearer
token — otherwise a cookie-logged-in caller (operator UI v2 password login)
gets a spurious 401 on the Sources tab.
"""

import types

import pytest
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


def _fake_request(cookies=None, headers=None):
    # The helper only touches request.state.user (getattr), request.cookies.get,
    # and request.headers.get — a SimpleNamespace + dicts suffice.
    return types.SimpleNamespace(
        state=types.SimpleNamespace(),
        cookies=cookies or {},
        headers=headers or {},
    )


@pytest.mark.asyncio
async def test_session_cookie_authenticates(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    user = types.SimpleNamespace(id="u1", active=True, role=None)

    class FakeSessions:
        async def get_by_token_hash(self, _hash):
            return types.SimpleNamespace(user_id="u1")

    class FakeUsers:
        async def get_by_id(self, uid):
            return user if uid == "u1" else None

    monkeypatch.setattr(_state, "_session_store", FakeSessions())
    monkeypatch.setattr(_state, "_user_store", FakeUsers())

    req = _fake_request(cookies={"treeloom_session": "raw-token"})
    assert await _authz._require_authenticated_user(req) is user


@pytest.mark.asyncio
async def test_inactive_user_with_session_is_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")

    class FakeSessions:
        async def get_by_token_hash(self, _hash):
            return types.SimpleNamespace(user_id="u1")

    class FakeUsers:
        async def get_by_id(self, _uid):
            return types.SimpleNamespace(id="u1", active=False, role=None)

    monkeypatch.setattr(_state, "_session_store", FakeSessions())
    monkeypatch.setattr(_state, "_user_store", FakeUsers())

    req = _fake_request(cookies={"treeloom_session": "raw-token"})
    # Deactivated account → no cookie match → falls through to bearer → 401.
    with pytest.raises(HTTPException) as ei:
        await _authz._require_authenticated_user(req)
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_no_cookie_no_bearer_raises_401(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setattr(_state, "_session_store", None)
    with pytest.raises(HTTPException) as ei:
        await _authz._require_authenticated_user(_fake_request())
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    assert await _authz._require_authenticated_user(_fake_request()) is None
