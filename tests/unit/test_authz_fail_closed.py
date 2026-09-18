"""Authorization and revocation must fail CLOSED, not quietly open.

Two SAST findings that share a shape: a store swallowed a database error into
a benign-looking return value, and the caller could not tell that value apart
from a real answer.

* finding 37 (CWE-636) — GrantStore.effects_for_source / denied_sources returned an
  empty result on failure. Empty does not read as "deny"; it reads as "this
  principal has no deny grants". For a caller with all_access the deny grant
  is the ONLY thing withholding the source, so a database blip granted the
  access the grant existed to prevent.

* finding 16 (CWE-636) — revoke() returned False on failure, the endpoint turned
  False into 404 "not found", and an operator revoking a compromised
  credential reads that as "already gone". The credential keeps working.
"""

import pytest
from treeloom.application import routes_auth as _rauth
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.adapters.authorization.api_key_store import RevocationUnavailable
from treeloom.adapters.authorization.grant_store import GrantLookupUnavailable
from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


class TestAuthorizeScopeRefusesOnStoreFailure:
    @pytest.fixture
    def authed(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        user = mocker.Mock(id="u1", all_access=True)
        mocker.patch.object(
            _authz, "_require_authenticated_user", new=mocker.AsyncMock(return_value=user)
        )
        return user

    @pytest.mark.asyncio
    async def test_single_source_lookup_failure_is_503_not_allow(
        self, mocker, authed
    ):
        """The decisive case: all_access + a deny grant that cannot be read."""
        authz = mocker.Mock()
        authz.authorize_source = mocker.AsyncMock(
            side_effect=GrantLookupUnavailable("pg down")
        )
        mocker.patch.object(_authz, "_authz", return_value=authz)
        with pytest.raises(HTTPException) as exc:
            await _authz._authorize_scope(mocker.Mock(), "secret-repo")
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_shared_query_lookup_failure_is_503_not_unfiltered(
        self, mocker, authed
    ):
        """A shared query returns the EXCLUSION list. Failing open here does
        not just admit one source — it drops every exclusion at once."""
        authz = mocker.Mock()
        authz.authorize_shared = mocker.AsyncMock(
            side_effect=GrantLookupUnavailable("pg down")
        )
        mocker.patch.object(_authz, "_authz", return_value=authz)
        with pytest.raises(HTTPException) as exc:
            await _authz._authorize_scope(mocker.Mock(), None)
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_a_genuine_deny_is_still_403_not_503(self, mocker, authed):
        """503 must mean "could not decide", not blur into "decided no" —
        otherwise the new path hides real denials behind an outage-looking
        status and nobody investigates."""
        authz = mocker.Mock()
        authz.authorize_source = mocker.AsyncMock(
            return_value=mocker.Mock(allowed=False, excluded_source_ids=[])
        )
        mocker.patch.object(_authz, "_authz", return_value=authz)
        mocker.patch.object(_authz, "_audit_search", new=mocker.AsyncMock())
        with pytest.raises(HTTPException) as exc:
            await _authz._authorize_scope(mocker.Mock(), "s1")
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_a_normal_allow_still_returns_exclusions(self, mocker, authed):
        authz = mocker.Mock()
        authz.authorize_shared = mocker.AsyncMock(
            return_value=mocker.Mock(allowed=True, excluded_source_ids=["denied-1"])
        )
        mocker.patch.object(_authz, "_authz", return_value=authz)
        mocker.patch.object(_authz, "_audit_search", new=mocker.AsyncMock())
        assert await _authz._authorize_scope(mocker.Mock(), None) == ["denied-1"]

    @pytest.mark.asyncio
    async def test_enforcement_off_is_untouched(self, mocker, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        assert await _authz._authorize_scope(mocker.Mock(), "s1") is None


class TestRevocationReportsFailureAsFailure:
    @pytest.mark.asyncio
    async def test_api_key_store_failure_is_503_not_404(self, mocker):
        mocker.patch.object(_authz, "_require_admin")
        mocker.patch.object(
            _state._api_key_store, "revoke",
            new=mocker.AsyncMock(side_effect=RevocationUnavailable("pg down")),
        )
        with pytest.raises(HTTPException) as exc:
            await _rauth.revoke_api_key("key-1", mocker.Mock())
        assert exc.value.status_code == 503
        assert "still ACTIVE" in exc.value.detail, (
            "the operator has to know the credential was NOT revoked"
        )

    @pytest.mark.asyncio
    async def test_a_genuinely_missing_key_is_still_404(self, mocker):
        """The 404 has to keep working, or the fix just trades one wrong
        answer for another."""
        mocker.patch.object(_authz, "_require_admin")
        mocker.patch.object(
            _state._api_key_store, "revoke", new=mocker.AsyncMock(return_value=False)
        )
        with pytest.raises(HTTPException) as exc:
            await _rauth.revoke_api_key("nope", mocker.Mock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_successful_revocation_is_unchanged(self, mocker):
        mocker.patch.object(_authz, "_require_admin")
        mocker.patch.object(
            _state._api_key_store, "revoke", new=mocker.AsyncMock(return_value=True)
        )
        assert await _rauth.revoke_api_key("key-1", mocker.Mock()) is None


class TestStoresRaiseRatherThanReturnFalse:
    @pytest.mark.asyncio
    async def test_api_key_store_without_pool(self):
        from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore

        with pytest.raises(RevocationUnavailable):
            await PostgreSQLApiKeyStore(pool=None).revoke("k1")

    @pytest.mark.asyncio
    async def test_token_store_without_pool(self):
        from treeloom.adapters.authorization.token_store import (
            PostgreSQLPersonalAccessTokenStore,
        )

        with pytest.raises(RevocationUnavailable):
            await PostgreSQLPersonalAccessTokenStore(pool=None).revoke("t1", "u1")
