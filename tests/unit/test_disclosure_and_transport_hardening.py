"""Five smaller SAST findings, grouped by what they leak or let through."""

import pytest
from treeloom.application import routes_auth as _rauth
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


class TestCorsWildcardInExplicitList:
    """finding 34 CWE-942. A '*' inside the EXPLICIT list took the credentialed
    branch, pairing allow_origins=['*'] with allow_credentials=True.
    Starlette then echoes the requesting origin and sets
    Access-Control-Allow-Credentials: true — so every site on the web gets
    credentialed cross-origin access with the victim's session cookie. That
    is exactly what the spec stops the literal wildcard from doing,
    reintroduced via the echo path."""

    def _origins_and_credentials(self, raw):
        from fastapi import FastAPI

        from treeloom.application.cors import add_cors_middleware

        app = FastAPI()
        configured = add_cors_middleware(app, raw)
        mw = [m for m in app.user_middleware if "CORS" in str(m.cls)][0]
        return configured, mw.kwargs

    def test_wildcard_never_pairs_with_credentials(self):
        _, kwargs = self._origins_and_credentials("https://ui.example,*")
        assert kwargs["allow_credentials"] is False

    def test_wildcard_downgrades_rather_than_honouring_the_list(self):
        """The operator asked for 'anyone'; the only safe reading is anyone
        WITHOUT credentials, not these-plus-anyone with them."""
        configured, kwargs = self._origins_and_credentials("https://ui.example,*")
        assert configured == []
        assert kwargs["allow_origins"] == ["*"]

    def test_an_ordinary_explicit_list_still_gets_credentials(self):
        configured, kwargs = self._origins_and_credentials("https://ui.example")
        assert configured == ["https://ui.example"]
        assert kwargs["allow_credentials"] is True
        assert kwargs["allow_origins"] == ["https://ui.example"]

    def test_unset_is_the_permissive_credential_less_default(self):
        _, kwargs = self._origins_and_credentials(None)
        assert kwargs["allow_origins"] == ["*"]
        assert kwargs["allow_credentials"] is False


class TestWebhookSecretComparison:
    """finding 30 CWE-208. `!=` on str short-circuits at the first differing byte, so
    response timing is a prefix-match oracle and the shared secret can be
    recovered byte by byte.

    The finding is about how a webhook secret is compared, and every adapter
    compares through `domain.webhook.validate_hmac`, so the guard targets that
    shared helper rather than any one adapter.
    """

    def test_validate_hmac_uses_a_constant_time_comparison(self):
        import inspect

        from treeloom.domain import webhook

        src = inspect.getsource(webhook.validate_hmac)
        assert "compare_digest" in src
        assert "expected ==" not in src and "== sig_value" not in src

    def test_a_wrong_signature_is_refused(self):
        import hashlib
        import hmac as _hmac

        from treeloom.domain.webhook import validate_hmac

        body = b'{"action":"closed"}'
        good = _hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        assert validate_hmac("s3cret", body, good) is True
        assert validate_hmac("s3cret", body, "0" * len(good)) is False
        assert validate_hmac("s3cret", body, "") is False

    def test_an_empty_secret_does_not_accept_a_supplied_signature(self):
        """Unconfigured must not mean "anything passes"."""
        from treeloom.domain.webhook import validate_hmac

        assert validate_hmac("", b"body", "deadbeef") is False


class TestNoTracebackInResponseBody:
    """finding 29 CWE-209. The MCP search handler returned traceback.format_exc() to
    the caller: absolute source paths, the dependency tree and its versions,
    and frame locals — on an unauthenticated transport."""

    def test_handler_does_not_serialize_a_traceback(self):
        import inspect

        from treeloom.application import main

        src = inspect.getsource(main)
        assert "format_exc" not in src
        assert '"traceback"' not in src


class TestLogoutReportsAFailedRevocation:
    """finding 41 CWE-613. Swallowing the delete returned 204 while the session row
    stayed valid: the cookie vanished, the user believed they were logged
    out, and anyone holding that token could keep using it until TTL."""

    @pytest.mark.asyncio
    async def test_failed_server_side_delete_is_not_a_success(self, mocker):
        mocker.patch.object(
            _state._session_store, "delete",
            new=mocker.AsyncMock(side_effect=RuntimeError("pg down")),
        )
        request = mocker.Mock()
        request.cookies = {"treeloom_session": "tok"}
        response = mocker.Mock()

        with pytest.raises(HTTPException) as exc:
            await _rauth.auth_logout(request, response)

        assert exc.value.status_code == 503
        assert "NOT be revoked" in exc.value.detail
        response.delete_cookie.assert_called_once_with("treeloom_session"), (
            "the browser in front of us should still stop sending the token"
        )

    @pytest.mark.asyncio
    async def test_successful_logout_is_unchanged(self, mocker):
        mocker.patch.object(
            _state._session_store, "delete", new=mocker.AsyncMock(return_value=None)
        )
        request = mocker.Mock()
        request.cookies = {"treeloom_session": "tok"}
        response = mocker.Mock()
        assert await _rauth.auth_logout(request, response) is None
        response.delete_cookie.assert_called_once()

    @pytest.mark.asyncio
    async def test_logout_without_a_cookie_still_clears(self, mocker):
        request = mocker.Mock()
        request.cookies = {}
        response = mocker.Mock()
        assert await _rauth.auth_logout(request, response) is None


class TestReturnPoolIsAnOperatorTool:
    """finding 35 CWE-284. return_pool skips rerank, graph AND top_k, returning the
    whole pre-rerank pool of raw bodies. The ACL prefilter still applies, so
    it is not a confidentiality bypass — but it is a bulk-extraction
    accelerator available to every caller."""

    @pytest.mark.asyncio
    async def test_non_admin_is_refused(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"search"}
        authorize = mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        req = svc.SearchRequest(query="q", cross_repo=True, return_pool=True)
        with pytest.raises(HTTPException) as exc:
            await svc.handle_search(req, request)
        assert exc.value.status_code == 403
        authorize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_ordinary_search_is_unaffected(self, mocker, monkeypatch):
        """The gate must not touch the normal path."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"search"}
        mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        mocker.patch.object(
            svc, "embed_query", new=mocker.AsyncMock(return_value=[[0.0]])
        )
        graph = mocker.patch.object(
            svc, "graph_search", new=mocker.AsyncMock(return_value={"chunks": []})
        )
        req = svc.SearchRequest(query="q", cross_repo=True)
        await svc.handle_search(req, request)
        graph.assert_awaited_once()
