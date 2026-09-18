"""Unit tests for the MCP transport Origin guard (DNS-rebinding defence).

Detroit-style: the pure predicate is tested directly, and the ASGI wrapper is
exercised through a real Starlette app so the 403 short-circuit and the
delegate-untouched path are both covered without mocks.
"""

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from treeloom.application.mcp_origin_guard import (
    OriginGuardMiddleware,
    is_loopback_origin,
    is_origin_allowed,
    parse_allowed_origins,
)


class TestParseAllowedOrigins:
    def test_none_and_empty_yield_no_origins(self):
        assert parse_allowed_origins(None) == []
        assert parse_allowed_origins("") == []
        assert parse_allowed_origins("   ") == []

    def test_splits_and_strips(self):
        raw = " https://a.example , https://b.example ,, "
        assert parse_allowed_origins(raw) == ["https://a.example", "https://b.example"]


class TestIsLoopbackOrigin:
    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost",
            "http://localhost:8000",
            "http://127.0.0.1:3001",
            "https://127.0.0.1",
            "http://[::1]:8000",
            "http://LOCALHOST:8000",
        ],
    )
    def test_loopback_forms_accepted(self, origin):
        assert is_loopback_origin(origin) is True

    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.example",
            "https://localhost.evil.example",  # suffix trick
            "http://127.0.0.1.evil.example",   # prefix trick
            "null",                            # sandboxed iframe
            "",
            "not-a-url",
            "http://192.0.2.10:8000",
        ],
    )
    def test_non_loopback_rejected(self, origin):
        assert is_loopback_origin(origin) is False


class TestIsOriginAllowed:
    def test_absent_origin_is_allowed(self):
        """Non-browser MCP clients send no Origin; the spec only requires
        rejecting one that is present and invalid."""
        assert is_origin_allowed(None, []) is True
        assert is_origin_allowed(None, ["https://a.example"]) is True

    def test_default_is_loopback_only(self):
        assert is_origin_allowed("http://localhost:8000", []) is True
        assert is_origin_allowed("http://evil.example", []) is False

    def test_explicit_list_is_exact_match(self):
        allowed = ["https://ui.example"]
        assert is_origin_allowed("https://ui.example", allowed) is True
        assert is_origin_allowed("https://ui.example/", allowed) is False
        assert is_origin_allowed("https://other.example", allowed) is False

    def test_explicit_list_replaces_the_loopback_default(self):
        """Configuring origins is opt-in precision: loopback is no longer
        implicitly trusted once an explicit list is supplied."""
        assert is_origin_allowed("http://localhost:8000", ["https://ui.example"]) is False


def _app(allowed):
    inner = Starlette(routes=[Route("/sse", lambda r: PlainTextResponse("ok"))])
    return OriginGuardMiddleware(inner, allowed_origins=allowed)


class TestOriginGuardMiddleware:
    def test_disallowed_origin_gets_403_and_never_reaches_the_app(self):
        client = TestClient(_app([]), raise_server_exceptions=False)
        resp = client.get("/sse", headers={"Origin": "http://evil.example"})
        assert resp.status_code == 403
        assert resp.json()["error"]["message"] == "Origin not allowed"
        assert "ok" not in resp.text

    def test_loopback_origin_passes_through(self):
        client = TestClient(_app([]))
        resp = client.get("/sse", headers={"Origin": "http://localhost:8000"})
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_missing_origin_passes_through(self):
        client = TestClient(_app([]))
        resp = client.get("/sse")
        assert resp.status_code == 200

    def test_configured_origin_passes_and_others_do_not(self):
        client = TestClient(_app(["https://ui.example"]), raise_server_exceptions=False)
        assert client.get("/sse", headers={"Origin": "https://ui.example"}).status_code == 200
        assert client.get("/sse", headers={"Origin": "http://localhost:8000"}).status_code == 403
