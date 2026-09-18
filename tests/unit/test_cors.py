"""Unit tests for indexer CORS configuration.

Hermetic: builds throwaway FastAPI apps and drives the middleware with
Starlette's TestClient. No database, no indexer app import.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from treeloom.application.cors import add_cors_middleware, parse_cors_origins


def test_parse_cors_origins_empty_is_disabled():
    assert parse_cors_origins(None) == []
    assert parse_cors_origins("") == []
    assert parse_cors_origins("  ") == []


def test_parse_cors_origins_single_and_multi_with_whitespace():
    assert parse_cors_origins("https://ui.test") == ["https://ui.test"]
    assert parse_cors_origins(" https://a.test , https://b.test ,,https://c.test") == [
        "https://a.test",
        "https://b.test",
        "https://c.test",
    ]


def _app_with(raw_origins):
    app = FastAPI()
    add_cors_middleware(app, raw_origins)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return TestClient(app)


def test_explicit_origin_allows_credentialed_cross_origin():
    client = _app_with("https://ui.test")
    # Preflight from the configured origin.
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://ui.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://ui.test"
    # Explicit origin => credentials are actually usable (cookie path works).
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_unlisted_origin_is_not_reflected():
    client = _app_with("https://ui.test")
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://evil.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette does not echo an unlisted origin.
    assert resp.headers.get("access-control-allow-origin") != "https://evil.test"


def test_permissive_default_does_not_combine_wildcard_with_credentials():
    client = _app_with(None)  # empty => permissive dev default
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://anything.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "*"
    # Wildcard must NOT be paired with allow-credentials (spec violation that
    # the old hardcoded config had).
    assert resp.headers.get("access-control-allow-credentials") is None
