"""The routing table itself — that each path reaches the handler written for it.

`GET /jobs` shipped to main bound to `_source_id_for_job`, a private helper
that takes a `job_id` argument, so the endpoint answered 422 "missing query
parameter" and `list_jobs` was never registered at all.

The cause was textual: a helper was inserted using `async def list_jobs(...)`
as the anchor, and `@app.get("/jobs")` sits on the line above it, so the new
function landed between the decorator and its handler and inherited the route.

Nothing caught it. Every /jobs test calls `list_jobs(...)` directly rather
than over HTTP, so none of them touch the routing table; the route COUNT was
unchanged, because a decorator was still registered — just on the wrong
function — so counting routes verified the thing that could not move instead
of the thing that had. A code review and a SAST re-scan both passed over it.

These tests assert the binding, which is the property that was actually lost.
"""

import ast
import pathlib

import pytest
from fastapi.testclient import TestClient

from treeloom.application.indexer_service import app


def _walk_routes(routes):
    """Yield every concrete (method-bearing) route, descending into routers.

    FastAPI 0.14x stopped flattening ``include_router`` into ``app.routes``:
    an included router is kept as a single ``_IncludedRouter`` node whose
    ``original_router.routes`` holds the real ``APIRoute``s. Older versions
    flattened them. Walking both shapes keeps this test meaningful on either.
    """
    for r in routes:
        if hasattr(r, "methods"):
            yield r
        inner = getattr(r, "original_router", None) or (
            r if not hasattr(r, "methods") and hasattr(r, "routes") else None
        )
        if inner is not None:
            yield from _walk_routes(inner.routes)


def _routes():
    return list(_walk_routes(app.routes))


class TestNoRouteIsBoundToAPrivateHelper:
    """The generalisation of the bug: a leading underscore means the function
    was never meant to be reachable over HTTP, so a route on one is an
    orphaned decorator that slid onto the next definition."""

    @pytest.mark.parametrize(
        "module",
        ["indexer_service.py", "main.py"],
    )
    def test_no_app_decorator_on_a_private_function(self, module):
        tree = ast.parse(
            (pathlib.Path("src/treeloom/application") / module).read_text()
        )
        orphans = []
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for d in n.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if (isinstance(f, ast.Attribute)
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "app"):
                    path = (d.args[0].value
                            if isinstance(d, ast.Call) and d.args else "?")
                    if n.name.startswith("_"):
                        orphans.append(
                            f"{f.attr.upper()} {path} -> {n.name}() at line {n.lineno}"
                        )
        assert orphans == [], (
            "route decorator landed on a private helper:\n  " + "\n  ".join(orphans)
        )


class TestKnownRoutesReachTheirHandlers:
    """Spot-checks on endpoints whose handler name is load-bearing. A rename
    is a deliberate act; an orphaned decorator is not, and looks identical
    from a route count."""

    EXPECTED = {
        ("/jobs", "GET"): "list_jobs",
        ("/job-groups", "GET"): "list_job_groups",
        ("/jobs/{job_id}", "GET"): "get_job",
        ("/jobs/{job_id}/errors", "GET"): "get_job_errors",
        ("/job-groups/{group_id}", "GET"): "get_job_group",
        ("/status", "GET"): "status",
        ("/health", "GET"): "health",
        ("/search", "POST"): "handle_search",
        ("/sources", "GET"): "get_sources",
        ("/sources/{source_id}", "DELETE"): "remove_source",
        ("/auth/login", "POST"): "auth_login",
        ("/index-graph", "POST"): "handle_index_graph",
        ("/build-community", "POST"): "handle_build_community",
    }

    @pytest.mark.parametrize("key,handler", sorted(EXPECTED.items()))
    def test_binding(self, key, handler):
        path, method = key
        matches = [r for r in _routes() if r.path == path and method in r.methods]
        assert matches, f"no {method} {path} registered at all"
        assert matches[0].name == handler, (
            f"{method} {path} is bound to {matches[0].name}(), expected {handler}()"
        )


class TestTheEndpointActuallyAnswers:
    """The binding check above is static. This one drives the app, which is
    what none of the existing /jobs tests do — they call the handler directly,
    so the routing table was never exercised."""

    def test_jobs_returns_a_list_not_a_422(self, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        resp = TestClient(app, raise_server_exceptions=False).get("/jobs")
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json(), list)

    def test_job_groups_returns_a_list(self, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        resp = TestClient(app, raise_server_exceptions=False).get("/job-groups")
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json(), list)

    def test_no_route_demands_an_unexpected_query_parameter(self, monkeypatch):
        """The symptom the orphan produced: a handler with a bare positional
        argument becomes a REQUIRED query parameter, so the endpoint 422s for
        every ordinary caller."""
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        client = TestClient(app, raise_server_exceptions=False)
        for path in ("/jobs", "/job-groups", "/status", "/health"):
            r = client.get(path)
            assert r.status_code != 422, f"{path} demands a query parameter: {r.text}"
