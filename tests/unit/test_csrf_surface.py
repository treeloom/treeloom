"""CSRF surface: what makes the cookie session safe, pinned.

The session cookie is a real ambient credential — the browser attaches it to
any request the page makes, including ones another site caused. Two properties
keep that from being CSRF, and neither is enforced by anything except this
file:

1. `SameSite=Lax` — the browser withholds the cookie on cross-site POST /
   PUT / PATCH / DELETE. This is the whole defense for mutating routes; there
   is no CSRF token.
2. No GET route has side effects. Lax still sends the cookie on top-level GET
   navigation, so a mutating GET would be reachable from `evil.com` with a
   plain link or <img>. `POST /build-community` is admin-only and its GET
   sibling is a progress poll — that split is load-bearing, not incidental.

The residual, stated rather than fixed: SameSite is a browser-side control,
so a client that ignores it is unprotected, and there is no defense-in-depth
token. Bearer-token callers (the SPA's primary path) are unaffected either
way — a token is not ambient.
"""

import inspect

import pytest

# Routes that are allowed to appear as GET. Each is a read. Adding a GET route
# means adding it here, which is the point: the next person has to state that
# their new GET does not mutate.
READ_ONLY_GETS = {
    "/health", "/status", "/metrics", "/jobs", "/jobs/{job_id}",
    "/jobs/{job_id}/errors", "/jobs/dead-letter", "/job-groups",
    "/job-groups/{group_id}", "/sources", "/sources/{source_id}/staleness",
    "/sources/{source_id}/grants", "/fleet", "/users", "/users/{user_id}/tokens",
    "/groups", "/groups/{group_id}/members", "/api-keys", "/auth/me",
    "/audit/search", "/embedding-backends", "/build-community", "/debug/tasks",
    "/find-definition", "/find-callers", "/find-references", "/openapi.json",
    "/docs", "/docs/oauth2-redirect", "/redoc", "/sse",
}

# Verbs a browser will not send cross-site with a Lax cookie.
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@pytest.fixture
def app(mock_milvus_collection, mock_neo4j_driver):
    from treeloom.indexer_service import app as _app
    return _app


def _routes(app):
    for r in app.routes:
        methods = getattr(r, "methods", None)
        if methods:
            yield r, methods


class TestNoStateChangingGet:
    def test_every_get_route_is_a_known_read(self, app):
        """A new GET route must be declared read-only here. If this fails,
        check whether the new route mutates before adding it to the list —
        SameSite=Lax does NOT protect GET."""
        gets = {
            r.path for r, m in _routes(app) if "GET" in m and r.path not in ("/",)
        }
        assert gets <= READ_ONLY_GETS, (
            f"undeclared GET route(s): {sorted(gets - READ_ONLY_GETS)}"
        )

    def test_build_community_mutates_only_under_post(self, app):
        """The one route with both verbs. GET reports progress; POST starts a
        fleet-wide rebuild. If GET ever gained the side effect, Lax would not
        stop a cross-site trigger."""
        get_h = [r.endpoint for r, m in _routes(app)
                 if r.path == "/build-community" and "GET" in m]
        post_h = [r.endpoint for r, m in _routes(app)
                  if r.path == "/build-community" and "POST" in m]
        assert get_h and post_h and get_h[0] is not post_h[0]
        src = inspect.getsource(get_h[0])
        for starter in ("create_task", "ensure_future", "BackgroundTasks", "await _run"):
            assert starter not in src, f"GET /build-community may start work: {starter}"


class TestTheCookieCarriesItsDefense:
    def test_login_sets_samesite_lax_and_httponly(self):
        """Read off the call site: the cookie's attributes ARE the CSRF
        control, so they belong in a test rather than only in review."""
        from treeloom.application import routes_auth

        src = inspect.getsource(routes_auth)
        idx = src.index('response.set_cookie(\n        "treeloom_session"')
        call = src[idx:idx + 400]
        assert 'samesite="lax"' in call
        assert "httponly=True" in call

    def test_logout_is_not_reachable_by_navigation(self, app):
        """A GET /auth/logout would let any site log a user out by linking to
        it. Minor, but it is the classic example and costs one assertion."""
        logout = {m for r, m in _routes(app) if r.path == "/auth/logout" for m in m}
        assert "GET" not in logout


class TestCorsCannotBeWidenedIntoCredentialedWildcard:
    """CORS is the other way a cross-origin page reaches this API. The
    wildcard-with-credentials combination (CWE-942) was fixed earlier; this
    keeps the guard next to the CSRF reasoning that depends on it."""

    def test_wildcard_in_the_explicit_list_does_not_allow_credentials(self):
        from fastapi import FastAPI

        from treeloom.application.cors import add_cors_middleware

        probe = FastAPI()
        add_cors_middleware(probe, "https://ui.example.com,*")
        opts = [
            m.kwargs for m in probe.user_middleware
            if "CORSMiddleware" in str(m.cls)
        ]
        assert opts, "CORS middleware not installed"
        cfg = opts[0]
        if "*" in cfg.get("allow_origins", []):
            assert not cfg.get("allow_credentials", False), (
                "wildcard origin paired with credentials"
            )
