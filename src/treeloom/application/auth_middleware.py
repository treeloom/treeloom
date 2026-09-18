"""FastAPI middleware for API key authentication and authorization.

Resolution order (first match wins):
  1. Session cookie ``treeloom_session`` (HttpOnly, HMAC-hashed)
  2. Bearer token:
     a. Personal Access Token (PAT) — token_store lookup by HMAC hash
     b. API key — api_key_store lookup by HMAC hash
     c. Legacy api_key_hash on users table (backward compatibility)

Sets both ``request.state.user`` and ``request.state.scopes`` on success.
Skips authentication for /health, /search, /auth/login, /auth/logout.
Disabled entirely when AUTH_ENABLED environment variable is not "true".
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from treeloom.domain.authorization import (
    Role,
    User,
    UserStorePort,
    TokenStorePort,
    ApiKeyStorePort,
    SessionStorePort,
    scopes_for_role,
)
from treeloom.adapters.authorization.user_store import hash_api_key


def _derive_scoped_principal(owner: User, scopes: list[str]) -> User:
    """Return an owner copy whose role follows the TOKEN's scopes.

    A token's effective data-access privilege must follow its scopes, not its
    creator's role. Without this, a ``search``-scoped key/PAT minted
    by an admin still authenticates as the admin user (role ADMIN) and inherits
    the ``access.py`` admin see-everything bypass — leaking every private repo.

    The effective role is ADMIN only when the token carries the ``admin`` scope;
    otherwise USER. We return a ``dataclasses.replace`` COPY (never mutate the
    cached owner) and preserve the owner's ``id`` (the source-owner-grant rule in
    access.py keys on ``owner_id == user.id``) and ``all_access`` so per-source
    grants still compose for a derived non-admin principal.
    """
    effective_role = Role.ADMIN if "admin" in scopes else Role.USER
    if owner.role == effective_role:
        return owner
    return dataclasses.replace(owner, role=effective_role)


# Loopback addresses that qualify a request as "local" for the
# TREELOOM_DEV_MODE escape hatch. IPv4 loopback is 127.0.0.0/8, but
# uvicorn typically reports the connect address verbatim — checking
# the canonical forms covers the common cases without false positives.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Evidence that a request passed through a proxy, which means `client.host`
# may have been set by that proxy rather than observed on the socket.
_FORWARDED_HEADERS = frozenset(
    {"x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "forwarded"}
)


def _is_loopback_client(request) -> bool:
    """True when the request's TCP peer is on loopback.

    Used by the dev-mode escape hatch so admin gates relax for the
    local dashboard but stay locked from any routable client even when
    the server binds 0.0.0.0.
    """
    client = getattr(request, "client", None)
    if client is None:
        return False

    # `client.host` is not necessarily the TCP peer. uvicorn runs
    # ProxyHeadersMiddleware with proxy_headers=True by default and rewrites
    # scope["client"] from X-Forwarded-For whenever the immediate peer is in
    # forwarded_allow_ips (default 127.0.0.1) — which a reverse proxy on the
    # same host always is. uvicorn 0.34 walks the list right-to-left and
    # returns the first untrusted hop, so an appending proxy still reports the
    # real client and this cannot be spoofed outright; but a proxy that emits
    # `X-Forwarded-For: 127.0.0.1` itself leaves every hop trusted and hands
    # back loopback for every remote caller.
    #
    # A request carrying forwarded headers has provably traversed something,
    # so it does not get the dev-mode admin grant. This can only narrow the
    # grant, never widen it.
    headers = getattr(request, "headers", None) or {}
    try:
        if any(k.lower() in _FORWARDED_HEADERS for k in headers):
            return False
    except TypeError:  # a header container that isn't iterable — stay closed
        return False

    host = (client.host or "").lower()
    if host in _LOOPBACK_HOSTS:
        return True
    # Catch the IPv4 127/8 range without depending on `ipaddress` for
    # every request — cheap prefix check.
    return host.startswith("127.")


def _synthetic_dev_admin() -> User:
    """Build the in-memory admin principal used only when
    AUTH_ENABLED=false + TREELOOM_DEV_MODE=1 + loopback client.
    Never persisted, never has an API key."""
    return User(
        id="dev-mode-admin",
        username="dev",
        email="",
        api_key_hash="",
        role=Role.ADMIN,
        active=True,
    )


logger = logging.getLogger(__name__)

# Endpoints that do not require authentication (all HTTP methods).
# `/dashboard` is reserved for a same-origin operator UI shell and stays
# public so a logged-out user can load the page and reach the login form;
# every data call the page makes is authenticated individually by this
# middleware.
PUBLIC_PATHS = {"/health", "/status", "/search", "/auth/login", "/auth/logout", "/dashboard"}


class AuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces authentication.

    On every request to a protected endpoint:
    1. Tries session cookie (treeloom_session).
    2. Tries Bearer token: PAT → API key → legacy.
    3. Attaches the resolved User to request.state.user and the token
       scopes (or role-derived scopes for session/legacy) to
       request.state.scopes.
    4. Returns 401 if nothing matched.

    Skips /health, /status, /search, /auth/login, /auth/logout unconditionally.
    Bypasses entirely when AUTH_ENABLED != "true".
    """

    def __init__(
        self,
        app,
        user_store: UserStorePort,
        session_store: Optional[SessionStorePort] = None,
        token_store: Optional[TokenStorePort] = None,
        api_key_store: Optional[ApiKeyStorePort] = None,
        skip_paths: Optional[set[str]] = None,
        skip_routes: Optional[dict[str, Optional[set[str]]]] = None,
    ):
        """Initialize the middleware.

        Args:
            app: The ASGI application.
            user_store: The UserStorePort adapter for user lookups.
            session_store: Optional SessionStorePort for cookie-based sessions.
            token_store: Optional TokenStorePort for Personal Access Tokens.
            api_key_store: Optional ApiKeyStorePort for service API keys.
            skip_paths: Additional paths to skip auth for all methods
                        (beyond PUBLIC_PATHS).
            skip_routes: Mapping of URL path prefix → set of HTTP methods
                        to skip, or None for all methods.
        """
        super().__init__(app)
        self._user_store = user_store
        self._session_store = session_store
        self._token_store = token_store
        self._api_key_store = api_key_store
        self._skip_paths = skip_paths or set()
        self._skip_routes = skip_routes or {}

    @property
    def _all_skip_paths(self) -> set[str]:
        return PUBLIC_PATHS | self._skip_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process each request through the auth middleware.

        Returns 401 for unauthorized requests to protected endpoints.
        """
        # CORS preflight (OPTIONS) carries no credentials and must reach the
        # (inner) CORS middleware that answers it — gating it on auth makes the
        # browser block every cross-origin call from the SPA. Auth still applies
        # to the real request that follows the preflight.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check if auth is globally disabled
        if os.environ.get("AUTH_ENABLED", "").lower() != "true":
            # Dev convenience: when both AUTH_ENABLED=false AND
            # TREELOOM_DEV_MODE=1 are set AND the caller is on loopback,
            # inject a synthetic admin so admin-gated endpoints
            # (/embedding-backends, /users, etc.) work for local dashboard
            # use without granting admin to anything routable. Without
            # the dev flag, admin-gated endpoints stay closed even when
            # auth is off — `_require_admin` requires `request.state.user`.
            if (
                os.environ.get("TREELOOM_DEV_MODE", "").lower() in ("1", "true")
                and _is_loopback_client(request)
            ):
                request.state.user = _synthetic_dev_admin()
                request.state.scopes = {"search", "index", "admin"}
            return await call_next(request)

        # Skip auth for public paths (exact match, all methods)
        if request.url.path in self._all_skip_paths:
            return await call_next(request)

        # Skip auth for method-specific route prefixes
        if self._is_skipped_route(request.method, request.url.path):
            return await call_next(request)

        # Resolve principal and scopes
        result = await self._authenticate(request)
        if result is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        user, scopes = result

        # Attach user and scopes to request state for downstream handlers
        request.state.user = user
        request.state.scopes = scopes
        return await call_next(request)

    def _is_skipped_route(self, method: str, path: str) -> bool:
        """Check if the path matches a skip_route prefix with the given method."""
        for prefix, allowed_methods in self._skip_routes.items():
            if path == prefix or path.startswith(prefix + "/"):
                if allowed_methods is None or method.upper() in allowed_methods:
                    return True
        return False

    async def _authenticate(
        self, request: Request
    ) -> Optional[tuple[User, set[str]]]:
        """Resolve principal from session cookie or bearer token.

        Resolution order (first match wins):
        1. Session cookie → full role scopes
        2. PAT bearer → token's own scopes
        3. API key bearer → key's own scopes
        4. Legacy api_key_hash → full role scopes

        Returns (user, scopes) on success, None on failure.
        """
        # 1. Session cookie
        session_cookie = request.cookies.get("treeloom_session")
        if session_cookie and self._session_store is not None:
            try:
                cookie_hash = hash_api_key(session_cookie)
                session = await self._session_store.get_by_token_hash(cookie_hash)
                if session is not None:
                    user = await self._user_store.get_by_id(session.user_id)
                    # Deactivated accounts (active=False) must lose access even
                    # if a session row survives — get_by_id does not filter on
                    # active, so check it here.
                    if user is not None and user.active:
                        # Touch session last_seen_at (best-effort)
                        try:
                            await self._session_store.touch(cookie_hash)
                        except Exception:
                            pass
                        return user, {s.value for s in scopes_for_role(user.role)}
            except Exception:
                logger.exception("session cookie authentication failed")

        # 2–5: Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:].strip()
        if not token:
            return None

        # HMAC-hash the bearer token once for all remaining lookups
        try:
            key_hash = hash_api_key(token)
        except Exception:
            logger.exception("Failed to hash bearer token")
            return None

        # 3. PAT lookup
        if self._token_store is not None:
            try:
                pat = await self._token_store.get_by_token_hash(key_hash)
                if pat is not None:
                    user = await self._user_store.get_by_id(pat.user_id)
                    # A deactivated owner revokes their PATs (get_by_id does
                    # not filter on active).
                    if user is not None and user.active:
                        # Data-access privilege follows the TOKEN's scopes, not
                        # the owner's role: an admin who mints a search-only PAT
                        # must not read every private repo through it. Derive an
                        # effective role from the scopes on a COPY of the owner
                        # (never mutate the cached object); keep created_by and
                        # all_access so per-source grants still compose.
                        user = _derive_scoped_principal(user, pat.scopes)
                        # Touch last_used (best-effort)
                        try:
                            await self._token_store.touch_last_used(pat.id)
                        except Exception:
                            pass
                        return user, set(pat.scopes)
            except Exception:
                logger.exception("PAT authentication failed")

        # 4. API key lookup
        if self._api_key_store is not None:
            try:
                api_key = await self._api_key_store.get_by_token_hash(key_hash)
                if api_key is not None:
                    user = None
                    if api_key.created_by:
                        # Owner-bound key: authenticate as the owner, but only
                        # while that account exists and is active. A deactivated
                        # owner revokes the key — do NOT fall through to a
                        # synthetic admin principal (that would be an escalation).
                        owner = await self._user_store.get_by_id(api_key.created_by)
                        if owner is not None and owner.active:
                            # Data-access privilege follows the KEY's scopes,
                            # not the owner's role: a search-scoped key minted by
                            # an admin must not inherit the admin see-everything
                            # bypass. Derive the effective role from the key
                            # scopes on a COPY of the owner (never mutate the
                            # cached object); keep created_by and the owner's
                            # all_access so all_access + per-source grants still
                            # compose for the derived principal.
                            user = _derive_scoped_principal(owner, api_key.scopes)
                        else:
                            logger.warning(
                                "api key %s owner missing/inactive — rejecting",
                                api_key.id,
                            )
                    else:
                        # Ownerless service key → synthetic service principal.
                        key_role = (
                            Role.ADMIN if "admin" in api_key.scopes else Role.USER
                        )
                        user = User(
                            id=f"apikey:{api_key.id}",
                            username=api_key.name,
                            email="",
                            api_key_hash="",
                            role=key_role,
                            active=True,
                            all_access=api_key.all_access,
                        )

                    if user is not None:
                        # Touch last_used (best-effort)
                        try:
                            await self._api_key_store.touch_last_used(api_key.id)
                        except Exception:
                            pass
                        return user, set(api_key.scopes)
            except Exception:
                logger.exception("API key authentication failed")

        # 5. Legacy api_key_hash fallback
        try:
            user = await self._user_store.get_by_api_key_hash(key_hash)
            if user is not None:
                return user, {s.value for s in scopes_for_role(user.role)}
        except Exception:
            logger.exception("Failed to look up user by API key hash")

        return None
