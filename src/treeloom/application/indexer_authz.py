"""Authorization, authentication, and the redaction built on them.

Split out of `indexer_service.py` (stage 3/6). Every route that asks "may this
caller do this?" funnels through here: the per-source ACL choke point
(`_authorize_scope`), the scope and admin gates, the principal resolution the
skip-route endpoints must do for themselves, and the job-view redaction that
decides what an unauthenticated caller sees.

These group cleanly because they were already self-contained — none of them
called anything else in the module.

Nothing here is a route. A route decorator must stay attached to the handler
written for it: `@app.get(...)` binds to whatever `def` follows it, so moving
a helper that sits between a decorator and its handler silently re-points the
endpoint. That is not hypothetical — it shipped as GET /jobs answering 422
(see tests/unit/test_route_table.py). The extraction refuses to move any
route-decorated function, and that guard is asserted, not assumed.

One-way dependency: `indexer_service` imports this, never the reverse. Shared
state comes from `indexer_state`.
"""

from __future__ import annotations

import logging
from fastapi import HTTPException, Request
from treeloom import graph_store
from treeloom.adapters.authorization.grant_store import GrantLookupUnavailable
from treeloom.application import indexer_state as _state
from treeloom.domain.authorization import Role
from treeloom.domain.authorization.access import AuthorizationService
from treeloom.domain.search_audit import SearchAuditRecord
import os
import uuid
from treeloom.application import indexer_state as _state

logger = logging.getLogger(__name__)

# Job fields that describe WHERE the work is happening or WHY it failed, as
# opposed to how far along it is. `source` is an absolute host path or a clone
# URL, `current_file` walks the tree one entry at a time, and `error`/`message`
# carry exception text. None of it is needed to render progress.
_JOB_SENSITIVE_FIELDS = frozenset(
    {"source", "source_url", "source_branch", "current_file", "error", "message"}
)


# Only trust X-Forwarded-For (proxy-supplied client IP) when explicitly behind
# a trusted reverse proxy; otherwise a client can spoof the header to dodge the
# per-IP limiter.
LOGIN_TRUST_FORWARDED_FOR = (
    os.environ.get("TREELOOM_LOGIN_TRUST_FORWARDED_FOR", "0") == "1"
)


# Allowlist of root directories the /preflight endpoint will scan when
# called from outside the process. CLI invocations bypass this — they
# already run with the operator's filesystem privileges. Empty (the
# default) means the endpoint refuses any `path` argument, and only
# remote URL scans are allowed.
PREFLIGHT_ALLOWED_ROOTS: list[str] = [
    os.path.realpath(p) for p in os.environ.get("PREFLIGHT_ALLOWED_ROOTS", "").split(":") if p
]


def _path_is_under_allowed_root(candidate: str) -> bool:
    if not PREFLIGHT_ALLOWED_ROOTS:
        return False
    real = os.path.realpath(candidate)
    return any(
        real == root or real.startswith(root + os.sep)
        for root in PREFLIGHT_ALLOWED_ROOTS
    )


def _require_path_in_scope(expanded: str) -> None:
    """Reject a local indexing path that escapes PREFLIGHT_ALLOWED_ROOTS.

    Enforced ONLY when PREFLIGHT_ALLOWED_ROOTS is configured. `/preflight`
    fails closed on an empty allow-list because it is an enumeration oracle
    nobody needs by default; the indexing endpoints cannot — an empty
    allow-list is the shipped default, and failing closed here would 403 every
    ordinary `index this repo` call. So this is opt-in scoping for deployments
    that want it, not a guard that makes the default posture safe.

    What actually keeps these endpoints off the network is the loopback
    INDEXER_HOST default plus AUTH_ENABLED; this narrows the blast radius for
    an operator who has set roots, and closes the gap where `/preflight`
    validated a path but `/index-*` did not (CWE-22).

    Checked BEFORE any existence test, so an out-of-scope probe gets the same
    403 whether or not the target exists — a 404/403 split would itself be a
    filesystem oracle.
    """
    if not PREFLIGHT_ALLOWED_ROOTS:
        return
    if not _path_is_under_allowed_root(expanded):
        raise HTTPException(
            403,
            "path is not under any PREFLIGHT_ALLOWED_ROOTS entry; "
            "use `url` for arbitrary git sources, or add the root to "
            "PREFLIGHT_ALLOWED_ROOTS on the indexer.",
        )


def _public_job(job: dict, *, redact: bool = False) -> dict:
    """Serialize a job for the wire, optionally stripping sensitive fields.

    /jobs, /jobs/{id}, /jobs/{id}/errors and /status are deliberately reachable
    without a credential — they are progress endpoints, and that is documented.
    But they returned the whole record, so an unauthenticated caller could read
    the absolute filesystem path of every indexed source, watch the indexer walk
    a private tree file by file, and collect exception text from failures
    (CWE-306 / CWE-209). That is reconnaissance, not progress.

    Redaction applies only when AUTH_ENABLED=true and the caller presented no
    valid credential: an install that has not turned auth on is unchanged, and
    an authenticated caller still sees everything.
    """
    out = {k: v for k, v in job.items() if k != "last_log_time"}
    if redact:
        for field in _JOB_SENSITIVE_FIELDS:
            if field in out:
                out[field] = None
        out["redacted"] = True
    return out


async def _job_view_is_redacted(request: Request) -> bool:
    """True when this caller should get the progress-only view of a job."""
    if os.environ.get("AUTH_ENABLED", "").lower() != "true":
        return False
    try:
        return await _require_authenticated_user(request) is None
    except HTTPException:
        return True


async def _source_id_for_job(job_id: str) -> str | None:
    """The source a job operated on, for authorization.

    In-memory first (the live job table), then the durable store, because a
    finished job's errors outlive its _state._jobs entry — and an unknown job must
    resolve to None, which _visible_job_filter treats as "not yours".
    """
    job = _state._jobs.get(job_id)
    if job:
        return job.get("source_id")
    if _state._job_store is not None:
        try:
            rec = await _state._job_store.get(job_id)
            if rec is not None:
                return rec.source_id
        except Exception:
            logger.exception("job lookup failed while authorizing %s", job_id)
    return None


async def _visible_job_filter(request: Request):
    """Return a predicate deciding which jobs this caller may see.

    Jobs carry no owner column, so there is nothing to compare a caller
    against directly (CWE-639) — but every job carries the `source_id` it
    operates on, and per-source authorization already exists. Reusing it gives
    the right answer with no schema change and, more importantly, no second
    permission model to keep in step with the first.

    Returns a callable; `None` means "no filtering" (enforcement off), which
    keeps the open-mode behaviour byte-identical.

    Decisions are cached per source_id for the life of one request: a listing
    is usually many jobs over few sources, and without the cache this would be
    one authorization round-trip per job.
    """
    if not _search_auth_enforced():
        return None
    try:
        user = await _require_authenticated_user(request)
    except HTTPException:
        # Unauthenticated on a route the middleware exempts. Progress stays
        # visible (redacted elsewhere); ownership-scoped listing does not.
        return lambda source_id: False
    if user is None or getattr(user, "role", None) == Role.ADMIN:
        return None

    svc = _authz()
    cache: dict[str, bool] = {}

    async def _allowed(source_id: str | None) -> bool:
        if not source_id:
            return False  # unattributable job — not yours to see
        if source_id not in cache:
            try:
                cache[source_id] = (
                    await svc.authorize_source(user, source_id)
                ).allowed
            except GrantLookupUnavailable:
                cache[source_id] = False  # fail closed, as everywhere else
        return cache[source_id]

    return _allowed


def _require_admin(request: Request):
    """Fail-closed admin gate. Requires `request.state.user` to be present,
    have ADMIN role, AND hold the `admin` scope.

    The scope check closes the over-privileged-token hole: a scoped PAT or
    API key owned by an admin still carries only the scopes it was minted
    with, so an admin's `search`-only token cannot reach admin endpoints.
    Full-role auth paths (session login, legacy api_key, the
    dev-mode synthetic admin) all set scopes via `scopes_for_role`, so a
    genuine admin session always has the `admin` scope.

    The dev-workflow escape hatch lives in the auth middleware, NOT here:
    when `AUTH_ENABLED=false` AND `TREELOOM_DEV_MODE=1` AND the request
    is from loopback, the middleware injects a synthetic admin user with
    full scopes. Any path that bypasses the middleware — or any external
    caller in dev mode — gets 401/403 from this helper.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Authentication required")
    if user.role != Role.ADMIN:
        raise HTTPException(403, "Admin access required")
    if "admin" not in getattr(request.state, "scopes", set()):
        raise HTTPException(403, "Scope 'admin' required")
    return user


def _require_scope(request: Request, scope: str):
    """Scope gate: passes only when the caller's token carries the scope.

    There is deliberately NO `role == ADMIN` bypass: that bypass made
    per-token scopes meaningless for admin-owned tokens (a scoped PAT/API
    key owned by an admin could reach any endpoint). Full-role auth paths
    (session, legacy api_key, dev synthetic admin) populate
    `request.state.scopes` from `scopes_for_role`, so an admin via those
    paths still holds every scope and passes. Only scope-limited PAT/API-key
    principals are constrained — which is the point.

    When AUTH_ENABLED is off, always passes (mirrors _require_admin behavior).
    Returns None when auth is disabled, the User otherwise.

    Coverage today is DELETE /sources/{source_id} only. The mutating index
    endpoints authenticate but do not yet check a scope, so a search-only PAT
    can still drive them — gating them is a deliberate change with a blast
    radius (every scope-limited token in existing deployments), not something
    to slip in beside a fix.
    """
    if os.environ.get("AUTH_ENABLED", "").lower() != "true":
        return None
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Authentication required")
    caller_scopes = getattr(request.state, "scopes", set())
    if scope in caller_scopes:
        return user
    raise HTTPException(403, f"Scope '{scope}' required")


def _acting_as_admin(request: Request) -> bool:
    """True when the caller is an admin acting through a credential that
    actually carries the `admin` scope.

    The self-or-admin endpoints use this (not a bare ``role == ADMIN``
    check) so a scope-limited admin token — e.g. a search-only PAT owned by
    an admin — cannot perform admin-on-others actions like resetting another
    user's password or minting their tokens. Full-role auth paths
    (session/legacy/dev synthetic admin) always include the `admin`
    scope for admins, so genuine admin sessions pass.
    """
    user = getattr(request.state, "user", None)
    if user is None or user.role != Role.ADMIN:
        return False
    return "admin" in getattr(request.state, "scopes", set())


# "auto" (default) | "1" | "0". See _cookie_secure().
_COOKIE_SECURE_MODE = os.environ.get("TREELOOM_COOKIE_SECURE", "auto").strip().lower()


def _cookie_secure(request: Request) -> bool:
    """Whether the session cookie should carry the Secure attribute.

    This was a plain env flag defaulting to OFF, so every deployment that
    did not remember to set it served a session cookie a browser would
    happily send over plain HTTP — interceptable on any shared network, and
    silent, because nothing about the login looks different (CWE-614). The
    default has to be right for people who never read this setting.

    "auto" derives it from the request instead: Secure when the connection
    that carried the login was HTTPS. Local HTTP development keeps working
    without an opt-out, and an HTTPS deployment is protected without an
    opt-in — the case that was getting it wrong.

    X-Forwarded-Proto is consulted ONLY under
    TREELOOM_LOGIN_TRUST_FORWARDED_FOR, the existing behind-a-trusted-proxy
    switch. Untrusted, a client could send `X-Forwarded-Proto: http` and
    strip Secure from its own cookie — which is the direction that matters;
    claiming `https` only ever adds the flag.

    "1" and "0" remain available for an operator who knows their topology and
    wants it stated rather than derived.
    """
    if _COOKIE_SECURE_MODE in ("1", "true", "yes"):
        return True
    if _COOKIE_SECURE_MODE in ("0", "false", "no"):
        return False
    scheme = request.url.scheme
    if LOGIN_TRUST_FORWARDED_FOR:
        forwarded = request.headers.get("x-forwarded-proto", "")
        if forwarded:
            scheme = forwarded.split(",")[0].strip().lower()
    return scheme == "https"


_DUMMY_PASSWORD_HASH: str | None = None


def _dummy_password_hash() -> str:
    """A real bcrypt hash of a random value, for the no-such-user branch.

    Computed once on first use rather than at import so startup does not pay
    a KDF round, and never compared successfully — the input is random and
    discarded. Its only job is to make the absent-user path cost the same as
    the present-user path.
    """
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        from treeloom.adapters.authorization.user_store import hash_password

        _DUMMY_PASSWORD_HASH = hash_password(uuid.uuid4().hex)
    return _DUMMY_PASSWORD_HASH


def _login_client_ip(request: Request) -> str:
    """Resolve the caller IP for per-IP login throttling.

    Only honors the proxy-supplied X-Forwarded-For when
    TREELOOM_LOGIN_TRUST_FORWARDED_FOR=1 (we're behind a trusted reverse
    proxy); otherwise a client could spoof the header to evade the limiter.
    Falls back to "unknown" so a None peer never crashes the handler.
    """
    if LOGIN_TRUST_FORWARDED_FOR:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


async def _load_source(source_id: str) -> dict | None:
    """Load source metadata from PostgreSQL, falling back to graph store."""
    try:
        record = await _state._source_repo.get_by_id(source_id)
        if record is not None:
            return {
                "id": record.id,
                "path": record.path,
                "url": record.url,
                "branch": record.branch,
                "commit_sha": record.commit_sha,
                "created_by": record.created_by,
            }
    except Exception:
        pass
    for s in await graph_store.list_sources():
        if s.get("id") == source_id:
            return s
    return None


async def _require_authenticated_user(request: Request):
    """Authenticate the caller for endpoints the middleware skip_routes exempt.

    `/sources` GET is in the middleware's skip_routes prefix list, which also
    inadvertently exempts `/sources/{id}/...` paths. This helper performs the
    same session-cookie + Bearer-token lookup the middleware would have done.
    Returns None when AUTH_ENABLED is not enabled globally (matches middleware
    behavior).
    """
    if os.environ.get("AUTH_ENABLED", "").lower() != "true":
        return None

    user = getattr(request.state, "user", None)
    if user is not None:
        return user

    # Session cookie (the SPA's username/password login). The middleware resolves
    # this for non-skip routes; skip-route handlers (/sources*) must do it too,
    # else a cookie-authenticated caller gets a spurious 401. Mirrors
    # AuthMiddleware._authenticate step 1.
    session_cookie = request.cookies.get("treeloom_session")
    if session_cookie and _state._session_store is not None:
        from treeloom.adapters.authorization.user_store import hash_api_key as _hak
        try:
            session = await _state._session_store.get_by_token_hash(_hak(session_cookie))
            if session is not None:
                su = await _state._user_store.get_by_id(session.user_id)
                if su is not None and su.active:
                    return su
        except Exception:
            logger.exception("session cookie authentication failed")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Authentication required")
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(401, "Authentication required")

    from treeloom.adapters.authorization.user_store import hash_api_key
    try:
        key_hash = hash_api_key(token)
        user = await _state._user_store.get_by_api_key_hash(key_hash)
    except Exception:
        raise HTTPException(401, "Authentication failed")
    if user is None:
        raise HTTPException(401, "Authentication failed")
    return user


def _search_auth_enforced() -> bool:
    """Whether read endpoints enforce per-principal authorization.

    Off when AUTH_ENABLED is not "true" (fully open / single-tier), and also
    off when TREELOOM_SEARCH_OPEN is set — the escape hatch for a deployment
    that wants protected *indexing* (AUTH_ENABLED=true) but deliberately open
    *search* (the simple single-tier evaluator path).
    """
    if os.environ.get("AUTH_ENABLED", "").lower() != "true":
        return False
    if os.environ.get("TREELOOM_SEARCH_OPEN", "").lower() in ("1", "true"):
        return False
    return True


def _write_auth_enforced() -> bool:
    """Whether DESTRUCTIVE endpoints enforce per-principal authorization.

    Deliberately NOT `_search_auth_enforced()`: TREELOOM_SEARCH_OPEN opens
    *search*, and must not also open deletion. An evaluator deployment that
    sets it to make search anonymous would otherwise silently revert
    DELETE /sources/{id} to "any caller may destroy any source" — reopening
    the CWE-862 the guard on that endpoint exists to close, via a flag whose
    name says nothing about writes.

    AUTH_ENABLED remains the global on/off switch, as it is everywhere else.
    """
    return os.environ.get("AUTH_ENABLED", "").lower() == "true"


async def _owner_of(source_id: str) -> str | None:
    """Resolve a source's owner (created_by) for the ownership grant rule."""
    src = await _load_source(source_id)
    return (src.get("created_by") if src else None) or None


def _authz() -> AuthorizationService:
    """Build the authorization service over the live grant store."""
    return AuthorizationService(grants=_state._grant_store, owner_of=_owner_of)


async def _audit_search(
    user, action: str, scope: str, decision: str, source_id: str | None
) -> None:
    """Best-effort who-searched-what record (never raises)."""
    await _state._search_audit.record(
        SearchAuditRecord(
            user_id=user.id,
            action=action,
            scope=scope,
            decision=decision,
            source_id=source_id,
            principal_groups=tuple(getattr(user, "group_ids", []) or ()),
        )
    )


async def _authorize_scope(
    request: Request,
    source_id: str | None,
    action: str = "search",
    enforced: bool | None = None,
) -> list[str] | None:
    """Enforce per-principal authorization for a read scope.

    Returns the list of source_ids to EXCLUDE from a shared-index query (may be
    empty), or None when there is nothing to exclude (single-source query, or
    enforcement disabled). Raises 401 (no/invalid credential) or 403 (denied).
    Emits a who-searched-what audit record for each decision.

    Callers may override the enforcement predicate with *enforced*; write
    endpoints pass `_write_auth_enforced()` so that TREELOOM_SEARCH_OPEN — an
    escape hatch for anonymous *search* — cannot switch off authorization for
    a destructive operation.

    * enforcement off    → returns None, no checks (open mode preserved).
    * source_id pinned   → single-source decision; 403 on deny.
    * not source-pinned  → shared-index decision (cross_repo or a path_prefix
      spanning sources); 403 without all_access, else the caller's denied
      sources as the exclusion set.
    """
    if enforced is None:
        enforced = _search_auth_enforced()
    if not enforced:
        return None
    user = await _require_authenticated_user(request)
    if user is None:  # defensive — only when AUTH is globally off
        return None

    svc = _authz()
    try:
        return await _authorize_scope_decide(svc, user, request, source_id, action)
    except GrantLookupUnavailable:
        # The grant store could not answer. Proceeding would mean deciding on
        # an empty grant set, which for an all_access caller reads as "no deny
        # grants" and admits them to the very source a deny grant excludes.
        logger.exception(
            "Authorization store unavailable; refusing %s (source_id=%s)",
            action, source_id,
        )
        raise HTTPException(
            503, "Authorization store unavailable — request refused"
        )


async def _authorize_scope_decide(svc, user, request, source_id, action):
    """The decision itself, split out so the 503 path above stays readable."""
    if source_id:
        decision = await svc.authorize_source(user, source_id)
        await _audit_search(
            user, action, "source",
            "allow" if decision.allowed else "deny", source_id,
        )
        if not decision.allowed:
            raise HTTPException(403, "Not authorized for this source")
        return None

    # Not pinned to one source → shared index (cross_repo or path_prefix).
    decision = await svc.authorize_shared(user)
    await _audit_search(
        user, action, "shared",
        "allow" if decision.allowed else "deny", None,
    )
    if not decision.allowed:
        raise HTTPException(
            403, "Shared-index search requires all_access"
        )
    return list(decision.excluded_source_ids)


