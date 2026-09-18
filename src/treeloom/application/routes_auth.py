"""Authentication and identity-management endpoints.

Split out of `indexer_service.py` (stage 4/6): /auth/*, /users/*, /groups/*
and /api-keys/*, plus the request models only these routes use.

Registered on an APIRouter that `indexer_service` includes, rather than on
`app` directly. The paths, methods and handler names are unchanged, so
`tests/unit/test_route_table.py` pins the bindings across the move — which
matters here more than anywhere, since a route decorator binds to whatever
`def` follows it and this stage moves 26 of them at once.

Authorization helpers come from `indexer_authz`, shared state from
`indexer_state`, both module-qualified. Nothing here imports
`indexer_service`.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from datetime import datetime, timezone
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field
from treeloom.adapters.authorization.api_key_store import RevocationUnavailable
from treeloom.adapters.postgresql.login_attempt_store import LoginThrottleContention
from treeloom.application import indexer_authz as authz
from treeloom.application import indexer_state as _state
from treeloom.domain.authorization import Role, scopes_for_role
import asyncio
from treeloom.application import indexer_authz as authz
from treeloom.application import indexer_state as _state

logger = logging.getLogger(__name__)

router = APIRouter()


def _session_ttl_hours() -> int:
    """Session lifetime in hours; falls back to 12 on a malformed env value
    rather than crashing module import / startup."""
    raw = os.environ.get("TREELOOM_SESSION_TTL_HOURS", "12")
    try:
        hours = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid TREELOOM_SESSION_TTL_HOURS=%r — using default 12", raw
        )
        return 12
    return hours if hours > 0 else 12


# ── Login rate-limit config ───────────────────────────────────
# Brute-force protection for POST /auth/login: at most LOGIN_MAX_ATTEMPTS
# failed logins per (username, client_ip) within LOGIN_WINDOW_SECONDS before
# the next attempt is refused with 429 + Retry-After.
LOGIN_MAX_ATTEMPTS = int(os.environ.get("TREELOOM_LOGIN_MAX_ATTEMPTS", "5"))


LOGIN_WINDOW_SECONDS = int(os.environ.get("TREELOOM_LOGIN_WINDOW_SECONDS", "300"))


# Per-username ceiling across ALL client IPs (CWE-307). The per-(username, IP)
# limit alone lets an attacker with N source IPs get N × LOGIN_MAX_ATTEMPTS
# guesses; this account-scoped counter caps total failures per username per
# window regardless of IP rotation. Set well above the per-IP limit so
# legitimate users sharing a NAT/proxy egress aren't locked out.
LOGIN_MAX_ATTEMPTS_PER_USERNAME = int(
    os.environ.get("TREELOOM_LOGIN_MAX_ATTEMPTS_PER_USERNAME", "50")
)


def _user_to_response(user) -> dict:
    """Convert a User entity to a safe API response (no api_key_hash)."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "active": user.active,
        "created_at": user.created_at.isoformat(),
    }


_SESSION_TTL_HOURS = _session_ttl_hours()


def _pat_to_response(pat, include_hash: bool = False) -> dict:
    """Convert a PersonalAccessToken to a safe API response."""
    data = {
        "id": pat.id,
        "user_id": pat.user_id,
        "name": pat.name,
        "scopes": pat.scopes,
        "created_at": pat.created_at.isoformat(),
        "expires_at": pat.expires_at.isoformat() if pat.expires_at else None,
        "last_used_at": pat.last_used_at.isoformat() if pat.last_used_at else None,
        "revoked": pat.revoked,
    }
    # Never include token_hash in responses
    return data


def _api_key_to_response(key) -> dict:
    """Convert an ApiKey to a safe API response (no token_hash)."""
    return {
        "id": key.id,
        "name": key.name,
        "scopes": key.scopes,
        "all_access": key.all_access,
        "created_by": key.created_by,
        "created_at": key.created_at.isoformat(),
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
        "revoked": key.revoked,
    }


def _group_to_response(g) -> dict:
    return {
        "id": g.id,
        "name": g.name,
        "all_access": g.all_access,
        "created_at": g.created_at.isoformat(),
    }


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1)
    email: str = ""
    role: str = "user"


class LoginRequest(BaseModel):
    username: str
    password: str


class SetPasswordRequest(BaseModel):
    password: str
    # Required for self-service changes; ignored when an admin resets
    # another user's password (CWE-620 re-authentication guard).
    current_password: str = ""


class CreateTokenRequest(BaseModel):
    name: str = Field(..., min_length=1)
    scopes: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1)
    scopes: list[str] = Field(default_factory=list)
    all_access: bool = False
    expires_at: datetime | None = None


class SetRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(admin|user)$")


class CreateGroupRequest(BaseModel):
    name: str = Field(..., min_length=1)
    all_access: bool = False


class GroupMemberRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class SetAllAccessRequest(BaseModel):
    all_access: bool


@router.post("/users", status_code=201)
async def create_user(req: CreateUserRequest, request: Request):
    """Create a new user (admin only).

    Returns the user record with the raw API key included (shown once).
    """
    authz._require_admin(request)

    import uuid as _uuid
    from treeloom.adapters.authorization.user_store import hash_api_key

    # Validate email format (basic: must contain @ if non-empty)
    if req.email and "@" not in req.email:
        raise HTTPException(400, "Invalid email format")

    # Generate a new API key and hash it
    raw_key = _uuid.uuid4().hex + _uuid.uuid4().hex  # 64-char key
    api_key_hash = hash_api_key(raw_key)

    # Resolve role
    role_value = req.role.lower()
    if role_value not in ("admin", "user"):
        raise HTTPException(400, f"Invalid role: {req.role}. Must be 'admin' or 'user'.")
    role = Role.ADMIN if role_value == "admin" else Role.USER

    user = await _state._user_store.create_user(
        username=req.username,
        email=req.email,
        role=role,
        api_key_hash=api_key_hash,
    )

    if user is None:
        raise HTTPException(503, "Database unavailable")

    response_data = _user_to_response(user)
    response_data["api_key"] = raw_key  # Show raw key once
    return response_data


@router.get("/users")
async def list_users(request: Request):
    """List all users (admin only).

    Returns a list of user records without api_key_hash.
    """
    authz._require_admin(request)
    users = await _state._user_store.list_users()
    return [_user_to_response(u) for u in users]


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, request: Request):
    """Delete (soft-delete) a user (admin only).

    Sets active=False. Returns 204 on success.
    """
    authz._require_admin(request)
    success = await _state._user_store.delete_user(user_id)
    if not success:
        raise HTTPException(404, f"User not found: {user_id}")
    return None


@router.post("/users/{user_id}/rotate-key")
async def rotate_key(user_id: str, request: Request):
    """Rotate a user's API key (self or admin).

    Generates a new API key, updates the hash in the store,
    and returns the raw key (shown once).
    """
    current_user = getattr(request.state, "user", None)
    if current_user is None:
        raise HTTPException(401, "Authentication required")

    # Must be self or admin
    if current_user.id != user_id and not authz._acting_as_admin(request):
        raise HTTPException(403, "Cannot rotate another user's API key")

    import uuid as _uuid
    from treeloom.adapters.authorization.user_store import hash_api_key

    raw_key = _uuid.uuid4().hex + _uuid.uuid4().hex
    api_key_hash = hash_api_key(raw_key)

    success = await _state._user_store.update_api_key_hash(user_id, api_key_hash)
    if not success:
        raise HTTPException(404, f"User not found: {user_id}")

    return {"api_key": raw_key}


@router.patch("/users/{user_id}")
async def set_user_all_access(
    user_id: str, req: SetAllAccessRequest, request: Request
):
    """Set a user's all_access flag (admin only).

    all_access lets the user run shared / cross-repo searches (minus any
    sources they're explicitly denied).
    """
    authz._require_admin(request)
    success = await _state._user_store.update_all_access(user_id, req.all_access)
    if not success:
        raise HTTPException(404, f"User not found: {user_id}")
    return {"id": user_id, "all_access": req.all_access}


@router.post("/auth/login")
async def auth_login(req: LoginRequest, request: Request, response: Response):
    """Authenticate with username + password, issue a session cookie.

    Returns the user record and scopes. Generic 401 on bad credentials
    (does not reveal which field was wrong). Rate-limited per
    (username, client_ip) via a Postgres sliding-window counter: after
    TREELOOM_LOGIN_MAX_ATTEMPTS failures within TREELOOM_LOGIN_WINDOW_SECONDS,
    further attempts get a 429 + Retry-After before bcrypt even runs.
    """
    from treeloom.adapters.authorization.user_store import verify_password, hash_api_key
    from treeloom.domain.authorization import login_throttle
    import uuid as _uuid

    client_ip = authz._login_client_ip(request)

    # Throttle BEFORE the expensive bcrypt verify so a flood can't burn CPU.
    # Atomically record this attempt as a failure and read the post-insert
    # counts in one serialized round-trip (CWE-362): recording + counting share
    # a transaction guarded by a per-username advisory lock, so N concurrent
    # requests can no longer all observe the same pre-increment count and bypass
    # the threshold. A successful login clears this row below, so legitimate
    # logins are not penalised. Enforce two ceilings and lock if EITHER trips
    # (CWE-307): the per-(username, client_ip) counter bounds a single source,
    # and the per-username counter across ALL IPs bounds a distributed brute
    # force via IP rotation.
    # Returns PRIOR failure counts and records this attempt only if it is not
    # already locked — recording while locked would refresh the sliding window
    # on every rejected request and make the lockout permanent.
    try:
        failures, username_failures = (
            await _state._login_attempt_store.atomic_record_and_count(
                req.username,
                client_ip,
                LOGIN_WINDOW_SECONDS,
                LOGIN_MAX_ATTEMPTS,
                LOGIN_MAX_ATTEMPTS_PER_USERNAME,
            )
        )
    except LoginThrottleContention:
        # More simultaneous attempts on this username than the serializing
        # lock could drain — a flood by definition. Refusing here is both the
        # correct answer and what stops the login path from holding a pool
        # connection per waiter and starving every other endpoint.
        logger.warning(
            "Login lock contention: username=%s client_ip=%s — refusing",
            req.username, client_ip,
        )
        raise HTTPException(
            429,
            "Too many failed login attempts — try again later",
            headers={"Retry-After": str(LOGIN_WINDOW_SECONDS)},
        )
    per_ip_locked = login_throttle.is_locked(failures, LOGIN_MAX_ATTEMPTS)
    per_user_locked = login_throttle.is_locked(
        username_failures, LOGIN_MAX_ATTEMPTS_PER_USERNAME
    )
    if per_ip_locked or per_user_locked:
        retry_after = max(
            login_throttle.retry_after_seconds(
                failures, LOGIN_WINDOW_SECONDS, LOGIN_MAX_ATTEMPTS
            ),
            login_throttle.retry_after_seconds(
                username_failures, LOGIN_WINDOW_SECONDS, LOGIN_MAX_ATTEMPTS_PER_USERNAME
            ),
        )
        logger.warning(
            "Login rate-limit hit: username=%s client_ip=%s failures=%d "
            "username_failures=%d window_s=%d retry_after_s=%d "
            "per_ip_locked=%s per_user_locked=%s",
            req.username, client_ip, failures, username_failures,
            LOGIN_WINDOW_SECONDS, retry_after, per_ip_locked, per_user_locked,
        )
        raise HTTPException(
            429,
            "Too many failed login attempts — try again later",
            headers={"Retry-After": str(retry_after)},
        )

    # The attempt is already recorded as a failure by atomic_record_and_count
    # above, so these branches must NOT record again (double-counting) — they
    # only reject. A successful login clears the pre-recorded row below.
    user = await _state._user_store.get_by_username(req.username)
    if user is None or not user.password_hash:
        # Do the bcrypt work anyway. Returning here without it made a
        # non-existent username answer in microseconds while a real one took
        # the full KDF cost — a timing difference of two or more orders of
        # magnitude, measurable over the network on a single request, which
        # turns this endpoint into a username oracle (CWE-208). The 401 text
        # was already identical; the clock was the tell.
        await asyncio.to_thread(verify_password, req.password, authz._dummy_password_hash())
        raise HTTPException(401, "Invalid username or password")
    # Off-thread, same as the decoy above. Moving only the decoy — which is
    # what the timing fix did — left the REAL verify blocking the event loop
    # for the duration of a bcrypt round (~168ms measured at this cost
    # factor) on every successful-username login. That is the same CWE-400
    # class fixed for parse_and_chunk, reintroduced on the auth path, and
    # it is the branch an attacker can drive hardest.
    if not await asyncio.to_thread(
        verify_password, req.password, user.password_hash
    ):
        raise HTTPException(401, "Invalid username or password")

    # Issue a session
    raw_token = _uuid.uuid4().hex + _uuid.uuid4().hex
    token_hash = hash_api_key(raw_token)
    from datetime import timedelta
    expires_at = datetime.now(timezone.utc) + timedelta(hours=_SESSION_TTL_HOURS)
    session = await _state._session_store.create(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    if session is None:
        raise HTTPException(503, "Database unavailable — cannot create session")

    max_age = _SESSION_TTL_HOURS * 3600
    response.set_cookie(
        "treeloom_session",
        raw_token,
        httponly=True,
        samesite="lax",
        secure=authz._cookie_secure(request),
        max_age=max_age,
    )
    # Successful login: record success and clear the failure window so the
    # user's earlier typos don't count against their next login.
    await _state._login_attempt_store.record(req.username, client_ip, succeeded=True)
    await _state._login_attempt_store.clear(req.username, client_ip)

    scopes = sorted(s.value for s in scopes_for_role(user.role))
    return {"user": _user_to_response(user), "scopes": scopes}


@router.post("/auth/logout", status_code=204)
async def auth_logout(request: Request, response: Response):
    """Clear the session cookie and delete the server-side session."""
    from treeloom.adapters.authorization.user_store import hash_api_key

    cookie_val = request.cookies.get("treeloom_session")
    deleted = True
    if cookie_val:
        try:
            token_hash = hash_api_key(cookie_val)
            await _state._session_store.delete(token_hash)
        except Exception:
            # Swallowing this returned 204 while the session row stayed valid:
            # the cookie vanished from the browser, the user believed they had
            # logged out, and anyone holding that token — a shared machine, a
            # proxy log — could keep using it until its TTL expired (CWE-613).
            logger.exception("logout: server-side session delete failed")
            deleted = False

    # Clear the cookie either way: the browser in front of us should stop
    # sending the token even when the server-side row survives.
    response.delete_cookie("treeloom_session")
    if not deleted:
        raise HTTPException(
            503,
            "Logged out of this browser, but the server-side session could "
            "NOT be revoked and remains valid. Retry, or revoke it as an admin.",
        )
    return None


@router.get("/auth/me")
async def auth_me(request: Request):
    """Return the current authenticated caller's info.

    Requires authentication (middleware enforces this for non-public paths).
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Authentication required")
    caller_scopes = getattr(request.state, "scopes", set())
    # Determine auth method (best-effort heuristic for informational use)
    if request.cookies.get("treeloom_session"):
        auth_method = "session"
    elif request.headers.get("Authorization", "").startswith("Bearer "):
        auth_method = "bearer"
    else:
        auth_method = "unknown"
    return {
        "user": _user_to_response(user),
        "scopes": sorted(caller_scopes),
        "auth_method": auth_method,
    }


@router.post("/users/{user_id}/set-password", status_code=204)
async def set_password(user_id: str, req: SetPasswordRequest, request: Request):
    """Set (or reset) a user's local-login password (self or admin).

    Hashes the password with bcrypt and stores only the hash.
    """
    from treeloom.adapters.authorization.user_store import (
        BCRYPT_MAX_PASSWORD_BYTES,
        PasswordTooLongError,
        hash_password,
        verify_password,
    )

    current_user = getattr(request.state, "user", None)
    if current_user is None:
        raise HTTPException(401, "Authentication required")
    is_self = current_user.id == user_id
    if not is_self and not authz._acting_as_admin(request):
        raise HTTPException(403, "Cannot set another user's password")

    # Self-service change requires re-authentication with the current
    # password (CWE-620). Two exemptions: admins resetting another user's
    # password (already gated above), and a user setting their FIRST password
    # (no existing hash to verify — e.g. an API-key/bootstrapped account
    # enabling local login for the first time). The caller is already
    # authenticated as themselves, so an initial set is safe.
    if is_self:
        fresh = await _state._user_store.get_by_id(user_id)
        if fresh is None:
            raise HTTPException(404, f"User not found: {user_id}")
        if fresh.password_hash and not verify_password(
            req.current_password, fresh.password_hash
        ):
            raise HTTPException(403, "Current password is incorrect")

    try:
        pw_hash = hash_password(req.password)
    except PasswordTooLongError as exc:
        # 400, not 500: the caller can fix this, and silently truncating was
        # the old behaviour we are replacing.
        raise HTTPException(400, str(exc))
    success = await _state._user_store.update_password(user_id, pw_hash)
    if not success:
        raise HTTPException(404, f"User not found: {user_id}")

    # Invalidate all of the user's existing sessions so a stolen session
    # can't outlive the password change. PATs/API keys are independent
    # long-lived credentials and are left intact (revoke them explicitly).
    try:
        await _state._session_store.delete_for_user(user_id)
    except Exception:
        logger.exception("set_password: failed to clear sessions for %s", user_id)
    return None


@router.put("/users/{user_id}/role")
async def set_user_role(user_id: str, req: SetRoleRequest, request: Request):
    """Update a user's role (admin only)."""
    authz._require_admin(request)
    role = Role.ADMIN if req.role == "admin" else Role.USER
    target = await _state._user_store.get_by_id(user_id)
    if target is None:
        raise HTTPException(404, f"User not found: {user_id}")
    await _state._user_store.update_role(user_id, role)
    return {"id": user_id, "role": role.value}


@router.get("/users/{user_id}/tokens")
async def list_tokens(user_id: str, request: Request):
    """List PATs for a user (self or admin)."""
    current_user = getattr(request.state, "user", None)
    if current_user is None:
        raise HTTPException(401, "Authentication required")
    if current_user.id != user_id and not authz._acting_as_admin(request):
        raise HTTPException(403, "Cannot list another user's tokens")

    tokens = await _state._token_store.list_for_user(user_id)
    return [_pat_to_response(t) for t in tokens]


@router.post("/users/{user_id}/tokens", status_code=201)
async def create_token(user_id: str, req: CreateTokenRequest, request: Request):
    """Create a PAT for a user (self or admin).

    Scopes may not exceed the owner's role-derived scopes.
    Returns the token value once (not recoverable later).
    """
    import uuid as _uuid

    current_user = getattr(request.state, "user", None)
    if current_user is None:
        raise HTTPException(401, "Authentication required")
    if current_user.id != user_id and not authz._acting_as_admin(request):
        raise HTTPException(403, "Cannot create tokens for another user")

    # Resolve the owner to check their role
    owner = await _state._user_store.get_by_id(user_id)
    if owner is None:
        raise HTTPException(404, f"User not found: {user_id}")

    # Reject scopes exceeding owner's role
    allowed_scopes = {s.value for s in scopes_for_role(owner.role)}
    requested = set(req.scopes)
    excess = requested - allowed_scopes
    if excess:
        raise HTTPException(
            403, f"Scopes exceed owner's permissions: {', '.join(sorted(excess))}"
        )

    from treeloom.adapters.authorization.user_store import hash_api_key
    raw_token = _uuid.uuid4().hex + _uuid.uuid4().hex
    token_hash = hash_api_key(raw_token)

    pat = await _state._token_store.create(
        user_id=user_id,
        name=req.name,
        token_hash=token_hash,
        scopes=list(req.scopes),
        expires_at=req.expires_at,
    )
    if pat is None:
        raise HTTPException(503, "Database unavailable")

    data = _pat_to_response(pat)
    data["token"] = raw_token  # Shown once
    return data


@router.delete("/users/{user_id}/tokens/{token_id}", status_code=204)
async def revoke_token(user_id: str, token_id: str, request: Request):
    """Revoke a PAT (self or admin)."""
    current_user = getattr(request.state, "user", None)
    if current_user is None:
        raise HTTPException(401, "Authentication required")
    if current_user.id != user_id and not authz._acting_as_admin(request):
        raise HTTPException(403, "Cannot revoke another user's tokens")

    try:
        success = await _state._token_store.revoke(token_id, user_id)
    except RevocationUnavailable:
        logger.exception("PAT revocation failed for %s", token_id)
        raise HTTPException(
            503, "Revocation failed — the token is still ACTIVE. Retry."
        )
    if not success:
        raise HTTPException(404, f"Token not found: {token_id}")
    return None


@router.get("/api-keys")
async def list_api_keys(request: Request):
    """List all API keys (admin only)."""
    authz._require_admin(request)
    keys = await _state._api_key_store.list_keys()
    return [_api_key_to_response(k) for k in keys]


@router.post("/api-keys", status_code=201)
async def create_api_key(req: CreateApiKeyRequest, request: Request):
    """Create a new API key (admin only).

    Returns the raw key value once (not recoverable later).
    """
    import uuid as _uuid

    authz._require_admin(request)
    current_user = getattr(request.state, "user", None)
    created_by = current_user.id if current_user else None

    from treeloom.adapters.authorization.user_store import hash_api_key
    raw_token = _uuid.uuid4().hex + _uuid.uuid4().hex
    token_hash = hash_api_key(raw_token)

    key = await _state._api_key_store.create(
        name=req.name,
        token_hash=token_hash,
        scopes=list(req.scopes),
        all_access=req.all_access,
        created_by=created_by,
        expires_at=req.expires_at,
    )
    if key is None:
        raise HTTPException(503, "Database unavailable")

    data = _api_key_to_response(key)
    data["key"] = raw_token  # Shown once
    return data


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(key_id: str, request: Request):
    """Revoke an API key (admin only)."""
    authz._require_admin(request)
    try:
        success = await _state._api_key_store.revoke(key_id)
    except RevocationUnavailable:
        # Must not surface as 404: an operator revoking a key they believe is
        # compromised would read "not found" as "already gone" and stop, while
        # the key keeps working.
        logger.exception("API key revocation failed for %s", key_id)
        raise HTTPException(
            503, "Revocation failed — the key is still ACTIVE. Retry."
        )
    if not success:
        raise HTTPException(404, f"API key not found: {key_id}")
    return None


@router.post("/groups", status_code=201)
async def create_group(req: CreateGroupRequest, request: Request):
    """Create an authorization group (admin only)."""
    authz._require_admin(request)
    group = await _state._group_store.create_group(req.name, all_access=req.all_access)
    if group is None:
        raise HTTPException(503, "Database unavailable or duplicate group name")
    return _group_to_response(group)


@router.get("/groups")
async def list_groups(request: Request):
    """List authorization groups (admin only)."""
    authz._require_admin(request)
    return [_group_to_response(g) for g in await _state._group_store.list_groups()]


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(group_id: str, request: Request):
    """Delete a group and its memberships (admin only)."""
    authz._require_admin(request)
    if not await _state._group_store.delete_group(group_id):
        raise HTTPException(404, f"Group not found: {group_id}")
    return None


@router.patch("/groups/{group_id}")
async def set_group_all_access(
    group_id: str, req: SetAllAccessRequest, request: Request
):
    """Toggle a group's all_access flag (admin only)."""
    authz._require_admin(request)
    if not await _state._group_store.set_all_access(group_id, req.all_access):
        raise HTTPException(404, f"Group not found: {group_id}")
    return {"id": group_id, "all_access": req.all_access}


@router.get("/groups/{group_id}/members")
async def list_group_members(group_id: str, request: Request):
    """List the user ids in a group (admin only)."""
    authz._require_admin(request)
    return {"group_id": group_id, "members": await _state._group_store.list_members(group_id)}


@router.post("/groups/{group_id}/members", status_code=201)
async def add_group_member(
    group_id: str, req: GroupMemberRequest, request: Request
):
    """Add a user to a group (admin only)."""
    authz._require_admin(request)
    if await _state._group_store.get_by_id(group_id) is None:
        raise HTTPException(404, f"Group not found: {group_id}")
    if not await _state._group_store.add_member(group_id, req.user_id):
        raise HTTPException(503, "Database unavailable")
    return {"group_id": group_id, "user_id": req.user_id}


@router.delete("/groups/{group_id}/members/{user_id}", status_code=204)
async def remove_group_member(group_id: str, user_id: str, request: Request):
    """Remove a user from a group (admin only)."""
    authz._require_admin(request)
    if not await _state._group_store.remove_member(group_id, user_id):
        raise HTTPException(404, "Membership not found")
    return None
