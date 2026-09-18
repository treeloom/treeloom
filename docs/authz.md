# Authorization model & data flow

This document describes how Treeloom decides who may read which repository's
code: per-repo authorization for **read** paths (search and graph traversal),
the identity model (users, groups, sessions, tokens), and the data flow from
an MCP client to the vector and graph stores. It is written at the level of
detail a security reviewer needs; operators who only want to turn auth on and
mint a token can start at [Configuration](#configuration) and
[Local authentication, sessions, and tokens](#local-authentication-sessions-and-tokens).

None of this is enforced unless `AUTH_ENABLED=true`. With auth off, every
endpoint is open and every caller sees every source.

## Threat model

A shared index holds chunks + a code graph for many repositories. Some are
restricted (payments, security, M&A). The risk: a human or agent query
retrieving a chunk, graph neighbor, or symbol definition from a repo the
requesting principal is not allowed to see — a cross-tenant data-exfiltration
finding. The goal: **a principal without access to repo X never receives any
content derived from X via any read endpoint or MCP tool.**

## Design choice: scope-level rejection, not per-hit filtering

Rather than retrieve a fused cross-repo result set and scrub unauthorized hits
(which must also scrub graph neighbors — easy to get wrong), authorization is
enforced at the **query scope**:

- A query pinned to one source (`source_id=X`) is **rejected (403)** unless the
  caller is authorized for X. Every returned chunk/neighbor/definition is then
  necessarily from X.
- A shared / cross-repo query (no `source_id` — `cross_repo=true` or a
  `path_prefix` spanning sources) is **rejected (403)** unless the caller has
  `all_access`. The sources they are explicitly *denied* are returned as an
  exclusion set and pushed down as a `source_id NOT IN (...)` prefilter.

## Principals, grants, decision

A **principal** is the user plus each group they belong to:
`("user", user.id)` and `("group", gid)` for each `gid` in `user.group_ids`.

A **grant** is `(principal_type, principal_id, source_id, effect)` where
`effect ∈ {allow, deny}` (`source_grants` table). `all_access` is a boolean on
both `users` and `groups`; the user's *effective* `all_access` is the OR of
their own flag and any of their groups' (computed in the user-store hydration
JOIN).

**Single-source decision** (`decide_single_source`, pure):

```
admin            → allow         # Role.ADMIN bypasses everything
deny grant       → deny          # deny beats all_access (the CI exclusion case)
all_access       → allow
allow grant      → allow
owner            → allow         # source_records.created_by == user.id
otherwise        → deny (403)
```

**Shared-index decision** (`decide_shared`, pure):

```
admin            → allow, no exclusions
not all_access   → deny (403)
all_access       → allow, exclude = sources with a deny grant for any principal
```

The pure functions live in `src/treeloom/domain/authorization/access.py` and are
exhaustively unit-tested (`tests/unit/test_authorization_service.py`). The async
`AuthorizationService` only resolves inputs (grants, owner) and applies them.

## Enforcement point & data flow

1. MCP client → MCP server tool. There are **two distinct credential
   policies**, split by tool kind (`application/mcp_server.py`):
   - **Write/admin tools** (`index_*`, `remove_indexed_source`,
     `rebuild_all_graphs`, `source_staleness`, …) call
     `_tool_auth_headers(ctx)`: caller's own bearer (API key,
     via `_caller_auth_headers`) → `TREELOOM_MCP_TOKEN` (the caller's own
     credential supplied via the environment — the stdio credential model)
     → the gated service account (`TREELOOM_MCP_API_KEY`, only when
     `TREELOOM_MCP_SERVICE_ACCOUNT=true`) → nothing. Off by default; on a
     SHARED transport (the opt-in HTTP+SSE container) it re-creates a
     confused deputy if opted in.
   - **Read tools** — the nine per-repo-scoped ones: `search_code`,
     `search_code_enhanced`, `hydrate_chunks`, `explain_code`,
     `graph_explore`, `find_definition`, `find_callers`, `find_references`,
     `list_job_errors` — call `_read_tool_auth_headers(ctx)` instead: the
     same precedence MINUS the service-account step. This is deliberate and
     must never be "simplified" to call `_tool_auth_headers`: per-repo
     authorization below is enforced against the caller's own
     `request.state.user`, so if reads could fall back to the shared
     service account, an operator who opts into
     `TREELOOM_MCP_SERVICE_ACCOUNT=true` for write convenience would
     silently grant every caller on that transport the service account's
     *cross-repo* visibility instead of their own scoped grants — reopening,
     on the read path, the exact confused-deputy class the write-path policy
     exists to prevent. `TREELOOM_MCP_TOKEN` is safe on the read path
     because it represents the caller's OWN credential (just delivered via
     the environment instead of an HTTP header), so scoping below still
     applies to it exactly as it would to a forwarded bearer.
   In both cases: the shared service key is never substituted for a real
   caller, and nothing is forwarded if the caller sent nothing (no silent
   escalation). Regression coverage:
   `tests/unit/test_mcp_credential_forwarding.py`, including
   `test_read_tool_never_uses_service_account_even_when_opted_in`
   parametrized over all nine read tools (a first implementation pass
   routed all nine through the write-tool policy and passed every test that
   existed at the time, because only `search_code` had a guard).
2. Indexer read endpoint (`src/treeloom/application/indexer_service.py`) calls
   `_require_scope_xor` then **`_authorize_scope(request, source_id, action)`**
   (`src/treeloom/application/indexer_authz.py`) — the single choke point.
   - If `AUTH_ENABLED != true`, or `TREELOOM_SEARCH_OPEN=1` is set: returns
     `None`, no checks — open / single-tier mode.
   - Else authenticates the caller and runs the decision above; raises
     `401` (anonymous or bad credential) / `403` (denied) or returns the
     exclusion list.

   `POST /search` is listed in the auth middleware's `PUBLIC_PATHS`, so it
   skips the middleware and relies entirely on this in-endpoint check (that
   is what makes `TREELOOM_SEARCH_OPEN` possible). `/graph-explore` and the
   `/find-*` endpoints pass through the middleware first (which already
   rejects anonymous callers when auth is on) and then run the same
   `_authorize_scope` check.
3. The scope (`source_id` and/or `exclude_source_ids`) threads through
   `application/retrieval.py:graph_search` into:
   - the vector store (`milvus`/`lancedb` `_build_filter` → `source_id ==`
     and/or `NOT IN`; `chromadb` applies the `source_id` equality and **fails
     closed** on exclusions it cannot express);
   - the graph store neighbor/traversal/caller/reference queries (neo4j +
     sqlite), scoped via the `(:Source)-[:CONTAINS]->(:Entity)` relationship /
     `source_entities` join — closing the cross-source neighbor leak.
4. Each decision is recorded best-effort to `search_audit` (who, groups, action,
   scope, source, allow/deny). Admin read: `GET /audit/search`.

## Identity

- Bearer credentials — personal access tokens (`personal_access_tokens` table)
  and API keys (`api_keys` table), both HMAC-SHA256 hashed, plus the legacy
  `users.api_key_hash` — and browser sessions (all detailed below). Groups,
  grants, and `all_access` are managed locally via admin REST (`/groups*`,
  `/sources/{id}/grants`, `PATCH /users/{id}`, `PATCH /groups/{id}`).

## Configuration

| Var | Purpose |
|-----|---------|
| `AUTH_ENABLED` | Master switch. `false` → fully open (no enforcement). |
| `TREELOOM_SEARCH_OPEN` | With auth on, keep **search** anonymous (protect indexing only). |

## Local authentication, sessions, and tokens

### Login / logout / me

`POST /auth/login` accepts `{username, password}`. On success it creates a
session, sets an **HttpOnly, SameSite=Lax** cookie named `treeloom_session`
(the raw opaque token, 64 hex chars), and returns `{user, scopes}`. The raw
token is HMAC-SHA256 hashed server-side and stored in the `sessions` table;
the cookie value is never persisted directly. `POST /auth/logout` reads the
cookie, deletes the matching `sessions` row, and clears the cookie (204). Both
endpoints are exempt from auth middleware so they can bootstrap a session.
`GET /auth/me` is protected and returns `{user, scopes, auth_method}` for the
current principal.

### Login rate limiting

`POST /auth/login` is brute-force protected by a **Postgres-backed sliding
window** keyed on `(username, client_ip)` — the counter lives in the
`login_attempts` table (migration `019`, store
`adapters/postgresql/login_attempt_store.py`), so the limit holds across all
indexer workers, not just one process. The decision is pure and unit-tested in
`domain/authorization/login_throttle.py`.

Flow inside the handler:

1. Resolve `client_ip` from `request.client.host`. `X-Forwarded-For` is
   trusted **only** when `TREELOOM_LOGIN_TRUST_FORWARDED_FOR=1` (set this only
   behind a trusted reverse proxy; otherwise a client could spoof the header
   to dodge the per-IP limit).
2. Count failed attempts in the last `TREELOOM_LOGIN_WINDOW_SECONDS`. If that
   count is `>= TREELOOM_LOGIN_MAX_ATTEMPTS`, return **429** with a
   `Retry-After` header (exponential backoff capped at the window length) —
   **before** the expensive bcrypt verify runs, so a flood can't burn CPU. A
   structlog warning is emitted on every lockout. The 429 body does not reveal
   which field was wrong.
3. On bad credentials, record a failure and return the existing generic
   **401**. On success, record a success and **clear** the `(username,
   client_ip)` window so a user's earlier typos don't count against their next
   login.

The store is **fail-loud**: if the Postgres pool is unavailable every method
raises `RuntimeError` rather than silently reporting "0 failures" (which would
re-open the brute-force window during a DB outage).

| Env var | Default | Meaning |
| --- | --- | --- |
| `TREELOOM_LOGIN_MAX_ATTEMPTS` | `5` | Failed logins per `(username, client_ip)` before 429 |
| `TREELOOM_LOGIN_MAX_ATTEMPTS_PER_USERNAME` | `50` | Failed logins per username across **all** IPs (defeats IP rotation; keep well above the per-IP limit so users behind one NAT aren't locked out) |
| `TREELOOM_LOGIN_WINDOW_SECONDS` | `300` | Sliding-window length in seconds |
| `TREELOOM_LOGIN_TRUST_FORWARDED_FOR` | `0` | `1` → key the limiter on `X-Forwarded-For` (trusted-proxy only) |

### Two token types

**Personal Access Tokens (PATs)** are user-owned, managed via
`GET/POST/DELETE /users/{user_id}/tokens`. They live in the
`personal_access_tokens` table: HMAC-hashed, optionally expiring, revocable,
and carry a `scopes` array (`search`, `index`, `admin`). A PAT's scopes may
not exceed the owner's role — an admin can hold admin-scoped tokens; a
non-admin user is limited to `{search, index}`. The token value is shown
**once** on creation and never again.

**API keys** are service/admin-owned, managed via
`GET/POST/DELETE /api-keys` (admin only). They live in the `api_keys` table
with the same HMAC/expiry/revoke mechanics plus an `all_access` flag (mirrors
the user-level flag for shared-index authorization) and a `created_by`
reference. Scopes behave identically to PATs; `all_access` is honored by the
authorization layer.

### Auth resolution order

`AuthMiddleware` tries each method in order and stops at the first match:

1. **Session cookie** `treeloom_session` — HMAC the value, look up in
   `sessions`; load the owner; scopes = `scopes_for_role(user.role)` (browser
   sessions always carry the user's full role scopes). Best-effort
   `sessions.last_seen_at` update on each hit. Expired or missing → fall
   through.
2. **Bearer token** (`Authorization: Bearer <token>`):
   a. HMAC → `personal_access_tokens.token_hash` (PAT). On hit, load owner;
      scopes = the **token's** scopes, not the role's. The resolved principal's
      **effective role is derived from the token's scopes** (`admin` scope →
      ADMIN, else USER) on a copy of the owner — see "Token data access follows
      the token's scopes" below. Best-effort `last_used_at` update.
   b. HMAC → `api_keys.token_hash`. On hit, build principal from
      `created_by` if set, else a synthetic service user; scopes = the key's
      scopes; `all_access` forwarded to the authorization layer. For an
      owner-bound key the principal's **effective role is likewise derived from
      the key's scopes**, not the owner's role.
   c. Legacy fallback: `users.api_key_hash` (existing path). Scopes =
      `scopes_for_role(user.role)`.
3. No match → 401.

`/auth/login` and `/auth/logout` bypass the middleware entirely
(`PUBLIC_PATHS`). `AUTH_ENABLED=false` and dev-mode loopback behavior are
unchanged.

### Token data access follows the token's scopes, not the owner's role

A token's (or API key's) **effective data-access privilege follows the token's
scopes, not its creator's role.** When an admin mints a `search`-scoped key or
PAT, the resolved principal authenticates with effective role **USER**, not
ADMIN — so it does **not** inherit the `access.py` admin see-everything bypass
and is instead governed by `all_access` + per-source grants like any other
non-admin caller. Only a token that carries the `admin` scope resolves to role
ADMIN.

`_derive_scoped_principal` (`application/auth_middleware.py`) implements this:
it returns a `dataclasses.replace` **copy** of the owner with
`role = ADMIN if "admin" in scopes else USER` (the cached owner object is never
mutated), preserving the owner's `id` (so the source-owner-grant rule
`owner_id == user.id` still matches) and the owner's `all_access` flag (so
`all_access` + grants compose for the derived principal). It applies to both the
owner-bound API-key branch and the PAT branch. Session, legacy
`users.api_key_hash` paths are unchanged (sessions and legacy keys
carry full role scopes by design).

### Legacy api_key_hash migration

Migration `016_local_auth_tokens.sql` inserts one PAT row per existing user
with a non-empty `api_key_hash` (name `legacy`, token_hash copied verbatim,
scopes derived from role, `ON CONFLICT DO NOTHING`). The `users.api_key_hash`
column and the `get_by_api_key_hash` code path remain as the final auth
fallback, so existing deployments that set `TREELOOM_ADMIN_KEY` continue to
authenticate without any migration action.

### First-admin bootstrap

If `users` is empty at indexer startup and both `TREELOOM_BOOTSTRAP_ADMIN_USER`
and `TREELOOM_BOOTSTRAP_ADMIN_PASSWORD` are set, an admin user with a bcrypt
`password_hash` is created automatically. This is best-effort, logged (the
password is never logged), and idempotent (the `count_users()==0` guard means
it only fires once).

### Scope gate

`_require_scope(request, scope)` passes only when `scope in
request.state.scopes`. There is deliberately **no** `Role.ADMIN` bypass: that
bypass made per-token scopes meaningless for admin-owned tokens, since a
search-only PAT belonging to an admin could reach any endpoint. Full-role auth
paths (session, legacy API key, dev synthetic admin) populate the scope
set from `scopes_for_role`, so a genuine admin session still carries every
scope and passes; only scope-limited PAT and API-key principals are
constrained, which is the point. When `AUTH_ENABLED=false` the gate is a no-op.

**Where it is applied:** `DELETE /sources/{source_id}` and `POST /index-graph`
require `index`; `POST /search` with `return_pool=true` (an operator/mining
switch that returns the whole pre-rerank pool) requires `admin`. The remaining
mutating index endpoints (`/index-repo`, `/index-directory`, `/index-file`)
authenticate but do not yet consult a scope, so a search-only PAT can still
drive them. Admin endpoints are gated by `_require_admin`, which checks role
*and* the `admin` scope separately.

### Dashboard management

All of the above — local users, passwords, PATs, API keys, and sessions — is
accessible from the dashboard's **Users & Groups** tab (user management, role
assignment, password reset) and **Access Tokens** tab (PAT and API-key
create/revoke). Raw token values appear only at creation time.

### Schema (migration 016)

`016_local_auth_tokens.sql` adds `password_hash TEXT` (nullable) to `users`
and creates three new tables: `sessions` (id, user_id, token_hash UNIQUE,
created_at, expires_at, last_seen_at), `personal_access_tokens` (id, user_id,
name, token_hash UNIQUE, scopes TEXT[], created_at, expires_at, last_used_at,
revoked, UNIQUE(user_id, name)), and `api_keys` (id, name UNIQUE, token_hash
UNIQUE, scopes TEXT[], all_access, created_by, created_at, expires_at,
last_used_at, revoked). All three have indexes on `token_hash` for fast bearer
lookups. The migration is idempotent (`IF NOT EXISTS` / `ON CONFLICT`) and
auto-applies on indexer startup alongside the other migrations.

## Known limitations

- `path_prefix` (not source-pinned) is treated as a shared-index query and
  requires `all_access`. Resolving a prefix to its owning source(s) for
  finer-grained authorization is future work.
- `VECTOR_STORE=chromadb` cannot enforce shared-index exclusions and raises;
  use `milvus` or `lancedb` for multi-tier deployments.
