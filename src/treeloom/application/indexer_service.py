import asyncio
import inspect
import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Any

logger = logging.getLogger(__name__)

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from git import Repo
from pydantic import BaseModel, Field

from treeloom.embedder import MAX_BATCH_SIZE, embed, embed_query
from treeloom.infrastructure import metrics
from treeloom.infrastructure.config import (  # single source of truth
    FEATURE_SKIP_PATTERNS,
    LOG_INTERVAL,
    OTEL_TRACE_PER_CHUNK,
    SKIP_FILENAMES,
    USE_SUMMARY_VECTOR,
)
from treeloom.infrastructure.fileio import read_source
from treeloom.infrastructure.tracing import get_tracer

tracer = get_tracer("treeloom.indexer")

# Optional: emit a span per chunk that misses the summary cache. Hundreds
# of these per file balloon trace size, so default off — flip on when
# actively hunting a per-chunk LLM bottleneck.



from treeloom.indexer import (
    CodeIndexer,
    detect_language,
    SUPPORTED_EXTENSIONS,
)
from treeloom.retriever import init_collection, insert_chunks, delete_chunks_by_source
from treeloom.retriever import graph_search, hydrate_chunks
from treeloom.graph_extractor import extract_graph
from treeloom.sources import make_source_id
from treeloom import graph_store
from treeloom import community
from treeloom import llm

# Webhook adapters and domain
from treeloom.domain.webhook import WebhookPayload, Provider, detect_provider
from treeloom.adapters.webhook.github import GitHubAdapter
from treeloom.adapters.webhook.gitea import GiteaAdapter
from treeloom.retriever import delete_chunks_by_file
from treeloom.graph_store import delete_entities_by_file

# PostgreSQL source persistence (graceful degradation when unavailable)
from treeloom.domain.sources import SourceRecord
from treeloom.adapters.sources.repository import PostgreSourceRepository

# Fleet-level aggregation + refresh selection (pure)
from treeloom.application.fleet import (
    build_fleet_health,
    resolve_reindex_kind,
    select_sources_to_refresh,
)


app = FastAPI(title="Treeloom Indexer")

# Shared mutable state — see that module on why it is used via the module
# object rather than by importing the names.
from treeloom.application import indexer_state as _state

# Job execution lives in indexer_runners (stage 2/6). Re-exported here so
# existing callers and tests keep their import paths.
# Authorization and redaction live in indexer_authz (stage 3/6). Imported as
# a MODULE, never by name: a re-exported name has two bindings and a patch
# reaches only one of them, depending on whether the caller is a route here
# or a helper there. Qualifying leaves one target.
from treeloom.application import indexer_authz as authz

# Auth / identity endpoints live in routes_auth (stage 4/6).
from treeloom.application import routes_auth as _routes_auth
from treeloom.application import routes_webhook as _routes_webhook

# Lifecycle hooks live in lifecycle (stage 6/6). Registered explicitly
# below rather than re-decorated, so the binding is a visible call.
from functools import partial as _partial
from treeloom.application import lifecycle as _lifecycle


# Job execution lives in indexer_runners. Imported as a MODULE: a re-exported
# name has its own binding here, so patching one place and reading the other
# silently diverge — the same rule as indexer_state and indexer_authz.
from treeloom.application import indexer_runners as runners

# Public re-export: treeloom.indexer_service (the root shim) exposes `indexer`.
# The instance is OWNED by the runners leaf -- module-qualified here so there
# is exactly one chunker, not one per layer.
indexer = runners.indexer
# CORS for the operator UI v2 SPA (cross-origin). Driven by TREELOOM_CORS_ORIGINS;
# empty = permissive dev default. See treeloom.application.cors for why a wildcard
# origin must not be combined with credentials (it silently breaks cookie auth).
from treeloom.application.cors import add_cors_middleware

add_cors_middleware(app, os.environ.get("TREELOOM_CORS_ORIGINS"))

# ── Auth middleware (US2) ──────────────────────────────────────────
# GET /sources and GET /jobs are public; POST/DELETE mutations require auth.
from treeloom.application.auth_middleware import AuthMiddleware
from treeloom.adapters.authorization.api_key_store import RevocationUnavailable
from treeloom.adapters.authorization.grant_store import GrantLookupUnavailable
from treeloom.adapters.authorization.user_store import PostgreSQLUserStore
from treeloom.adapters.authorization.group_store import PostgreSQLGroupStore
from treeloom.adapters.authorization.grant_store import PostgreSQLGrantStore
from treeloom.adapters.authorization.search_audit_store import PostgreSQLSearchAuditStore
from treeloom.adapters.authorization.token_store import PostgreSQLPersonalAccessTokenStore
from treeloom.adapters.authorization.api_key_store import PostgreSQLApiKeyStore
from treeloom.adapters.authorization.session_store import PostgreSQLSessionStore
from treeloom.domain.authorization.access import AuthorizationService
from treeloom.domain.search_audit import SearchAuditRecord


app.add_middleware(
    AuthMiddleware,
    user_store=_state._user_store,
    session_store=_state._session_store,
    token_store=_state._token_store,
    api_key_store=_state._api_key_store,
    skip_routes={"/sources": {"GET"}, "/jobs": {"GET"}, "/job-groups": {"GET"}},
)

app.include_router(_routes_auth.router)
app.include_router(_routes_webhook.router)
# startup/shutdown take the app as an argument (they read and write
# app.state); partial keeps them async-callable for FastAPI.
app.add_event_handler("startup", _partial(_lifecycle.startup, app))
app.add_event_handler("shutdown", _partial(_lifecycle.shutdown, app))


class IndexFileRequest(BaseModel):
    file_path: str
    force: bool = False
    skip_graph: bool = False


class IndexDirectoryRequest(BaseModel):
    directory: str
    pattern: str = "**/*"
    force: bool = False
    auto_preflight: bool = True
    # Feature-flagged: when TREELOOM_FEATURE_SKIP_PATTERNS=1 the indexer
    # honors this; otherwise it's accepted and ignored (preflight still
    # surfaces the same suggestions in the JobAck).
    skip_patterns: list[str] | None = None
    # Two-pass indexing: skip graph extraction during chunks+embeddings
    # pass. Run later via POST /index-graph with the resulting source_id.
    skip_graph: bool = False


class IndexRepoRequest(BaseModel):
    path: str | None = None
    url: str | None = None
    branch: str | None = None
    force: bool = False
    auto_preflight: bool = True
    skip_patterns: list[str] | None = None
    skip_graph: bool = False
    # Attach this job to an existing job_group ("Job") instead of creating a
    # group-of-1. Used by fleet onboard to group a whole manifest.
    group_id: str | None = None


class IndexGraphRequest(BaseModel):
    source_id: str


class PreflightRequest(BaseModel):
    path: str | None = None
    url: str | None = None
    branch: str | None = None
    concurrency: int | None = None


class JobAck(BaseModel):
    job_id: str
    status: str
    source_id: str
    warnings: list[dict] | None = None
    recommendations: dict | None = None


# ── User management models (US3) ───────────────────────────────────


# ── Local auth + token management models ─────────────────


class GrantRequest(BaseModel):
    principal_type: str = Field(..., pattern="^(user|group)$")
    principal_id: str = Field(..., min_length=1)
    effect: str = Field("allow", pattern="^(allow|deny)$")


# ── Internal helpers ───────────────────────────────────────────────


# Per-file error log. Lazy-initialized — Postgres pool may not be up yet
# when the first read happens, and write paths swallow pool-absent.
from treeloom.adapters.postgresql.job_file_error_store import JobFileErrorStore


# Login rate-limit sliding-window store. Fail-loud on no pool.
from treeloom.adapters.postgresql.login_attempt_store import (
    LoginAttemptStore,
    LoginThrottleContention,
)


async def _attach_group(
    job: dict, created_by: str | None, group_id: str | None = None
) -> None:
    """Attach a freshly-created Task to a job_group ("Job") and stamp
    ``job['group_id']``. No-op if the group store isn't initialized.

    * ``group_id`` given -> attach to that existing group (fleet onboard):
      validate it exists (400 otherwise), then bump its task_count.
    * ``group_id`` is None -> create a group-of-1 for this single submission.

    Only called on the NEW-job path — never on a dedup short-circuit (which
    returns an existing job that already has its own group_id).
    """

    if _state._job_group_store is None:
        return
    if group_id:
        if await _state._job_group_store.get(group_id) is None:
            raise HTTPException(400, f"Unknown group_id: {group_id}")
        job["group_id"] = group_id
        await _state._job_group_store.increment_task_count(group_id, 1)
        return
    job["group_id"] = await _state._job_group_store.create(
        label=job.get("source") or job.get("source_id") or job["id"],
        kind=job.get("kind") or "repo",
        created_by=created_by,
        task_count=1,
    )




# ── Retry / dead-letter config ────────────────────────────────
# Retryable job kinds: incremental webhook jobs. Full repo/graph/file/directory
# jobs are NOT auto-retried (operators are expected to re-submit via the API).
_RETRYABLE_KINDS = {"incremental"}
MAX_JOB_ATTEMPTS = int(os.environ.get("MAX_JOB_ATTEMPTS", "3"))




def _find_done_job_for_source(source_id: str) -> dict | None:
    """Return the most recent DONE in-memory job for source_id, if any.

    `_state._jobs` is hydrated from JobStore DONE rows on startup, so this answers
    the cross-restart question as long as that hydration ran.
    """
    if not source_id:
        return None
    candidates = [
        j for j in _state._jobs.values()
        if j.get("source_id") == source_id and j.get("status") == "done"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda j: j.get("finished_at") or 0)




# Feature flag for the skip_patterns request field. Off by default in v1
# so the preflight surface can land without changing indexer semantics;
# preflight still recommends patterns in the report regardless.






# ── job runners ──────────────────────────────────────────────────


# ── endpoints ────────────────────────────────────────────────────


class AddBackendRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    klass: str = Field(..., description="'gpu' or 'cpu'")


async def _run_preflight_for_path(path: str, *, concurrency: int | None = None) -> dict:
    """Run a preflight scan + estimation against a local path. Pure I/O,
    no auth needed; the data is filesystem-public to the indexer process."""
    from treeloom.preflight.scanner import walk_repo
    from treeloom.preflight.coefficients import load_coefficients, fallback_coefficients
    from treeloom.preflight.model import estimate_total
    from treeloom.preflight.recommender import build_report

    # The module-split find/replace rewrote the ENV VAR NAME to
    # "runners.INDEX_FILE_CONCURRENCY", which can never be set -- so preflight
    # silently always assumed 8 and its time estimates ignored the real value.
    cc = concurrency or runners.INDEX_FILE_CONCURRENCY
    stats, unsupported = await asyncio.to_thread(walk_repo, path)
    try:
        coefs = await load_coefficients()
    except Exception:
        coefs = fallback_coefficients()
    per_file, total = estimate_total(stats, coefs, file_concurrency=cc)
    return build_report(path, stats, unsupported, per_file, total).to_dict()


@app.post("/preflight")
async def preflight(req: PreflightRequest, request: Request):
    """Scan a path (or shallow-clone a URL) and return a preflight report.

    Requires authentication when `AUTH_ENABLED=true` (same posture as
    `/index-*`). When called with `req.path`, the path must resolve under
    an entry of `authz.PREFLIGHT_ALLOWED_ROOTS` — otherwise rejected as 403
    even for authenticated callers. This prevents using preflight as a
    filesystem-enumeration oracle for anything the indexer process can
    read. URL mode shallow-clones into a fresh tempdir and is always safe.
    """
    # AuthN — same Bearer flow as the sibling indexer endpoints
    await authz._require_authenticated_user(request)

    if not req.path and not req.url:
        raise HTTPException(400, "Provide a `path` or `url`")

    cleanup_path: str | None = None
    target = req.path

    if req.url:
        if not runners._is_safe_git_url(req.url):
            raise HTTPException(400, f"Unsafe git URL: {req.url!r}")
        if req.branch and req.branch.startswith("-"):
            raise HTTPException(400, f"Unsafe git branch: {req.branch!r}")
        target = tempfile.mkdtemp(prefix="treeloom_preflight_")
        cleanup_path = target
        kwargs: dict = {
            "depth": 1,
            "env": runners._git_env(),
            "allow_unsafe_protocols": False,
            "allow_unsafe_options": False,
        }
        if req.branch:
            kwargs["branch"] = req.branch
        Repo.clone_from(req.url, target, **kwargs)
    else:
        # Path mode: must resolve under an allowed root.
        expanded = os.path.expanduser(req.path or "")
        if not authz._path_is_under_allowed_root(expanded):
            raise HTTPException(
                403,
                "path is not under any authz.PREFLIGHT_ALLOWED_ROOTS entry; "
                "use `url` for arbitrary git sources, or configure "
                "authz.PREFLIGHT_ALLOWED_ROOTS on the indexer to whitelist roots.",
            )
        target = expanded

    try:
        if not target or not os.path.isdir(target):
            raise HTTPException(404, f"Path not found: {target!r}")
        return await _run_preflight_for_path(target, concurrency=req.concurrency)
    finally:
        if cleanup_path:
            shutil.rmtree(cleanup_path, ignore_errors=True)


@app.get("/embedding-backends")
async def list_embedding_backends(request: Request):
    """Registry rows + this process's live counters keyed by URL.

    The `local` block reflects only the calling worker process. Real
    cross-worker aggregation is not implemented yet.
    """
    authz._require_admin(request)
    store = getattr(app.state, "embedding_backend_store", None)
    proxy = getattr(app.state, "embedding_proxy", None)
    if store is None or proxy is None:
        raise HTTPException(503, "Embedding backend registry not initialized")

    rows = await store.list_all()
    local: dict[str, dict[str, int]] = {}
    for b in proxy._backends:
        local[b.url] = {
            "in_flight_tokens": b.in_flight_tokens,
            "failures": b.failures,
        }

    import socket as _socket
    return {
        "registry": [r.to_dict() for r in rows],
        "local": local,
        "process_id": f"{_socket.gethostname()}:{os.getpid()}",
    }


@app.post("/embedding-backends", status_code=201)
async def add_embedding_backend(req: AddBackendRequest, request: Request):
    authz._require_admin(request)
    if req.klass not in ("gpu", "cpu"):
        raise HTTPException(400, "klass must be 'gpu' or 'cpu'")
    if not (req.url.startswith("http://") or req.url.startswith("https://")):
        raise HTTPException(400, "url must start with http:// or https://")
    store = getattr(app.state, "embedding_backend_store", None)
    if store is None:
        raise HTTPException(503, "Embedding backend registry not initialized")
    try:
        row = await store.add(req.url, req.klass, created_by=_caller_id(request))
    except Exception as exc:
        # asyncpg raises UniqueViolationError on duplicate URL — surface as 409
        if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
            raise HTTPException(409, f"backend with url {req.url!r} already exists")
        raise
    return row.to_dict()


@app.delete("/embedding-backends/{backend_id}", status_code=204)
async def delete_embedding_backend(backend_id: str, request: Request):
    authz._require_admin(request)
    store = getattr(app.state, "embedding_backend_store", None)
    if store is None:
        raise HTTPException(503, "Embedding backend registry not initialized")
    from uuid import UUID as _UUID
    try:
        uid = _UUID(backend_id)
    except ValueError:
        raise HTTPException(400, "backend_id must be a UUID")
    deleted = await store.delete(uid)
    if not deleted:
        raise HTTPException(404, f"backend {backend_id} not found")
    return None


@app.get("/debug/tasks", include_in_schema=False)
async def debug_tasks():
    """Dump every live asyncio Task with its suspended stack.

    Public diagnostic endpoint. Returns the task graph as seen from
    inside the event loop — unlike py-spy, this shows coroutines that
    are awaiting I/O (TEI, LLM, Milvus, Neo4j, Postgres) where they're
    parked. Use it when py-spy reports the main thread idle but the
    indexer is making no visible progress.
    """
    out: list[dict] = []
    for t in asyncio.all_tasks():
        coro = t.get_coro()
        coro_name = getattr(coro, "__qualname__", None) or repr(coro)
        frames = []
        for f in t.get_stack():
            frames.append({
                "file": f.f_code.co_filename,
                "line": f.f_lineno,
                "name": f.f_code.co_name,
            })
        out.append({
            "name": t.get_name(),
            "coro": coro_name,
            "done": t.done(),
            "cancelled": t.cancelled() if t.done() else None,
            "stack": frames,
        })
    return {"task_count": len(out), "tasks": out}


@app.get("/health")
async def health():
    """FR-010: Health check with database status."""
    pool = None
    try:
        from treeloom.adapters.postgresql.connection import get_pool
        pool = await get_pool()
    except Exception:
        pass

    return {
        "status": "ok",
        "database": "connected" if pool else "unavailable",
        "auth_enabled": os.environ.get("AUTH_ENABLED", "").lower() == "true",
    }


@app.get("/metrics")
async def metrics_endpoint():
    from treeloom.infrastructure.metrics import get_metrics
    from fastapi.responses import Response
    return Response(content=get_metrics(), media_type="text/plain")


@app.get("/status")
async def status(request: Request):
    """Back-compat: returns the most recently created job's state."""
    if _state._last_job_id and _state._last_job_id in _state._jobs:
        redact = await authz._job_view_is_redacted(request)
        return authz._public_job(_state._jobs[_state._last_job_id], redact=redact)
    return {"status": "idle"}


@app.get("/jobs")
async def list_jobs(request: Request):
    """List jobs the caller is authorized to see.

    This returned every job in the process to every caller, so a non-admin
    could enumerate the absolute paths and clone URLs of every other
    principal's indexed sources plus their failure messages (CWE-639).
    """
    redact = await authz._job_view_is_redacted(request)
    allowed = await authz._visible_job_filter(request)
    jobs = list(_state._jobs.values())
    if allowed is not None:
        jobs = [j for j in jobs if await allowed(j.get("source_id"))]
    return [authz._public_job(j, redact=redact) for j in jobs]


class CreateJobGroupRequest(BaseModel):
    label: str
    kind: str = "fleet"


@app.post("/job-groups", status_code=201)
async def create_job_group(req: CreateJobGroupRequest, request: Request):
    """Create an empty Job (group) to attach subsequent index Tasks to.

    Used by fleet onboard so a whole manifest becomes one Job; Tasks
    bump its task_count as they attach via the `group_id` passthrough on
    /index-repo. Mutation -> requires auth when AUTH_ENABLED (not in skip_routes).
    """
    if _state._job_group_store is None:
        raise HTTPException(503, "Job group store unavailable")
    group_id = await _state._job_group_store.create(
        label=req.label,
        kind=req.kind or "fleet",
        created_by=_caller_id(request),
        task_count=0,
    )
    return {"id": group_id, "label": req.label, "kind": req.kind or "fleet"}


@app.get("/job-groups")
async def list_job_groups(request: Request):
    """List Jobs (submission groups), newest first, with rolled-up progress.

    Mirrors GET /jobs, INCLUDING its authorization: a group is summarised from
    its member jobs, so returning every group to every caller leaked the same
    source paths and error text by another route (CWE-639). Groups whose
    members are all invisible to the caller are dropped entirely.
    """
    from treeloom.application.job_groups import summarize_group
    if _state._job_group_store is None or _state._job_store is None:
        return []
    groups = await _state._job_group_store.list_all()  # newest-first
    all_jobs = await _state._job_store.list_all()
    allowed = await authz._visible_job_filter(request)
    by_group: dict[str, list] = {}
    for j in all_jobs:
        if not j.group_id:
            continue
        if allowed is not None and not await allowed(j.source_id):
            continue
        by_group.setdefault(j.group_id, []).append(j)
    out = []
    for g in groups:
        members = by_group.get(g.id, [])
        if allowed is not None and not members:
            continue
        out.append(summarize_group(g, members))
    return out


@app.get("/job-groups/{group_id}")
async def get_job_group(group_id: str, request: Request):
    """A single Job: its rolled-up summary plus its Tasks (public job shape).

    This took no `request` at all, so it could neither authorize nor redact —
    and it returned `t.to_dict()`, the RAW job rows rather than the redacted
    view every other job endpoint uses. Since the middleware exempts the
    /job-groups prefix, that handed any caller the absolute paths and error
    text of every task in any group they could name (CWE-306 / CWE-639).
    Scoped and redacted like /jobs now.
    """
    from treeloom.application.job_groups import summarize_group
    if _state._job_group_store is None or _state._job_store is None:
        raise HTTPException(404, f"Job group not found: {group_id}")
    group = await _state._job_group_store.get(group_id)
    if group is None:
        raise HTTPException(404, f"Job group not found: {group_id}")

    tasks = await _state._job_store.list_by_group(group_id)
    allowed = await authz._visible_job_filter(request)
    if allowed is not None:
        visible = [t for t in tasks if await allowed(t.source_id)]
        if not visible:
            # Every task is someone else's — do not confirm the group exists.
            raise HTTPException(404, f"Job group not found: {group_id}")
        tasks = visible

    redact = await authz._job_view_is_redacted(request)
    summary = summarize_group(group, tasks)
    summary["tasks"] = [authz._public_job(t.to_dict(), redact=redact) for t in tasks]
    return summary


@app.get("/jobs/dead-letter")
async def list_dead_letter_jobs(request: Request):
    """List all jobs in DEAD_LETTER status (admin-only when AUTH_ENABLED).

    Dead-lettered jobs are those that exhausted MAX_JOB_ATTEMPTS retries.
    Use POST /jobs/{id}/retry to re-queue them manually.

    MUST be registered before /jobs/{job_id} so FastAPI doesn't swallow
    "dead-letter" as a job_id path parameter.
    """
    if os.environ.get("AUTH_ENABLED", "").lower() == "true":
        user = await authz._require_authenticated_user(request)
        if user is None or getattr(user, "role", None) != "admin":
            raise HTTPException(403, "Admin role required")
    if _state._job_store is None:
        return {"jobs": [], "total": 0}
    jobs = await _state._job_store.list_dead_letter()
    result = []
    for j in jobs:
        result.append({
            "job_id": j.id,
            "source": j.source,
            "source_id": j.source_id,
            "kind": j.kind,
            "attempts": j.attempts,
            "error": j.error,
            "finished_at": j.finished_at,
        })
    return {"jobs": result, "total": len(result)}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    job = _state._jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    allowed = await authz._visible_job_filter(request)
    if allowed is not None and not await allowed(job.get("source_id")):
        # 404, not 403: a job id the caller may not see should not be
        # confirmed to exist.
        raise HTTPException(404, f"Job not found: {job_id}")
    return authz._public_job(job, redact=await authz._job_view_is_redacted(request))


@app.get("/jobs/{job_id}/errors")
async def get_job_errors(
    job_id: str, request: Request, limit: int = 1000, offset: int = 0
):
    """List per-file errors for a job. Returns each errored file with
    error_kind (timeout/exception), error_message, elapsed_s, and
    occurred_at — populated from the `job_file_errors` table.

    Combined with the job's `errors` counter, lets operators investigate
    which specific files failed without grepping process logs.

    This is the most disclosive endpoint under the `/jobs` prefix, which the
    middleware exempts from auth wholesale (the exemption is a PREFIX, so it
    covers this path too). Every row pairs an absolute file path with raw
    exception text. When AUTH_ENABLED=true an unauthenticated caller now gets
    the shape and the timings but neither the paths nor the messages — enough
    to see that files are failing and how badly, not enough to map a private
    tree or read a stack trace.
    """
    # Redaction alone was not enough here. It hides paths from an
    # UNAUTHENTICATED caller, but this endpoint was never scoped to the
    # caller's grants, so any authenticated non-admin could read any job's
    # per-file error log by id — the same CWE-639 that list_jobs and get_job
    # were fixed for, left open on the one endpoint under /jobs that is
    # purely composed of file paths and exception text.
    allowed = await authz._visible_job_filter(request)
    if allowed is not None and not await allowed(await authz._source_id_for_job(job_id)):
        # 404 rather than 403, matching get_job: a job id the caller may not
        # see should not be confirmed to exist.
        raise HTTPException(404, f"Job not found: {job_id}")

    rows = await _state._job_file_error_store.list_for_job(
        job_id, limit=max(1, min(int(limit), 5000)), offset=max(0, int(offset)),
    )
    errors = [r.to_dict() for r in rows]
    redact = await authz._job_view_is_redacted(request)
    if redact:
        for e in errors:
            for field in ("file_path", "error_message"):
                if field in e:
                    e[field] = None
    return {
        "job_id": job_id,
        "count": len(rows),
        "errors": errors,
        **({"redacted": True} if redact else {}),
    }


@app.post("/jobs/{job_id}/retry")
async def retry_dead_letter_job(job_id: str, request: Request):
    """Re-queue a DEAD_LETTER job for another attempt (admin-only when AUTH_ENABLED).

    Resets the status to 'queued' (attempts counter preserved) and re-enqueues.
    Returns 404 if the job does not exist, 409 if the job is not in a terminal
    or dead-letter state (cannot retry a running job).
    """
    if os.environ.get("AUTH_ENABLED", "").lower() == "true":
        user = await authz._require_authenticated_user(request)
        if user is None or getattr(user, "role", None) != "admin":
            raise HTTPException(403, "Admin role required")
    if _state._job_store is None:
        raise HTTPException(503, "Job store not available")
    job = await _state._job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    terminal_statuses = {"dead_letter", "failed", "done"}
    if job.status.value not in terminal_statuses:
        raise HTTPException(
            409,
            f"Job {job_id} is in status {job.status.value!r} — "
            "can only retry terminal or dead-letter jobs",
        )
    jd = job.to_dict()
    jd["status"] = "queued"
    jd["finished_at"] = None
    jd["error"] = ""
    await runners._persist_job(jd)
    await _state._job_queue.enqueue(job_id)
    return {"job_id": job_id, "status": "queued", "attempts": job.attempts}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job = _state._jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return {"status": "deleted", "job_id": job_id}


def _caller_id(request: Request) -> str | None:
    """Return the authenticated caller's id, or None when auth is disabled."""
    user = getattr(request.state, "user", None)
    return user.id if user is not None else None


@app.post("/index-file", response_model=JobAck, status_code=202)
async def handle_index_file(req: IndexFileRequest, request: Request):
    file_path = os.path.expanduser(req.file_path)
    authz._require_path_in_scope(file_path)
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    if not detect_language(str(path)):
        raise HTTPException(415, f"Unsupported language: {path.suffix}")

    source_id = make_source_id(path=str(path))

    active = await _routes_webhook._find_active_job_for_source(source_id)
    if active:
        logger.info(
            "File %s already has %s job %s — refusing duplicate submission",
            file_path, active["status"], active["job_id"],
        )
        return JobAck(job_id=active["job_id"], status=active["status"], source_id=source_id)

    if not req.force:
        existing = _find_done_job_for_source(source_id)
        if existing:
            logger.info(
                "File %s already indexed (job %s) — skipping; pass force=true to reindex",
                file_path, existing["job_id"],
            )
            return JobAck(job_id=existing["job_id"], status="done", source_id=source_id)

    job = runners._new_job("file", source_id, file_path)
    job["source_path"] = str(path)
    job["created_by"] = _caller_id(request)
    await _attach_group(job, job["created_by"])
    await runners._persist_job(job)
    await _state._job_queue.enqueue(job["job_id"])
    return JobAck(job_id=job["job_id"], status="queued", source_id=source_id)


@app.post("/index-directory", response_model=JobAck, status_code=202)
async def handle_index_directory(req: IndexDirectoryRequest, request: Request):
    directory = os.path.expanduser(req.directory)
    authz._require_path_in_scope(directory)
    if not os.path.isdir(directory):
        raise HTTPException(404, f"Directory not found: {directory}")

    source_id = make_source_id(path=directory)

    active = await _routes_webhook._find_active_job_for_source(source_id)
    if active:
        logger.info(
            "Directory %s already has %s job %s — refusing duplicate submission",
            directory, active["status"], active["job_id"],
        )
        return JobAck(job_id=active["job_id"], status=active["status"], source_id=source_id)

    if not req.force:
        existing = _find_done_job_for_source(source_id)
        if existing:
            logger.info(
                "Source %s already indexed (job %s) — skipping; pass force=true to reindex",
                directory, existing["job_id"],
            )
            return JobAck(job_id=existing["job_id"], status="done", source_id=source_id)

    job = runners._new_job("directory", source_id, directory)
    job["source_path"] = directory
    job["created_by"] = _caller_id(request)
    await _attach_group(job, job["created_by"])
    if FEATURE_SKIP_PATTERNS and req.skip_patterns:
        job["skip_patterns"] = list(req.skip_patterns)

    pf_warnings: list[dict] | None = None
    pf_recommendations: dict | None = None
    if req.auto_preflight:
        try:
            pf = await _run_preflight_for_path(directory)
            if pf["verdict"] in ("yellow", "red"):
                pf_warnings = pf["warnings"]
                pf_recommendations = pf["recommendations"]
        except Exception:
            logger.exception("auto-preflight failed; proceeding without warnings")

    await runners._persist_job(job)
    await _state._job_queue.enqueue(job["job_id"])
    return JobAck(
        job_id=job["job_id"], status="queued", source_id=source_id,
        warnings=pf_warnings, recommendations=pf_recommendations,
    )


@app.post("/index-repo", response_model=JobAck, status_code=202)
async def handle_index_repo(req: IndexRepoRequest, request: Request):
    if not req.path and not req.url:
        raise HTTPException(400, "Provide a `path` or `url`")
    if req.url and not runners._is_safe_git_url(req.url):
        raise HTTPException(400, f"Unsafe git URL: {req.url!r}")
    if req.branch and req.branch.startswith("-"):
        raise HTTPException(400, f"Unsafe git branch: {req.branch!r}")

    if req.path:
        authz._require_path_in_scope(os.path.expanduser(req.path))
    label = req.url or os.path.expanduser(req.path or "")
    source_id = make_source_id(path=req.path, url=req.url, branch=req.branch)

    active = await _routes_webhook._find_active_job_for_source(source_id)
    if active:
        logger.info(
            "Repo %s already has %s job %s — refusing duplicate submission",
            label, active["status"], active["job_id"],
        )
        return JobAck(job_id=active["job_id"], status=active["status"], source_id=source_id)

    if not req.force:
        existing = _find_done_job_for_source(source_id)
        if existing:
            logger.info(
                "Repo %s already indexed (job %s) — skipping; pass force=true to reindex",
                label, existing["job_id"],
            )
            return JobAck(job_id=existing["job_id"], status="done", source_id=source_id)

    job = runners._new_job("repo", source_id, label)
    job["source_url"] = req.url or ""
    job["source_path"] = label if not req.url else ""
    job["source_branch"] = req.branch or ""
    job["created_by"] = _caller_id(request)
    await _attach_group(job, job["created_by"], req.group_id)
    if FEATURE_SKIP_PATTERNS and req.skip_patterns:
        job["skip_patterns"] = list(req.skip_patterns)

    # Auto-preflight (best-effort; non-blocking). Only runs for local
    # paths — for remote URLs we'd need a shallow clone which is expensive
    # to do twice. Skip patterns are surfaced regardless of feature flag.
    pf_warnings: list[dict] | None = None
    pf_recommendations: dict | None = None
    if req.auto_preflight and req.path:
        try:
            pf = await _run_preflight_for_path(os.path.expanduser(req.path))
            if pf["verdict"] in ("yellow", "red"):
                pf_warnings = pf["warnings"]
                pf_recommendations = pf["recommendations"]
        except Exception:
            logger.exception("auto-preflight failed; proceeding without warnings")

    await runners._persist_job(job)
    await _state._job_queue.enqueue(job["job_id"])
    return JobAck(
        job_id=job["job_id"], status="queued", source_id=source_id,
        warnings=pf_warnings, recommendations=pf_recommendations,
    )


# ── User management endpoints (US3) ────────────────────────────────

from treeloom.domain.authorization import Role, scopes_for_role


# ── Local auth: session endpoints ────────────────────────────


# ── Password management ───────────────────────────────────────


# ── Role management ──────────────────────────────────────────


# ── PAT endpoints ────────────────────────────────────────────


# ── API key endpoints ────────────────────────────────────────


# ── authorization: groups & per-source grants (admin only) ─────────


async def _require_admin_authenticated(request: Request):
    """Admin gate for endpoints the middleware's /sources GET skip-route exempts.

    The skip-route means request.state.user is never set for GET /sources/*, so
    `authz._require_admin` would 401 even valid admins. Re-authenticate from the
    header (like get_source_staleness) and require ADMIN. Returns None when AUTH
    is globally disabled (open mode), matching the rest of the service.
    """
    user = await authz._require_authenticated_user(request)
    if user is None:
        return None  # AUTH disabled — open mode
    if user.role != Role.ADMIN:
        raise HTTPException(403, "Admin access required")
    return user


@app.get("/audit/search")
async def get_search_audit(
    request: Request,
    user_id: str | None = None,
    source_id: str | None = None,
    limit: int = 100,
):
    """Who-searched-what trail (admin only): recent authorization decisions,
    optionally filtered by user or source."""
    authz._require_admin(request)
    return await _state._search_audit.recent(
        user_id=user_id, source_id=source_id, limit=limit
    )


@app.get("/sources/{source_id}/grants")
async def list_source_grants(source_id: str, request: Request):
    """List the access grants on a source (admin only)."""
    await _require_admin_authenticated(request)
    return await _state._grant_store.list_for_source(source_id)


@app.post("/sources/{source_id}/grants", status_code=201)
async def add_source_grant(
    source_id: str, req: GrantRequest, request: Request
):
    """Grant (or deny) a principal access to a source (admin only).

    effect='deny' carves an exclusion out of an otherwise all_access principal.
    """
    authz._require_admin(request)
    ok = await _state._grant_store.grant(
        req.principal_type, req.principal_id, source_id, req.effect
    )
    if not ok:
        raise HTTPException(503, "Database unavailable")
    return {
        "source_id": source_id,
        "principal_type": req.principal_type,
        "principal_id": req.principal_id,
        "effect": req.effect,
    }


@app.delete(
    "/sources/{source_id}/grants/{principal_type}/{principal_id}",
    status_code=204,
)
async def revoke_source_grant(
    source_id: str, principal_type: str, principal_id: str, request: Request
):
    """Revoke a principal's grant on a source (admin only)."""
    authz._require_admin(request)
    if not await _state._grant_store.revoke(principal_type, principal_id, source_id):
        raise HTTPException(404, "Grant not found")
    return None


# ── source registry ──────────────────────────────────────────────


@app.get("/sources")
async def get_sources(request: Request):
    """Return persisted sources, filtered to the caller's own records.

    When AUTH_ENABLED, the caller must be authenticated; admins see all
    records, non-admins see only those they created (plus legacy records
    with no `created_by`, since they predate the authorization model and
    can't be attributed). Falls back to the Neo4j graph store when the
    PostgreSQL registry is empty or unavailable — those records have no
    `created_by` and are treated as legacy.
    """
    user = await authz._require_authenticated_user(request)

    def _visible_to_user(created_by) -> bool:
        if user is None:
            return True  # AUTH_ENABLED off — preserve dev behavior
        if user.role == Role.ADMIN:
            return True
        if not created_by:
            return True  # Legacy / pre-auth record
        return created_by == user.id

    try:
        records = await _state._source_repo.list_all()
        if records:
            return [
                {
                    "id": r.id,
                    "path": r.path,
                    "url": r.url,
                    "branch": r.branch,
                    "indexed_at": int(r.indexed_at.timestamp()),
                    "file_count": r.file_count,
                    "chunk_count": r.chunk_count,
                    "commit_sha": r.commit_sha,
                    "graph_indexed": r.graph_indexed,
                }
                for r in records
                if _visible_to_user(r.created_by)
            ]
    except Exception:
        pass  # Degraded mode — fall through to graph store

    fallback = await graph_store.list_sources()
    return [s for s in fallback if _visible_to_user(s.get("created_by"))]


@app.get("/sources/{source_id}/staleness")
async def get_source_staleness(source_id: str, request: Request):
    """Compare the stored HEAD SHA against the current HEAD of the source.

    Resolves current HEAD locally (`git rev-parse`) for path-backed sources
    and via `git ls-remote` for URL-backed sources. `is_stale` is null when
    either SHA cannot be determined (e.g., non-git input, network failure).
    """
    user = await authz._require_authenticated_user(request)
    src = await authz._load_source(source_id)
    if src is None:
        raise HTTPException(404, f"Source not found: {source_id}")

    # IDOR guard: when auth is on, only the source creator (or admin) may
    # query a source. Records with no `created_by` (legacy / pre-auth) stay
    # accessible. Return 404 on mismatch to avoid leaking existence.
    if user is not None:
        created_by = src.get("created_by")
        if created_by and created_by != user.id and user.role != Role.ADMIN:
            raise HTTPException(404, f"Source not found: {source_id}")

    staleness = await runners.compute_source_staleness(src)
    branch = src.get("branch") or ""
    return {
        "source_id": source_id,
        "branch": branch,
        "indexed_sha": staleness["indexed_sha"],
        "current_sha": staleness["current_sha"],
        "resolved_via": staleness["resolved_via"],
        "is_stale": staleness["is_stale"],
    }


# ── Fleet operations ────────────────────────────────────────────────

# Above this many sources, live staleness resolution (git ls-remote / rev-parse
# per source) on a single /fleet?staleness=true call would be a self-inflicted
# DoS; we skip it and flag the rollup truncated. The freshness sampler's
# Prometheus gauges cover staleness at scale instead.
_FLEET_STALENESS_MAX_SOURCES = int(os.environ.get("FLEET_STALENESS_MAX_SOURCES", "200"))


def _source_visible_to_user(user, created_by) -> bool:
    """Visibility predicate shared with GET /sources.

    AUTH off (user None) → everything; admins → everything; otherwise the
    caller's own records plus legacy/pre-auth records (no created_by).
    """
    if user is None:
        return True
    if user.role == Role.ADMIN:
        return True
    if not created_by:
        return True
    return created_by == user.id


async def _all_source_dicts() -> list[dict]:
    """Every source as a dict (SourceRecord registry first, graph-store fallback).

    Includes ``created_by`` for downstream visibility filtering; callers that
    return these to clients should not echo it. Shape matches what
    ``runners.compute_source_staleness`` and ``build_fleet_health`` expect.
    """
    try:
        records = await _state._source_repo.list_all()
        if records:
            return [
                {
                    "id": r.id,
                    "path": r.path,
                    "url": r.url,
                    "branch": r.branch,
                    "indexed_at": int(r.indexed_at.timestamp()),
                    "file_count": r.file_count,
                    "chunk_count": r.chunk_count,
                    "commit_sha": r.commit_sha,
                    "graph_indexed": r.graph_indexed,
                    "created_by": r.created_by,
                    "kind": r.kind,
                }
                for r in records
            ]
    except Exception:
        pass  # Degraded mode — fall through to graph store
    return await graph_store.list_sources()


@app.get("/fleet")
async def get_fleet(
    request: Request,
    staleness: bool = False,
    entities: bool = False,
    summary_only: bool = False,
):
    """Fleet health rollup across every visible source in one response.

    Answers "is the fleet healthy?" without per-source API calls. Cheap by
    default (registry + latest-job join). Two opt-in enrichments guard the
    expensive paths:

    - ``staleness=true``: live git HEAD comparison per source, capped at
      FLEET_STALENESS_MAX_SOURCES (sets ``summary.staleness_truncated`` above).
    - ``entities=true``: one aggregate graph-store query for per-source entity
      counts; absent (not fatal) if the graph store is unavailable.

    ``summary_only=true`` returns just the aggregate block.
    """
    user = await authz._require_authenticated_user(request)
    sources = [
        s for s in await _all_source_dicts()
        if _source_visible_to_user(user, s.get("created_by"))
    ]

    jobs: list[dict] = []
    if _state._job_store is not None:
        try:
            jobs = [j.to_dict() for j in await _state._job_store.list_all()]
        except Exception:
            logger.warning("fleet rollup: failed to list jobs; last-error data omitted")

    staleness_by_id: dict | None = None
    truncated = False
    if staleness:
        if len(sources) > _FLEET_STALENESS_MAX_SOURCES:
            truncated = True
        else:
            staleness_by_id = {}
            for s in sources:
                sid = s.get("id") or ""
                if not sid:
                    continue
                try:
                    staleness_by_id[sid] = (await runners.compute_source_staleness(s))["is_stale"]
                except Exception:
                    staleness_by_id[sid] = None

    entity_counts: dict | None = None
    if entities:
        try:
            entity_counts = await graph_store.entity_counts_by_source()
        except Exception:
            logger.warning("fleet rollup: entity counts unavailable (graph store down?)")

    health = build_fleet_health(
        sources,
        jobs,
        staleness_by_id=staleness_by_id,
        entity_counts=entity_counts,
        staleness_truncated=truncated,
    )
    if summary_only:
        return {"summary": health["summary"]}
    return health


@app.delete("/sources/{source_id}")
async def remove_source(source_id: str, request: Request):
    """Permanently destroy one source's chunks, graph, communities and record.

    Authorized through the same per-source choke point as the read endpoints.
    This handler previously took no `request` at all, so it could not check
    anything: any caller who could reach the indexer could enumerate source_ids
    from GET /sources and irrecoverably wipe every one of them (CWE-862 /
    CWE-639). Contrast add_source_grant() and delete_user(), which do gate.

    NOTE on the permission model: the ACL has a single allow/deny axis, so a
    principal granted access to a source may now delete it. The `index` scope
    gate narrows that to credentials meant for writing, but it is still not a
    read/write split *within* the ACL — if deletion should be owner-or-admin
    only, that is a policy decision to make deliberately rather than a bug to
    patch here.

    Enforcement is off when AUTH_ENABLED is false, exactly as it is for the
    read endpoints; what keeps this off the network by default is the loopback
    INDEXER_HOST bind. It is NOT off under TREELOOM_SEARCH_OPEN — that flag
    opens search only, hence `authz._write_auth_enforced()` here.
    """
    # Two independent gates. The scope gate asks what this *credential* is
    # permitted to do; the grant gate asks whether its owner is authorized for
    # this *source*. A search-only PAT belonging to someone with full access to
    # a source would pass the second on its own, which is the whole reason
    # scopes exist — so it has to clear both.
    authz._require_scope(request, "index")
    await authz._authorize_scope(
        request, source_id, action="remove_source",
        enforced=authz._write_auth_enforced(),
    )
    try:
        delete_chunks_by_source(source_id)
    except Exception as e:
        print(f"warning: delete_chunks_by_source failed: {e}")
    await graph_store.delete_source(source_id)
    # Community nodes are namespaced "<source_id>#<n>", not CONTAINED by the
    # Source node, so delete_source leaves them behind — clear them here or
    # every deletion permanently grows the :Community label.
    await graph_store.delete_source_communities(source_id)
    # Remove from PostgreSQL source registry too (graceful degradation)
    try:
        await _state._source_repo.delete(source_id)
    except Exception:
        pass
    return {"status": "deleted", "source_id": source_id}


# ── graph indexing / community ───────────────────────────────────


@app.post("/index-graph", response_model=JobAck, status_code=202)
async def handle_index_graph(req: IndexGraphRequest, request: Request):
    """Pass 2 of two-pass indexing: graph-extract a chunks-only source.

    Requires the source to already exist in `source_records` (i.e., it
    was previously indexed at least once — typically with
    skip_graph=true). Drops the source's existing graph entities and
    rebuilds; safe to re-submit after extractor changes.

    Authorization: this used to check only that the caller was authenticated,
    so any principal could name any source_id and have its graph dropped and
    rebuilt (CWE-862). `graph_store.delete_source()` runs first, and between
    that and the rebuild completing, the target's `search_code_enhanced` and
    traversal results are degraded — a cheap way to damage another tenant's
    index. It is now gated exactly like deletion: the `index` scope for the
    credential, the per-source ACL for its owner.
    """
    authz._require_scope(request, "index")
    if not req.source_id:
        raise HTTPException(400, "source_id is required")
    await authz._authorize_scope(
        request, req.source_id, action="index_graph",
        enforced=authz._write_auth_enforced(),
    )

    record = await _state._source_repo.get_by_id(req.source_id)
    if record is None:
        raise HTTPException(
            404,
            f"source_id={req.source_id!r} not found — chunks-index it first "
            f"(POST /index-repo) before running /index-graph.",
        )

    active = await _routes_webhook._find_active_job_for_source(req.source_id)
    if active:
        return JobAck(
            job_id=active["job_id"], status=active["status"],
            source_id=req.source_id,
        )

    label = record.url or record.path
    job = runners._new_job("graph", req.source_id, label)
    job["source_path"] = record.path or ""
    job["source_url"] = record.url or ""
    job["source_branch"] = record.branch or ""
    job["created_by"] = _caller_id(request)

    await runners._persist_job(job)
    await _state._job_queue.enqueue(job["job_id"])
    return JobAck(job_id=job["job_id"], status="queued", source_id=req.source_id)


# Backfill runs across every source (~20s each), so it must not block the
# request. We track it as a single in-process background task with a small
# status record polled via GET /build-community. Only one runs at a time.


# ── Freshness sampler ───────────────────────────────────────────────
# Background task that periodically samples per-source staleness and updates
# Prometheus gauges. Gated by FRESHNESS_SAMPLER_ENABLED (default on).


# ── Login-attempt purge ──────────────────────────────────────────────────────
# Rows fall out of the sliding window but never out of the table: clear() only
# fires on a *successful* login, which is the one thing a brute-force run never
# produces. Without this loop the table and its index grow for the lifetime of
# the deployment.


# ── Fleet auto-refresh scheduler ────────────────────────────────────
# In-process loop that re-indexes sources whose git HEAD has moved, on a
# cadence, so a fleet stays fresh without manual `force`. Default OFF — the
# operator opts in. Mirrors the freshness sampler's task lifecycle.


async def _enqueue_source_reindex(src: dict) -> str:
    """Build + enqueue a re-index job for an existing source, of the SAME kind
    it was originally indexed as.

    Re-enqueueing every source as a `repo` job is wrong: a file-indexed
    source fails the repo runner's `isdir` check, and a directory-indexed
    source gets repo-walk semantics. We honor the persisted `kind` (falling
    back to filesystem shape for legacy rows without it). `runners.dispatch_job` then
    routes the job to the matching runner via its `kind` + `source_path`.

    Used by the fleet auto-refresh loop; the caller owns the
    one-active-job-per-source guard. Returns the new job_id.
    """
    url = src.get("url") or ""
    path = src.get("path") or ""
    expanded = os.path.expanduser(path) if path else ""
    kind = resolve_reindex_kind(
        stored_kind=src.get("kind"),
        has_url=bool(url),
        path_is_file=bool(expanded) and os.path.isfile(expanded),
    )
    label = url or path or src.get("id") or ""
    job = runners._new_job(kind, src["id"], label)
    job["source_url"] = url
    job["source_path"] = "" if url else expanded
    job["source_branch"] = src.get("branch") or ""
    job["created_by"] = src.get("created_by")
    await runners._persist_job(job)
    await _state._job_queue.enqueue(job["job_id"])
    return job["job_id"]


async def _run_community_backfill():

    started = time.time()
    _state._community_build_state = {
        "status": "running",
        "started_at": started,
        "sources_total": 0,
        "sources_done": 0,
        "communities": 0,
    }
    try:
        sources = await graph_store.list_sources()
        _state._community_build_state["sources_total"] = len(sources)
        communities = 0
        done = 0
        for s in sources:
            sid = s.get("id")
            if not sid:
                continue
            try:
                communities += await community.run_post_index_signals(sid)
            except Exception:
                logger.exception("community backfill failed for source %s", sid)
            done += 1
            _state._community_build_state["sources_done"] = done
            _state._community_build_state["communities"] = communities
        _state._community_build_state.update(
            {"status": "done", "finished_at": time.time()}
        )
        logger.info(
            "community backfill complete: %d/%d sources, %d communities",
            done, len(sources), communities,
        )
    except Exception as exc:
        logger.exception("community backfill crashed")
        _state._community_build_state = {
            "status": "failed",
            "error": str(exc),
            "finished_at": time.time(),
        }


@app.post("/build-community", status_code=202)
async def handle_build_community(request: Request):
    """Kick off a background backfill of graph signals for every source.

    Per-source community detection + centrality (~20s/source), so this
    returns immediately and the work runs in the background. Safe to
    re-run; poll GET /build-community for progress. Only one build runs at
    a time — a second POST while one is in flight returns its live status.

    ADMIN ONLY. This handler previously took no `request` at all, so it could
    not check anything: any authenticated principal could start a fleet-wide
    rebuild costing roughly 20 seconds per source, across every source in the
    install regardless of their grants. At a few hundred sources that is
    hours of work from one unprivileged POST, and the single-flight guard
    makes it worse rather than better — it means one caller can keep the
    build slot occupied and deny it to the operator.

    It is global and unscoped by nature, so there is no per-source ACL to
    apply; admin is the right gate.
    """
    authz._require_admin(request)

    if _state._community_build_task is not None and not _state._community_build_task.done():
        return {**_state._community_build_state, "status": "already_running"}
    _state._community_build_task = asyncio.create_task(_run_community_backfill())
    return {**_state._community_build_state, "status": "started"}


@app.get("/build-community")
async def handle_build_community_status():
    """Progress of the most recent /build-community backfill."""
    return _state._community_build_state


# ── webhook endpoint ────────────────────────────────────────────────


# ── incremental index job ──────────────────────────────────────────────


# ── search & graph endpoints (US2) ────────────────────────────────


_RESPONSE_MODES = ("full", "facet", "summary_tail")


# ── Request bounds for the read endpoints ────────────────────────────────────
# /search and /graph-explore are in the auth middleware's PUBLIC_PATHS, so
# these fields are UNAUTHENTICATED input. They were unbounded: a multi-megabyte
# `query` is embedded (a TEI round-trip sized by the caller) and then reranked,
# and `top_k`/`rerank_pool` size the vector search and the cross-encoder batch
# directly. One request could occupy the embedding backend and the reranker GPU
# for as long as the caller cared to make it (CWE-400).
#
# The caps are set where more stops being useful rather than where it starts to
# hurt, so no real query meets them:
#   query      — the embedding model (jina-v2-base-code) takes 8192 TOKENS, and
#                chars >= tokens, so anything past this is truncated by the
#                model anyway.
#   top_k      — the rerank-pool sweep found 50 == 150 == 300 on every
#                retrieval metric (benchmarks/results/embedding-ab/
#                rerank_pool_sweep.json); 200 is already far past the knee.
#   rerank_pool— same sweep; 150 costs 2.8x the search latency for nothing.
#   hydrate ids— one Milvus query per id.
MAX_QUERY_CHARS = int(os.environ.get("TREELOOM_MAX_QUERY_CHARS", "8192"))
MAX_TOP_K = int(os.environ.get("TREELOOM_MAX_TOP_K", "200"))
MAX_RERANK_POOL = int(os.environ.get("TREELOOM_MAX_RERANK_POOL", "500"))
MAX_HYDRATE_IDS = int(os.environ.get("TREELOOM_MAX_HYDRATE_IDS", "200"))


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    top_k: int = Field(10, ge=1, le=MAX_TOP_K)
    language: str | None = None
    path_prefix: str | None = Field(None, max_length=4096)
    source_id: str | None = Field(None, max_length=256)
    cross_repo: bool = False
    use_hybrid: bool | None = None
    use_hyde: bool | None = None
    use_summary_vector: bool | None = None
    use_graph_scoring: bool | None = None
    rerank_pool: int | None = Field(None, ge=1, le=MAX_RERANK_POOL)
    return_pool: bool = False
    check_staleness: bool = False
    # ── EXPERIMENTAL: benchmarked & REJECTED, retained for reproducibility ──
    # The fields below are payload-trimming experiments from the token-optimization
    # milestone. All were benchmarked and rejected (compensatory fetch raises
    # mean turns and erases the token saving — see docs/token-optimization-
    # rejected.md). They ship opt-in / default-off ONLY as reproducible evidence
    # and for the benchmark arms; do NOT enable them as a "savings" feature. The
    # rejection holds across 5 agent models and *strengthens* for frontier ones
    #. The token wins that DID ship are serialization-only, not
    # these content trims.
    #
    # Opt-in server-side body-shaping. NEVER default; "full" is
    # byte-identical to today. "facet" = metadata + chunk header, no code body
    # (rejected: turns +18%). "summary_tail" = top-2 full snippets, tail replaced
    # by the indexed LLM summary (rejected: loses at every verbosity tier).
    # Validated to a 400 at the endpoint (not a pydantic 422) per the contract.
    response_mode: str = "full"
    # (rejected): drop the community-summary payload. Default True
    # preserves current behavior.
    include_community_summaries: bool = True
    # (rejected): per-request override of the env ADAPTIVE_TOPK
    # flag/gap (None = use env default). Default behavior is unchanged.
    adaptive_topk: bool | None = None
    adaptive_topk_gap: float | None = None
    strip_imports: bool | None = None        #, rejected
    query_class_payload: bool | None = None   #, rejected
    # read an alternate indexed summary tier without a restart (powered the
    # verbosity sweep). Only consumed by summary_tail, itself rejected.
    summary_prompt_version: int | None = None


def _require_scope_xor(path_prefix: str | None, source_id: str | None,
                       cross_repo: bool) -> None:
    """Demand exactly one of: a repo scope (path_prefix or source_id) XOR an
    explicit cross_repo whole-index search. Reject neither (a silent
    whole-index search lets cross-repo symbol collisions corrupt
    name/definition signals) and reject both (ambiguous intent)."""
    scoped = bool(path_prefix) or bool(source_id)
    if scoped and cross_repo:
        raise HTTPException(
            status_code=400,
            detail="Specify a repo scope (path_prefix or source_id) OR "
                   "cross_repo=true, not both.",
        )
    if not scoped and not cross_repo:
        raise HTTPException(
            status_code=400,
            detail="Search must be repo-scoped: set path_prefix or source_id, "
                   "or pass cross_repo=true to deliberately search the whole "
                   "multi-repo index.",
        )


class GraphExploreRequest(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    # depth drives graph traversal, which fans out multiplicatively per hop.
    depth: int = Field(1, ge=1, le=5)
    language: str | None = None
    path_prefix: str | None = Field(None, max_length=4096)
    source_id: str | None = Field(None, max_length=256)
    use_hybrid: bool | None = None
    use_hyde: bool | None = None
    use_summary_vector: bool | None = None
    use_graph_scoring: bool | None = None
    rerank_pool: int | None = Field(None, ge=1, le=MAX_RERANK_POOL)
    check_staleness: bool = False
    response_mode: str = "full"
    include_community_summaries: bool = True
    adaptive_topk: bool | None = None
    adaptive_topk_gap: float | None = None
    strip_imports: bool | None = None
    query_class_payload: bool | None = None
    summary_prompt_version: int | None = None


def _require_valid_response_mode(response_mode: str) -> None:
    if response_mode not in _RESPONSE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"response_mode must be one of {_RESPONSE_MODES}, "
                   f"got {response_mode!r}.",
        )


async def _resolve_to_entities(
    name_or_id: str,
    kind: str | None = None,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    if "://" in name_or_id and "#" in name_or_id:
        entity = await graph_store.get_entity_by_id(name_or_id)
        return [entity] if entity else []
    return await graph_store.find_entities_by_name(
        name_or_id, entity_type=kind, source_id=source_id,
        exclude_source_ids=exclude_source_ids,
    )


async def _attach_provenance(result: dict, *, check_staleness: bool) -> None:
    """Join indexed commit_sha (+ opt-in live staleness) onto a search result.

    Best-effort: a source-store hiccup must never fail an otherwise-good search.
    """
    try:
        sids = {result.get("source_id")} | {
            c.get("source_id") for c in result.get("chunks", [])
        }
        sids = {s for s in sids if s}
        if not sids:
            return
        records_by_id = {}
        for sid in sids:
            rec = await _state._source_repo.get_by_id(sid)
            if rec:
                records_by_id[sid] = rec
        staleness_by_id = None
        if check_staleness and records_by_id:
            staleness_by_id = {}
            for sid, rec in records_by_id.items():
                st = await runners.compute_source_staleness({
                    "commit_sha": rec.commit_sha, "branch": rec.branch,
                    "path": rec.path, "url": rec.url, "indexed_at": rec.indexed_at,
                })
                staleness_by_id[sid] = {
                    "is_stale": st.get("is_stale"),
                    "current_sha": st.get("current_sha", ""),
                }
        from treeloom.domain.provenance import build_provenance
        build_provenance(result, records_by_id, staleness_by_id)
    except Exception:  # best-effort enrichment, never break search
        logger.warning("provenance enrichment failed", exc_info=True)


@app.post("/search")
async def handle_search(req: SearchRequest, request: Request):
    _require_scope_xor(req.path_prefix, req.source_id, req.cross_repo)
    _require_valid_response_mode(req.response_mode)
    if req.return_pool:
        # return_pool is the rerank-mining switch: it returns the
        # whole pre-rerank pool of raw chunk bodies and skips rerank, graph
        # and top_k entirely. The ACL prefilter still applies, so it is not a
        # confidentiality bypass — but it is a bulk-extraction accelerator,
        # handing back up to rerank_pool full bodies per request where an
        # ordinary search returns top_k. It is an operator tool; gate it like
        # one rather than leaving it on for every caller.
        authz._require_scope(request, "admin")
    exclude = await authz._authorize_scope(request, req.source_id)
    query_embedding = await embed_query([req.query])
    result = await graph_search(
        query_embedding[0],
        req.query,
        top_k=req.top_k,
        language=req.language,
        path_prefix=req.path_prefix,
        source_id=req.source_id,
        exclude_source_ids=exclude,
        use_hybrid=req.use_hybrid,
        use_hyde=req.use_hyde,
        use_summary_vector=req.use_summary_vector,
        use_graph_scoring=req.use_graph_scoring,
        rerank_pool=req.rerank_pool,
        return_pool=req.return_pool,
        response_mode=req.response_mode,
        include_community_summaries=req.include_community_summaries,
        adaptive_topk=req.adaptive_topk,
        adaptive_topk_gap=req.adaptive_topk_gap,
        strip_imports=req.strip_imports,
        query_class_payload=req.query_class_payload,
        summary_prompt_version=req.summary_prompt_version,
    )
    await _attach_provenance(result, check_staleness=req.check_staleness)
    return result


class HydrateRequest(BaseModel):
    # facet hit_id handles ("file_path:start-end") from a response_mode=facet
    # search; hydrate_chunks reconstructs the dropped code bodies in one batch.
    # Each id costs its own Milvus query, so the list is bounded.
    ids: list[str] = Field(..., max_length=MAX_HYDRATE_IDS)
    source_id: str | None = Field(None, max_length=256)


@app.post("/hydrate-chunks")
async def handle_hydrate_chunks(req: HydrateRequest, request: Request):
    """Companion to response_mode=facet: fetch the exact code
    bodies for selected facet hits in a single batched call."""
    # authz._authorize_scope returns the caller's denied sources for a shared-index
    # query. Discarding it, as this did, left the exclusion unapplied: a
    # source_id-pinned request was 403'd correctly, but an unpinned one
    # hydrated any hit_id the caller could name — and a hit_id is just a file
    # path and line range, guessable for a repo whose layout they know.
    exclude = await authz._authorize_scope(request, req.source_id, action="hydrate_chunks")
    chunks = await hydrate_chunks(
        req.ids, source_id=req.source_id, exclude_source_ids=exclude
    )
    return {"chunks": chunks}


@app.get("/find-definition")
async def handle_find_definition(
    request: Request,
    name: str,
    kind: str | None = None,
    source_id: str | None = None,
):
    exclude = await authz._authorize_scope(request, source_id, action="find_definition")
    entities = await graph_store.find_entities_by_name(
        name, entity_type=kind, source_id=source_id,
        exclude_source_ids=exclude,
    )
    # Entity dicts returned by find_entities_by_name are plain Neo4j/SQLite
    # node properties (dict(r["e"])).  Source membership is modelled as a
    # (:Source)-[:CONTAINS]->(e:Entity) graph relationship, not a property
    # on the Entity node itself, so there is no source_id field to resolve
    # provenance from.  Per-entity citation/commit_sha enrichment is therefore
    # skipped here; callers that need provenance should use /search instead.
    return entities


@app.get("/find-callers")
async def handle_find_callers(
    request: Request,
    name_or_id: str,
    source_id: str | None = None,
):
    exclude = await authz._authorize_scope(request, source_id, action="find_callers")
    targets = await _resolve_to_entities(
        name_or_id, kind="Function", source_id=source_id,
        exclude_source_ids=exclude,
    )
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for t in targets:
        callers = await graph_store.find_callers(
            t["id"], source_id=source_id, exclude_source_ids=exclude,
        )
        for c in callers:
            key = (c.get("id", ""), c.get("_target_id", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out


@app.get("/find-references")
async def handle_find_references(
    request: Request,
    name_or_id: str,
    source_id: str | None = None,
):
    exclude = await authz._authorize_scope(request, source_id, action="find_references")
    targets = await _resolve_to_entities(
        name_or_id, source_id=source_id, exclude_source_ids=exclude,
    )
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for t in targets:
        refs = await graph_store.find_references(
            t["id"], source_id=source_id, exclude_source_ids=exclude,
        )
        for r in refs:
            key = (
                r.get("id", ""),
                r.get("_rel_type", ""),
                r.get("_target_id", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


@app.post("/graph-explore")
async def handle_graph_explore(req: GraphExploreRequest, request: Request):
    _require_valid_response_mode(req.response_mode)
    exclude = await authz._authorize_scope(request, req.source_id, action="graph_explore")
    query_embedding = await embed_query([req.query])
    result = await graph_search(
        query_embedding[0],
        req.query,
        top_k=3,
        traverse_depth=req.depth,
        language=req.language,
        path_prefix=req.path_prefix,
        source_id=req.source_id,
        exclude_source_ids=exclude,
        use_hybrid=req.use_hybrid,
        use_hyde=req.use_hyde,
        use_summary_vector=req.use_summary_vector,
        use_graph_scoring=req.use_graph_scoring,
        rerank_pool=req.rerank_pool,
        response_mode=req.response_mode,
        include_community_summaries=req.include_community_summaries,
        adaptive_topk=req.adaptive_topk,
        adaptive_topk_gap=req.adaptive_topk_gap,
        strip_imports=req.strip_imports,
        query_class_payload=req.query_class_payload,
        summary_prompt_version=req.summary_prompt_version,
    )
    await _attach_provenance(result, check_staleness=req.check_staleness)
    return result


if __name__ == "__main__":
    uvicorn.run("treeloom.indexer_service:app", host="0.0.0.0", port=8001)
