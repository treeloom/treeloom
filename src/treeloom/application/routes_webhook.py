"""The incremental-indexing webhook endpoint.

Split out of `indexer_service.py` (stage 5/6). POST /webhook plus the
backpressure gate and job-group attachment it needs — the provider adapters
themselves live in `adapters/webhook/`.

On an APIRouter that `indexer_service` includes; path, method and handler
name are unchanged, and `tests/unit/test_route_table.py` pins the binding.

Authorization comes from `indexer_authz`, shared state from `indexer_state`,
job execution from `indexer_runners`. Nothing here imports `indexer_service`.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from treeloom.infrastructure import metrics
from fastapi import HTTPException, Request
from pydantic import BaseModel
from treeloom import graph_store
from treeloom.adapters.webhook.gitea import GiteaAdapter
from treeloom.adapters.webhook.github import GitHubAdapter
from treeloom.application import indexer_state as _state
from treeloom.application import indexer_runners as runners
from treeloom.domain.webhook import WebhookPayload, Provider, detect_provider
from treeloom.sources import make_source_id
from typing import Any
import inspect
import json
import os
import time
import uuid
from treeloom.application import indexer_authz as authz
from treeloom.application import indexer_state as _state

logger = logging.getLogger(__name__)

router = APIRouter()


async def _find_active_job_for_source(source_id: str) -> dict | None:
    """Return any queued/running job for this source, from the cross-worker
    JobStore. Unlike `_find_done_job_for_source` (which reads in-memory
    `_state._jobs`), this hits Postgres so it sees jobs submitted to peer workers."""
    if not source_id or _state._job_store is None:
        return None
    job = await _state._job_store.find_active_for_source(source_id)
    return job.to_dict() if job else None


async def _shed_if_saturated(repo_url: str):
    """Backpressure gate. Returns a 429 ``JSONResponse`` when the job
    queue is at/over ``_state.MAX_QUEUE_DEPTH``, else ``None``. Updates the
    ``treeloom_queue_saturation`` gauge as a side effect.

    Applied to *every* webhook-triggered enqueue path — incremental, the
    oversized full-reindex fallback, AND a first-time full index for a new
    source — so a flood on any path can't grow the backlog unbounded.
    """
    from fastapi.responses import JSONResponse

    if _state._job_queue is None:
        return None
    try:
        depth = await _state._job_queue.depth()
    except Exception:
        return None
    metrics.queue_saturation.set(min(depth / max(_state.MAX_QUEUE_DEPTH, 1), 1.0))
    if depth >= _state.MAX_QUEUE_DEPTH:
        metrics.webhook_shed_total.inc()
        logger.warning(
            "Webhook for %s shed: queue depth %d ≥ _state.MAX_QUEUE_DEPTH %d",
            repo_url, depth, _state.MAX_QUEUE_DEPTH,
        )
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={
                "detail": (
                    f"Queue saturated ({depth}/{_state.MAX_QUEUE_DEPTH} jobs). "
                    "Retry after the queue drains."
                ),
                "queue_depth": depth,
                "max_queue_depth": _state.MAX_QUEUE_DEPTH,
            },
        )
    return None


class WebhookRequest(BaseModel):
    job_id: str
    status: str
    source: str = ""
    files_to_process: int | str = 0
    files_removed: int = 0
    message: str = ""


async def _attach_webhook_group(job: dict, label: str) -> None:
    """Create a group-of-1 of kind 'webhook' for a webhook-triggered Task and
    stamp ``job['group_id']``. No-op if the group store isn't initialized.

    Webhook jobs are created outside the /index-* endpoints, so they need their
    own attach call to keep the "every job has a group_id" invariant and to show
    up as Jobs in the operator UI.
    """

    if _state._job_group_store is None:
        return
    job["group_id"] = await _state._job_group_store.create(
        label=label, kind="webhook", created_by=None, task_count=1,
    )


# ── Queue backpressure config ─────────────────────────────────


MAX_CHANGED_FILES = int(os.environ.get("MAX_CHANGED_FILES", "5000"))


@router.post("/webhook", status_code=202)
async def handle_webhook(request: Request):
    """Accept incoming webhook from GitHub or Gitea.

    Validates the provider, verifies HMAC / tokens, looks up the
    source, and dispatches either a full re-index or an incremental
    index of the changed files.
    """
    # 1. Get raw body and headers
    raw_body = await request.body()
    try:
        body_json = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    headers = dict(request.headers)

    # 2. Detect provider
    try:
        provider = detect_provider(headers, body_json)
    except ValueError:
        raise HTTPException(400, "Unknown webhook provider")

    # 3. Instantiate the correct adapter
    if provider == Provider.GITHUB:
        secret = os.environ.get("WEBHOOK_GITHUB_SECRET")
        if not secret:
            raise HTTPException(503, "WEBHOOK_GITHUB_SECRET not configured")
        token = os.environ.get("WEBHOOK_GITHUB_TOKEN")
        # Gated test/load mode: read changed files from the payload instead of
        # the rate-limited GitHub API. Default off; never enable in production.
        trust_inline = os.environ.get("WEBHOOK_TRUST_INLINE_FILES", "") == "1"
        adapter = GitHubAdapter(
            secret=secret, token=token, trust_inline_files=trust_inline
        )

    elif provider == Provider.GITEA:
        secret = os.environ.get("WEBHOOK_GITEA_SECRET")
        base_url = os.environ.get("WEBHOOK_GITEA_BASE_URL")
        if not secret or not base_url:
            raise HTTPException(
                503, "WEBHOOK_GITEA_SECRET or WEBHOOK_GITEA_BASE_URL not configured"
            )
        token = os.environ.get("WEBHOOK_GITEA_TOKEN")
        adapter = GiteaAdapter(secret=secret, base_url=base_url, token=token)

    else:  # pragma: no cover — every Provider member is handled above
        # Without this, an unhandled provider leaves `adapter` as None and the
        # AttributeError below is swallowed into a misleading 401.
        raise HTTPException(400, f"Unsupported webhook provider: {provider.value}")

    # 4. Call adapter.handle_webhook (may be sync or async)
    try:
        handle_fn = adapter.handle_webhook
        # Pass the RAW request bytes so HMAC providers (GitHub/Gitea) validate
        # the signature against exactly what the provider signed — not a
        # re-serialized ``json.dumps(parsed)`` (which differs from the wire
        # bytes and rejects real deliveries).
        if inspect.iscoroutinefunction(handle_fn):
            payload: WebhookPayload | None = await handle_fn(
                headers, body_json, raw_body=raw_body
            )
        else:
            payload = handle_fn(headers, body_json, raw_body=raw_body)
    except PermissionError as e:
        raise HTTPException(401, str(e))
    except Exception:
        raise HTTPException(401, "Webhook validation failed")

    if payload is None:
        # Return 401 when an HMAC signature was provided but validation
        # failed (GitHub/Gitea).  Non-merge events have no signature header.
        lower_hdrs = {k.lower(): v for k, v in headers.items()}
        has_sig = bool(
            lower_hdrs.get("x-hub-signature-256", "")
            or lower_hdrs.get("x-gitea-signature", "")
        )
        if has_sig:
            raise HTTPException(401, "Invalid webhook signature")

        # Non-merge event — acknowledge without action
        job_id = uuid.uuid4().hex
        ack: dict[str, Any] = {
            "job_id": job_id,
            "status": "done",
            "source": "",
            "files_to_process": 0,
            "files_removed": 0,
            "message": (
                "Webhook received but no indexing action required "
                "(non-merge event or invalid signature)"
            ),
        }
        _state._jobs[job_id] = ack
        return ack

    repo_url = payload.repo_url
    branch = payload.branch
    changed_files = payload.changed_files

    # 5. Look up source_id for repo_url in existing sources
    sources = await graph_store.list_sources()
    existing_source = None
    for s in sources:
        if s.get("url") == repo_url:
            existing_source = s
            break

    source_id = make_source_id(url=repo_url, branch=branch)

    if existing_source is None:
        # 6. Not indexed yet — create full index job. Guard against a
        # repeat webhook firing while the first full-index is in flight.
        active = await _find_active_job_for_source(source_id)
        if active:
            logger.info(
                "Webhook for %s arrived while %s job %s is in flight — skipping",
                repo_url, active["status"], active["job_id"],
            )
            return {
                "job_id": active["job_id"],
                "status": active["status"],
                "source": repo_url,
                "files_to_process": len(changed_files),
                "files_removed": 0,
                "message": "in-flight job already covers this source",
            }
        shed = await _shed_if_saturated(repo_url)
        if shed is not None:
            return shed
        job = runners._new_job("repo", source_id, repo_url)
        job["source_url"] = repo_url
        job["source_branch"] = branch
        job["files_to_process"] = len(changed_files)
        job["message"] = (
            f"Full index triggered for {repo_url} (branch: {branch})"
        )
        await _attach_webhook_group(job, f"webhook: {repo_url}")
        await runners._persist_job(job)
        await _state._job_queue.enqueue(job["job_id"])
        return {
            "job_id": job["job_id"],
            "status": "queued",
            "source": repo_url,
            "files_to_process": len(changed_files),
            "files_removed": 0,
            "message": job["message"],
        }

    # 7. Already indexed.
    #
    # Dedup (#2): if a job for this source is already queued/running — including
    # an incremental job that re-queued itself after a retry — don't create a
    # colliding job. The jobs_active_source_uniq partial index would otherwise
    # reject the second insert and 500 the webhook. Mirrors the new-source guard.
    active = await _find_active_job_for_source(source_id)
    if active:
        return {
            "job_id": active["job_id"],
            "status": active["status"],
            "source": repo_url,
            "files_to_process": len(changed_files),
            "files_removed": 0,
            "message": "in-flight job already covers this source",
        }

    # Backpressure: shed with 429 when the queue is saturated.
    # Applied to both the incremental job and the oversized full-reindex
    # fallback below — neither path may grow the backlog unbounded.
    shed = await _shed_if_saturated(repo_url)
    if shed is not None:
        return shed

    # Oversized payload check: if changed_files exceeds
    # MAX_CHANGED_FILES, fall back to a full re-index job instead of a giant
    # incremental payload. One bounded job replaces thousands of changed_files.
    if len(changed_files) > MAX_CHANGED_FILES:
        logger.warning(
            "Webhook for %s: %d changed files exceeds MAX_CHANGED_FILES=%d — "
            "falling back to full re-index",
            repo_url, len(changed_files), MAX_CHANGED_FILES,
        )
        full_job = runners._new_job("repo", source_id, repo_url)
        full_job["source_url"] = repo_url
        full_job["source_branch"] = branch
        full_job["files_to_process"] = len(changed_files)
        full_job["message"] = (
            f"Full re-index triggered: {len(changed_files)} changed files exceeds "
            f"MAX_CHANGED_FILES={MAX_CHANGED_FILES}"
        )
        await _attach_webhook_group(full_job, f"webhook: {repo_url}")
        await runners._persist_job(full_job)
        await _state._job_queue.enqueue(full_job["job_id"])
        return {
            "job_id": full_job["job_id"],
            "status": "queued",
            "source": repo_url,
            "files_to_process": len(changed_files),
            "files_removed": 0,
            "message": full_job["message"],
        }

    accept_time = time.time()
    job = runners._new_job("incremental", source_id, repo_url)
    job["source_url"] = repo_url
    job["source_branch"] = branch
    job["files_to_process"] = len(changed_files)
    job["files_removed"] = sum(
        1 for f in changed_files if f["action"] == "removed"
    )
    # Persist changed_files in payload so any worker can reconstruct and resume
    # this job after a crash — no longer reliant on the git provider re-firing.
    job["payload"] = {"changed_files": changed_files, "branch": branch, "accept_time": accept_time}
    await _attach_webhook_group(job, f"webhook: {repo_url} (incremental)")
    await runners._persist_job(job)
    await _state._job_queue.enqueue(job["job_id"])
    return {
        "job_id": job["job_id"],
        "status": "queued",
        "source": repo_url,
        "files_to_process": len(changed_files),
        "files_removed": sum(1 for f in changed_files if f["action"] == "removed"),
        "message": f"Incremental index for {len(changed_files)} changed files",
    }


