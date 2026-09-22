"""Process lifecycle: startup, shutdown, and the background loops.

Split out of `indexer_service.py` (stage 6/6). Pool creation and migrations,
admin seeding, job recovery, and the three periodic loops
(freshness sampler, login-attempt purge, fleet auto-refresh) with their
config, plus the cancellation that unwinds them.

`startup` and `shutdown` are plain coroutines taking the FastAPI app as an
argument — they read and write `app.state`, and importing the app from
`indexer_service` would be the cycle this split exists to avoid.
indexer_service binds them in its `_lifespan` context manager (passed as
`FastAPI(lifespan=...)`), which awaits `startup(app)` on enter and
`shutdown(app)` on exit — the ASGI lifespan protocol, since Starlette 1.6
removed the older `add_event_handler`/`on_event` API.

Nothing here imports `indexer_service`; state comes from `indexer_state`.
"""

from __future__ import annotations

import logging
from treeloom import graph_store
from treeloom.application import indexer_state as _state
from treeloom.application import indexer_runners as runners
from treeloom.application import routes_auth as rauth
from treeloom.domain.authorization import Role
from treeloom.application.fleet import select_sources_to_refresh
from treeloom.application import routes_webhook as _webhook
from treeloom.infrastructure import metrics
import asyncio
import os
import time

logger = logging.getLogger(__name__)




async def _fleet_refresh_once() -> int:
    """One tick: enqueue re-index jobs for stale sources. Returns count enqueued."""
    if _state._job_queue is None or _state._job_store is None:
        return 0
    # Lazy + module-qualified: indexer_service imports this module for the
    # startup hook, so a top-level import is a cycle; binding the NAMES would
    # also defeat patching (the rule SPLIT enforces).
    from treeloom.application import indexer_service as _svc

    try:
        sources = [s for s in await _svc._all_source_dicts() if s.get("id")]
    except Exception:
        logger.warning("Fleet auto-refresh: failed to list sources")
        return 0

    # Cap staleness checks per tick (each URL source = a git ls-remote). Above
    # the cap, check only a rotating window so the whole fleet is covered over
    # several ticks instead of firing hundreds of serial git calls each time.
    global _fleet_refresh_cursor
    n = len(sources)
    if n > _FLEET_REFRESH_MAX_SOURCES:
        start = _fleet_refresh_cursor % n
        window = (sources + sources)[start:start + _FLEET_REFRESH_MAX_SOURCES]
        _fleet_refresh_cursor = (start + _FLEET_REFRESH_MAX_SOURCES) % n
        logger.info(
            "Fleet auto-refresh: %d sources > cap %d — checking a rotating "
            "window of %d this tick (offset %d)",
            n, _FLEET_REFRESH_MAX_SOURCES, len(window), start,
        )
    else:
        window = sources

    annotated: list[dict] = []
    for s in window:
        try:
            st = await runners.compute_source_staleness(s)
        except Exception:
            continue
        annotated.append({**s, "is_stale": st["is_stale"]})

    by_id = {s["id"]: s for s in window}
    targets = select_sources_to_refresh(
        annotated, max_per_tick=_FLEET_REFRESH_MAX_PER_TICK
    )

    enqueued = 0
    for sid in targets:
        src = by_id.get(sid)
        if src is None:
            continue
        # The single-active-job-per-source invariant already prevents pile-ups;
        # skipping here avoids needless job churn and log noise.
        if await _webhook._find_active_job_for_source(sid):
            continue
        try:
            job_id = await _svc._enqueue_source_reindex(src)
            metrics.fleet_refresh_enqueued_total.inc()
            enqueued += 1
            logger.info(
                "Fleet auto-refresh: re-indexing stale source %s (job %s)", sid, job_id
            )
        except Exception:
            logger.exception("Fleet auto-refresh: failed to enqueue source %s", sid)
    return enqueued


def _warn_if_auth_disabled() -> None:
    """Log a loud warning when AUTH_ENABLED is not true.

    The indexer binds 0.0.0.0 (required so the Docker MCP container can reach
    it via host.docker.internal → host-gateway, which cannot reach loopback).
    Without auth, every endpoint — including /index-repo which reads arbitrary
    local paths and clones git URLs — is open to any process that can reach
    that interface.
    """
    if os.environ.get("AUTH_ENABLED", "").lower() != "true":
        logger.warning(
            "AUTH_ENABLED is false — every indexer endpoint (including "
            "/index-repo, which reads arbitrary local paths and clones git "
            "URLs) is unauthenticated. This process binds %s. Loopback is the "
            "default, which keeps those endpoints off the network; if you have "
            "set INDEXER_HOST=0.0.0.0 (needed only for the opt-in http-mcp "
            "container to reach it via host.docker.internal), do NOT run this "
            "on a machine with a public IP without a firewall, or set "
            "AUTH_ENABLED=true.",
            os.environ.get("INDEXER_HOST", "127.0.0.1"),
        )


_FRESHNESS_SAMPLER_ENABLED = os.environ.get("FRESHNESS_SAMPLER_ENABLED", "true").lower() not in ("0", "false", "no")


_FRESHNESS_SAMPLE_INTERVAL = float(os.environ.get("FRESHNESS_SAMPLE_INTERVAL_SECONDS", "60"))


_FRESHNESS_MAX_SOURCES = int(os.environ.get("FRESHNESS_MAX_SOURCES", "200"))


# Source-ids that carried per-source freshness labels on the previous tick.
# Each tick we remove the Prometheus series for ids that vanished, so stale
# per-source gauges don't linger forever.
_freshness_labeled_sources: set[str] = set()


_LOGIN_PURGE_ENABLED = os.environ.get(
    "TREELOOM_LOGIN_PURGE_ENABLED", "true"
).lower() not in ("0", "false", "no")


_LOGIN_PURGE_INTERVAL = float(
    os.environ.get("TREELOOM_LOGIN_PURGE_INTERVAL_SECONDS", "3600")
)


def _login_purge_retention() -> int:
    """Retention for login_attempts rows, in seconds.

    Kept well past the window so a purge can never delete a row the throttle
    is still counting, whatever the clock skew between workers. Unset, blank
    (``.env.example`` ships ``TREELOOM_LOGIN_PURGE_RETENTION_SECONDS=``) and
    ``0`` all mean "derive it" — a blank used to reach ``int("")`` and abort
    the indexer at import.
    """
    raw = os.environ.get("TREELOOM_LOGIN_PURGE_RETENTION_SECONDS", "").strip()
    return (int(raw) if raw else 0) or max(rauth.LOGIN_WINDOW_SECONDS * 4, 3600)


_LOGIN_PURGE_RETENTION = _login_purge_retention()


async def _run_login_attempt_purge() -> None:
    """Periodically drop login_attempts rows older than the retention window.

    Every worker runs this; the DELETE is idempotent and bounded by the same
    index the throttle query uses, so overlapping ticks cost a little
    duplicated work and nothing else.
    """
    while True:
        try:
            await asyncio.sleep(_LOGIN_PURGE_INTERVAL)
            deleted = await _state._login_attempt_store.purge_expired(_LOGIN_PURGE_RETENTION)
            if deleted:
                logger.info("Purged %d expired login_attempts rows", deleted)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Login-attempt purge tick failed; will retry next interval")


async def _run_freshness_sampler() -> None:
    """Periodically update treeloom_source_index_stale / _age_seconds gauges.

    Cardinality guard: if source count exceeds FRESHNESS_MAX_SOURCES, the
    per-source label is dropped and only the aggregate
    treeloom_sources_stale_total gauge is updated. This prevents label
    explosion on large installations.
    """
    while True:
        try:
            await asyncio.sleep(_FRESHNESS_SAMPLE_INTERVAL)
            await _sample_freshness_once()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Freshness sampler tick failed; will retry next interval")


def _prune_vanished_freshness_labels(current: set[str]) -> None:
    """Remove per-source freshness series for ids no longer present.

    Prometheus ``Gauge`` retains a child series for every label set it has
    ever seen until ``.remove()`` is called. Without this, a deleted source —
    or the switch into aggregate mode (where ``current`` is empty) — would
    leave its ``treeloom_source_index_stale``/``_age_seconds`` series exported
    forever at their last value.
    """
    global _freshness_labeled_sources
    for sid in _freshness_labeled_sources - current:
        try:
            metrics.source_index_stale.remove(sid)
        except KeyError:
            pass
        try:
            metrics.source_index_age_seconds.remove(sid)
        except KeyError:
            pass
    _freshness_labeled_sources = current


async def _sample_freshness_once() -> None:
    """One tick of the freshness sampler — updates Prometheus gauges."""
    # Keep the queue-saturation gauge fresh between webhooks: otherwise
    # it only moves on the webhook path (_shed_if_saturated) and goes stale
    # while the queue drains.
    if _state._job_queue is not None:
        try:
            depth = await _state._job_queue.depth()
            metrics.queue_saturation.set(min(depth / max(_state.MAX_QUEUE_DEPTH, 1), 1.0))
        except Exception:
            logger.debug("Freshness sampler: failed to sample queue depth")

    try:
        sources = await graph_store.list_sources()
    except Exception:
        logger.warning("Freshness sampler: failed to list sources")
        return

    now = time.time()
    if len(sources) > _FRESHNESS_MAX_SOURCES:
        # Cardinality guard: aggregate mode. Drop every per-source series
        # (current set is empty) so we don't double-count.
        _prune_vanished_freshness_labels(set())
        stale_count = 0
        for src in sources:
            try:
                s = await runners.compute_source_staleness(src)
                if s["is_stale"]:
                    stale_count += 1
            except Exception:
                pass
        metrics.sources_stale_total.set(stale_count)
        return

    # Per-source mode.
    current: set[str] = set()
    for src in sources:
        sid = src.get("id") or src.get("source_id") or ""
        if not sid:
            continue
        try:
            s = await runners.compute_source_staleness(src)
            metrics.source_index_stale.labels(source_id=sid).set(
                1 if s["is_stale"] else 0
            )
            current.add(sid)
            indexed_at = s.get("indexed_at")
            if indexed_at is not None:
                try:
                    age = now - float(indexed_at)
                    metrics.source_index_age_seconds.labels(source_id=sid).set(age)
                except (TypeError, ValueError):
                    pass
        except Exception:
            logger.debug("Freshness sampler: error sampling source %s", sid)

    _prune_vanished_freshness_labels(current)


_FLEET_AUTO_REFRESH_ENABLED = os.environ.get(
    "FLEET_AUTO_REFRESH_ENABLED", "false"
).lower() in ("1", "true", "yes")


_FLEET_REFRESH_INTERVAL = float(os.environ.get("FLEET_REFRESH_INTERVAL_SECONDS", "900"))


# Bound the per-tick re-index burst so a fleet-wide HEAD move (e.g. a mass
# rebase) doesn't flood the queue in one tick.
_FLEET_REFRESH_MAX_PER_TICK = int(os.environ.get("FLEET_REFRESH_MAX_PER_TICK", "25"))


# Bound the per-tick STALENESS checks (each URL source is a `git ls-remote`
# round-trip). Above this many sources we check only a rotating window per
# tick so the whole fleet is covered over several ticks without firing
# hundreds of serial git calls every interval — mirrors the /fleet endpoint's
# FLEET_STALENESS_MAX_SOURCES guard.
_FLEET_REFRESH_MAX_SOURCES = int(os.environ.get("FLEET_REFRESH_MAX_SOURCES", "200"))


_fleet_refresh_cursor = 0  # rotating offset into the source list across ticks


async def _run_fleet_refresh() -> None:
    """Periodically re-index sources whose HEAD moved."""
    while True:
        try:
            await asyncio.sleep(_FLEET_REFRESH_INTERVAL)
            await _fleet_refresh_once()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Fleet auto-refresh tick failed; will retry next interval")


async def _seed_admin_if_first_start() -> None:
    """Seed the admin user on first startup if no users exist.

    Checks both legacy TREELOOM_ADMIN_KEY (API-key-based) and the new
    TREELOOM_BOOTSTRAP_ADMIN_USER + TREELOOM_BOOTSTRAP_ADMIN_PASSWORD
    (local-login based). Either path creates an admin; both can coexist.
    """
    try:
        count = await _state._user_store.count_users()
        if count != 0:
            return

        admin_key = os.environ.get("TREELOOM_ADMIN_KEY", "")
        bootstrap_user = os.environ.get("TREELOOM_BOOTSTRAP_ADMIN_USER", "")
        bootstrap_pass = os.environ.get("TREELOOM_BOOTSTRAP_ADMIN_PASSWORD", "")
        has_password = bool(bootstrap_user and bootstrap_pass)

        if not admin_key and not has_password:
            logger.warning(
                "No TREELOOM_ADMIN_KEY or TREELOOM_BOOTSTRAP_ADMIN_USER/"
                "TREELOOM_BOOTSTRAP_ADMIN_PASSWORD set — cannot seed admin user"
            )
            return

        from treeloom.adapters.authorization.user_store import (
            BCRYPT_MAX_PASSWORD_BYTES,
            PasswordTooLongError,
            hash_api_key,
            hash_password,
        )

        # A single seeded admin can carry BOTH auth methods: an API key (from
        # TREELOOM_ADMIN_KEY) and local username/password (from the bootstrap
        # vars). They coexist on one account rather than racing on an `elif` or
        # colliding on the UNIQUE username. Username defaults to the bootstrap
        # name, else "admin".
        username = bootstrap_user or "admin"
        api_key_hash = hash_api_key(admin_key) if admin_key else ""
        user = await _state._user_store.create_user(
            username=username,
            email="",
            role=Role.ADMIN,
            api_key_hash=api_key_hash,
        )
        if user is not None and has_password:
            try:
                await _state._user_store.update_password(
                    user.id, hash_password(bootstrap_pass)
                )
            except PasswordTooLongError:
                logger.error(
                    "TREELOOM_BOOTSTRAP_ADMIN_PASSWORD exceeds bcrypt's "
                    "%d-byte limit; the bootstrap admin password was NOT set",
                    BCRYPT_MAX_PASSWORD_BYTES,
                )

        methods = []
        if api_key_hash:
            methods.append("TREELOOM_ADMIN_KEY")
        if has_password:
            methods.append("password")
        logger.info("Seeded admin user '%s' (%s)", username, ", ".join(methods))
    except Exception:
        logger.exception("Failed to seed admin user on startup")


async def startup(app):
    """Initialize PostgreSQL pool, run migrations, and seed the admin user."""
    from treeloom.infrastructure.config import validate_config
    from treeloom.infrastructure.logging import configure_logging
    from treeloom.infrastructure.tracing import init_tracer

    # Initialize tracing before anything else so subsequent startup work
    # (config validation, pool init, migrations) lands in
    # the trace tree under a single startup span.
    init_tracer("treeloom-indexer", app=app)
    validate_config()
    log_file = os.environ.get("STRUCTURED_LOG_FILE", "")
    configure_logging(json_file=log_file or None)
    _warn_if_auth_disabled()
    if _state.DATABASE_URL:
        from treeloom.adapters.postgresql.connection import init_pool
        from treeloom.adapters.postgresql import run_migrations

        pool = await init_pool(_state.DATABASE_URL)
        if pool is not None:
            await run_migrations()

    # Ensure graph-store constraints + indexes exist. Idempotent and online,
    # so safe on every startup; without the Entity(file_path)/Entity(name)
    # indexes, graph lookups full-scan the whole graph and search hangs.
    try:
        from treeloom import graph_store

        await graph_store.ensure_schema()
    except Exception:
        logger.exception("Failed to ensure graph-store schema on startup")

    # Build the embedding proxy from the Postgres-backed registry; seed
    # the registry from .env on the first startup. The proxy LISTENs for
    # mutations from other workers and reloads in place. TEI-only: cloud
    # embedding providers (EMBEDDING_PROVIDER=openai) bypass the registry.
    if _state.DATABASE_URL and os.environ.get("EMBEDDING_PROVIDER", "tei") == "tei":
        from treeloom.adapters.postgresql.embedding_backend_store import EmbeddingBackendStore
        from treeloom.adapters.tei.embedding_proxy import EmbeddingProxy, set_proxy

        backend_store = EmbeddingBackendStore()
        try:
            await backend_store.seed_from_env_if_empty()
            proxy = await EmbeddingProxy.from_registry(backend_store)
            await proxy.start_listener(backend_store, _state.DATABASE_URL)
            set_proxy(proxy)
            app.state.embedding_proxy = proxy
            app.state.embedding_backend_store = backend_store
            logger.info("EmbeddingProxy initialized from registry with %d backend(s)", len(proxy._backends))
        except Exception:
            logger.exception("Failed to initialize EmbeddingProxy from registry; falling back to env-based init")

    await _seed_admin_if_first_start()

    # ── JobStore: resume incomplete jobs from previous run ──────────

    from treeloom.adapters.postgresql.job_store import JobStore
    from treeloom.adapters.postgresql.job_group_store import JobGroupStore
    from treeloom.adapters.queue.postgres_queue import PostgresJobQueue
    from treeloom.domain.jobs import JobStatus

    _state._job_store = JobStore()
    await _state._job_store.init()

    _state._job_group_store = JobGroupStore()
    await _state._job_group_store.init()

    _state._job_queue = PostgresJobQueue()
    await _state._job_queue.start()

    # Load completed jobs first so dedup / supersede checks below can see them
    done_jobs = await _state._job_store.list_by_status(JobStatus.DONE)
    done_source_ids: set[str] = set()
    for job in done_jobs:
        jd = job.to_dict()
        jd["status"] = "done"
        _state._jobs[job.id] = jd
        if job.source_id:
            done_source_ids.add(job.source_id)
    if done_jobs:
        logger.info("Restored %d completed jobs from previous run", len(done_jobs))

    # DEAD_LETTER jobs are terminal — never auto-re-enqueued on restart.
    # Only `POST /jobs/{id}/retry` can move them back to queued.
    incomplete = []
    for st in (JobStatus.RUNNING, JobStatus.QUEUED):
        incomplete.extend(await _state._job_store.list_by_status(st))

    if incomplete:
        resumed = superseded = abandoned = 0
        for job in incomplete:
            jd = job.to_dict()

            # Source is already DONE elsewhere — mark this stale job as superseded.
            if job.source_id and job.source_id in done_source_ids:
                jd["status"] = "done"
                jd["finished_at"] = time.time()
                jd["message"] = "Superseded by prior completed index for this source"
                _state._jobs[job.id] = jd
                await runners._persist_job(jd)
                superseded += 1
                continue

            # Probe whether this job can be dispatched. `runners.dispatch_job` sets
            # `_state._jobs[job.id]` as a side effect and returns the runner coro,
            # which we close here — the queue worker will rebuild it when
            # it actually pops the job_id.
            probe = runners.dispatch_job(job)
            if probe is None:
                jd["status"] = "failed"
                jd["error"] = f"Cannot resume {job.kind!r} job after restart"
                jd["finished_at"] = time.time()
                await runners._persist_job(jd)
                abandoned += 1
                continue
            probe.close()
            _state._jobs[job.id]["_resume"] = True
            await _state._job_queue.enqueue(job.id)
            resumed += 1

        logger.info(
            "Job recovery: %d resumed, %d superseded (already done), %d abandoned",
            resumed, superseded, abandoned,
        )

    # ── Login-attempt purge ──────────────────────────────────────────────────

    if _LOGIN_PURGE_ENABLED:
        _state._login_purge_task = asyncio.create_task(_run_login_attempt_purge())
        logger.info(
            "Login-attempt purge started (interval=%.0fs, retention=%ds)",
            _LOGIN_PURGE_INTERVAL, _LOGIN_PURGE_RETENTION,
        )
    else:
        logger.info("Login-attempt purge disabled (TREELOOM_LOGIN_PURGE_ENABLED=false)")

    # ── Freshness sampler ────────────────────────────────

    if _FRESHNESS_SAMPLER_ENABLED:
        _state._freshness_sampler_task = asyncio.create_task(_run_freshness_sampler())
        logger.info(
            "Freshness sampler started (interval=%.0fs, max_sources=%d)",
            _FRESHNESS_SAMPLE_INTERVAL, _FRESHNESS_MAX_SOURCES,
        )
    else:
        logger.info("Freshness sampler disabled (FRESHNESS_SAMPLER_ENABLED=false)")

    # ── Fleet auto-refresh scheduler ─────────────────────

    if _FLEET_AUTO_REFRESH_ENABLED:
        _state._fleet_refresh_task = asyncio.create_task(_run_fleet_refresh())
        logger.info(
            "Fleet auto-refresh started (interval=%.0fs, max_per_tick=%d)",
            _FLEET_REFRESH_INTERVAL, _FLEET_REFRESH_MAX_PER_TICK,
        )
    else:
        logger.info("Fleet auto-refresh disabled (FLEET_AUTO_REFRESH_ENABLED=false)")


async def shutdown(app):
    # Cancel the freshness sampler task cleanly.
    if _state._freshness_sampler_task is not None and not _state._freshness_sampler_task.done():
        _state._freshness_sampler_task.cancel()
        try:
            await _state._freshness_sampler_task
        except asyncio.CancelledError:
            pass
    # Cancel the login-attempt purge task cleanly.
    if _state._login_purge_task is not None and not _state._login_purge_task.done():
        _state._login_purge_task.cancel()
        try:
            await _state._login_purge_task
        except asyncio.CancelledError:
            pass
    # Cancel the fleet auto-refresh task cleanly.
    if _state._fleet_refresh_task is not None and not _state._fleet_refresh_task.done():
        _state._fleet_refresh_task.cancel()
        try:
            await _state._fleet_refresh_task
        except asyncio.CancelledError:
            pass
    await graph_store.close()
    if _state._job_queue is not None:
        await _state._job_queue.stop()
    # Stop the embedding proxy's NOTIFY listener (releases its dedicated
    # connection). Must happen before close_pool() since this connection
    # is independent of the pool but still talks to the same Postgres.
    proxy = getattr(app.state, "embedding_proxy", None)
    if proxy is not None:
        try:
            await proxy.stop_listener()
        except Exception:
            logger.exception("Failed to stop EmbeddingProxy listener")
    # Close PostgreSQL pool if active
    try:
        from treeloom.adapters.postgresql.connection import close_pool
        await close_pool()
    except Exception:
        pass


