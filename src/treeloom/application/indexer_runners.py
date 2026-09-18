"""Index job execution — the runners and the helpers they need.

Split out of `indexer_service.py` (stage 2/6). That module was 5,400 lines,
large enough that the security scanner gives up slicing it and ships the whole
file into a prompt, and far past this repo's 800-line guidance.

Everything here is the *doing* of an index job rather than the HTTP surface of
it: the four `_run_index_*_job` runners plus the incremental webhook runner,
the walk/chunk/embed inner loop, job persistence and finalisation, and the git
URL/destination policy those clones depend on.

Direction of dependency is one-way: `indexer_service` imports this module, and
this module must NEVER import `indexer_service` back. Shared mutable state is
reached through `indexer_state` for the reasons set out there — reading a store
by name would snapshot the binding startup replaces.
"""

from __future__ import annotations

import logging
from opentelemetry import trace
from datetime import datetime, timezone
from git import Repo
from pathlib import Path
from treeloom import community
from treeloom import graph_store
from treeloom import llm
from treeloom.application import indexer_state as _state
from treeloom.domain.sources import SourceRecord
from treeloom.embedder import MAX_BATCH_SIZE, embed
from treeloom.graph_extractor import extract_graph
from treeloom.graph_store import delete_entities_by_file
from treeloom.indexer import CodeIndexer, detect_language, SUPPORTED_EXTENSIONS
from treeloom.infrastructure import metrics
from treeloom.infrastructure.config import (
    FEATURE_SKIP_PATTERNS,
    LOG_INTERVAL,
    OTEL_TRACE_PER_CHUNK,
    SKIP_FILENAMES,
    USE_SUMMARY_VECTOR,
)
from treeloom.infrastructure.fileio import read_source
from treeloom.retriever import delete_chunks_by_file
from treeloom.retriever import init_collection, insert_chunks, delete_chunks_by_source
from dataclasses import dataclass
from typing import Any, Coroutine
import asyncio
import os


# The chunker singleton. Lived in indexer_service, which this leaf may not
# import -- so every `indexer.parse_and_chunk` call raised. The service has no
# remaining use for it, so the leaf owns it outright. CodeIndexer builds its
# chunkers per call, so a single shared instance is safe here.
indexer = CodeIndexer()


@dataclass(frozen=True)
class _RepoJobSpec:
    """The fields a repo runner needs off an index-repo request.

    Deliberately NOT `indexer_service.IndexRepoRequest`: this module is a leaf
    and importing the service would be a cycle (enforced by
    tests/unit/test_indexer_state_wiring.py). A runner has no business
    depending on an HTTP request model either -- it needs four values.

    `_run_index_repo_job` accepts this or the pydantic request; both expose the
    same attributes. Rebuilding a job from the queue uses this one. Omitting it
    was the NameError that left every queued `repo` job stuck at `queued`.
    """

    path: str | None = None
    url: str | None = None
    branch: str | None = None
    force: bool = False
    skip_graph: bool = False

import shutil
import tempfile
import time
import uuid
from treeloom.application import indexer_state as _state

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _mark_span_error(span, exc: BaseException) -> None:
    """Record an exception on a span and flip its status to ERROR.

    Wrapped to keep call sites readable and so we degrade gracefully when
    the OTel SDK isn't installed (the no-op tracer has these methods).
    """
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        pass
    try:
        span.record_exception(exc)
    except Exception:
        pass


async def _persist_job(job_dict: dict) -> None:
    """Persist job state to SQLite, if store is initialized."""

    if _state._job_store is not None:
        from treeloom.domain.jobs import Job
        await _state._job_store.upsert(Job.from_dict(job_dict))


def _persist_job_sync(job_dict: dict) -> None:
    """Fire-and-forget persistence from sync context."""
    asyncio.ensure_future(_persist_job(job_dict))


def dispatch_job(job) -> "Coroutine | None":
    """Reconstruct a runner coroutine for a persisted Job row.

    Workers in the Postgres-backed queue use this to translate a popped
    job_id (after fetching the Job from JobStore) into the actual coroutine
    to await. Returns None if the job cannot be dispatched — the queue
    worker is expected to mark it failed and DELETE its queue row.

    Incremental jobs (webhook-triggered) persist their `changed_files` list
    in `job.payload` so they survive restarts and can be picked up by any
    worker via the queue. An incremental re-run is idempotent: each file is
    delete-then-insert, so re-running from the start after a crash is safe.
    Returns None only if the payload is missing or malformed.
    """
    jd = job.to_dict()
    kind = job.kind or "repo"

    if kind == "file":
        target = jd.get("source_path") or jd.get("source")
        if not target or not os.path.exists(target):
            return None
        _state._jobs[job.id] = jd
        return _run_index_file_job(jd, target)

    if kind == "directory":
        target = jd.get("source_path") or jd.get("source")
        if not target or not os.path.isdir(target):
            return None
        _state._jobs[job.id] = jd
        return _run_index_directory_job(jd, target, "**/*")

    if kind == "repo":
        rebuilt_req = _RepoJobSpec(
            path=jd.get("source_path") or None,
            url=jd.get("source_url") or None,
            branch=jd.get("source_branch") or None,
            skip_graph=bool(jd.get("skip_graph", False)),
        )
        if not rebuilt_req.path and not rebuilt_req.url:
            return None
        _state._jobs[job.id] = jd
        return _run_index_repo_job(jd, rebuilt_req)

    if kind == "graph":
        sid = jd.get("source_id")
        if not sid:
            return None
        _state._jobs[job.id] = jd
        return _run_index_graph_job(jd, sid)

    if kind == "incremental":
        payload = job.payload or {}
        changed_files = payload.get("changed_files")
        if not changed_files:
            return None
        repo_url = jd.get("source_url") or ""
        branch = payload.get("branch") or jd.get("source_branch") or ""
        source_id = jd.get("source_id") or ""
        if not repo_url or not source_id:
            return None
        _state._jobs[job.id] = jd
        return _run_incremental_index(source_id, repo_url, branch, changed_files, jd)

    return None


LOG_FILE_INTERVAL = int(os.environ.get("LOG_FILE_INTERVAL", "10"))


# Per-file ceiling — protects against a single file wedging the indexer.
# TEI / Milvus / Neo4j each have their own request timeouts, but a worst-case
# combination across stages can still exceed any single one. 10 minutes is
# generous for legitimately slow files; anything above this is a bug to fix.
INDEX_FILE_TIMEOUT = int(os.environ.get("INDEX_FILE_TIMEOUT_SECONDS", "600"))


# Files in-flight per repo job. Bounded by Semaphore inside _walk_and_index
# so TEI / LLM / Neo4j / Milvus can actually be utilized across files instead
# of serializing on one file at a time. Sweet spot depends on backend
# headroom — start at 8, raise toward 16-32 if the bottleneck moves further.
INDEX_FILE_CONCURRENCY = int(os.environ.get("INDEX_FILE_CONCURRENCY", "8"))


def _new_job(kind: str, source_id: str, source_label: str) -> dict:

    job_id = uuid.uuid4().hex
    _state._last_job_id = job_id
    job: dict[str, Any] = {
        "id": job_id,
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "source": source_label,
        "source_id": source_id,
        "total_files": 0,
        "processed_files": 0,
        "committed_files": 0,
        "total_chunks": 0,
        "current_file": "",
        "start_time": time.time(),
        "finished_at": 0.0,
        "last_log_time": 0.0,
        "error": None,
        "message": "",
    }
    _state._jobs[job_id] = job
    metrics.queue_depth.inc()
    metrics.job_creates.labels(status="queued").inc()
    return job


# Schemes git is allowed to use here; mirrors the GIT_ALLOW_PROTOCOL value
# passed to every clone/ls-remote invocation. `file` is deliberately absent.
_GIT_URL_SCHEMES = frozenset({"https", "http", "ssh", "git"})


# Exact hosts permitted, when configured. Empty = no host allow-list.
GIT_ALLOWED_HOSTS: frozenset[str] = frozenset(
    h.strip().lower()
    for h in os.environ.get("TREELOOM_GIT_ALLOWED_HOSTS", "").split(",")
    if h.strip()
)


# Also refuse RFC1918 / unique-local destinations. OFF by default: self-hosted
# git on a private network is a first-class use case for this product, so
# blocking it out of the box would break the common deployment. Operators who
# only ever clone from the public internet should turn this on.
GIT_BLOCK_PRIVATE = os.environ.get("TREELOOM_GIT_BLOCK_PRIVATE", "").lower() == "true"


# Whether git may follow HTTP redirects while cloning or listing refs.
#
# OFF by default, because a redirect is a hole straight through the host
# policy above: `_is_safe_git_url` judges the URL the caller submitted, but
# git connects to wherever that URL redirects. A submitted
# `https://attacker.example/repo` answering `302 ->
# http://169.254.169.254/latest/meta-data/` reaches exactly the address the
# check exists to refuse, and the refusal never fires because the URL it saw
# was a perfectly ordinary public one.
#
# git's own default is `initial`, which still follows the first redirect and
# so does not close this. The cost of `false` is that a moved or renamed
# upstream stops resolving until its URL is updated, and plain-http remotes
# that redirect to https must be given their https URL — hence the opt-out.
GIT_FOLLOW_REDIRECTS = (
    os.environ.get("TREELOOM_GIT_FOLLOW_REDIRECTS", "").lower() == "true"
)


# Permit git remotes on the loopback interface. OFF by default — a request to
# clone 127.0.0.1 is the shape of an SSRF probe, not of a deployment.
#
# The exception it exists for is a git daemon the operator started themselves:
# scripts/loadtest_merge_searchable.py serves throwaway fixture repos over
# `git://localhost/<repo>` precisely so the webhook path can be measured with
# no network and no credentials. Refusing loopback unconditionally makes that
# harness unrunnable, and a harness that cannot run stops being a check on
# anything. Named for what it does, so enabling it is a decision rather than
# a side effect of some broader "test mode".
GIT_ALLOW_LOOPBACK = (
    os.environ.get("TREELOOM_GIT_ALLOW_LOOPBACK", "").lower() == "true"
)


def _resolve_inside_clone(repo_path: str, rel_path: str) -> str | None:
    """Join *rel_path* onto *repo_path*, or None if it escapes.

    The path is attacker-influenced: it comes from the webhook payload's
    changed-files list, which is whatever the sender put there. An absolute
    path makes os.path.join discard repo_path entirely, and `..` segments walk
    out of it; realpath() also collapses a symlink planted inside the clone
    that points elsewhere.

    The boundary test appends os.sep so a sibling sharing a prefix
    (/tmp/treeloom_inc_A vs /tmp/treeloom_inc_AB) cannot pass.
    """
    if not rel_path or os.path.isabs(rel_path):
        return None
    root = os.path.realpath(repo_path)
    candidate = os.path.realpath(os.path.join(root, rel_path))
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    return candidate


def _git_env() -> dict[str, str]:
    """Environment for every git invocation that talks to a remote.

    GIT_ALLOW_PROTOCOL constrains the transport; the GIT_CONFIG_* triple is
    how a config value is injected without `-c`, which GitPython classifies
    as an unsafe option (and rightly — enabling `allow_unsafe_options` to
    pass one would also unlock `--upload-pack`).
    """
    env = {"GIT_ALLOW_PROTOCOL": "http:https:ssh:git"}
    if not GIT_FOLLOW_REDIRECTS:
        env |= {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.followRedirects",
            "GIT_CONFIG_VALUE_0": "false",
        }
    return env


def _git_url_host(url: str) -> str | None:
    """Best-effort host for a git URL, covering scp-like `user@host:path`."""
    from urllib.parse import urlsplit

    if "://" in url:
        try:
            host = urlsplit(url).hostname
        except ValueError:
            return None
        return host.lower() if host else None
    # scp-like: [user@]host:path  (no scheme, colon before the first slash)
    head = url.split("/", 1)[0]
    if ":" not in head:
        return None
    hostpart = head.rsplit(":", 1)[0]
    if "@" in hostpart:
        hostpart = hostpart.rsplit("@", 1)[1]
    return hostpart.lower() or None


def _host_is_forbidden(host: str) -> bool:
    """True when `host` names a destination git must never be pointed at.

    Always refused: link-local (which is where cloud metadata lives at
    169.254.169.254), unspecified, multicast and reserved. None of these is
    ever a legitimate git remote, so refusing them costs nothing.

    Loopback is refused unless TREELOOM_GIT_ALLOW_LOOPBACK is set — see that
    constant for the one case that needs it.

    RFC1918 / unique-local is refused only under TREELOOM_GIT_BLOCK_PRIVATE.

    A bare hostname is resolved only when that flag is on — resolving on the
    default path would add a DNS round-trip to every request for no gain,
    since private destinations are permitted there anyway.

    LIMITATION: resolution here and git's own resolution are separate lookups,
    so a DNS entry that changes between them (rebinding) is not prevented.
    Closing that needs resolve-then-connect-by-IP, which gitpython does not
    expose. Treat this as raising the bar, not as SSRF-proof.
    """
    import ipaddress
    import socket

    def _bad(ip: ipaddress._BaseAddress) -> bool:
        if ip.is_loopback:
            # Unspecified (0.0.0.0/::) stays refused regardless: it is not an
            # address a remote can live at, only a way of writing "somewhere
            # local" that the loopback opt-in was never meant to cover.
            return not GIT_ALLOW_LOOPBACK
        if (ip.is_link_local or ip.is_unspecified
                or ip.is_multicast or ip.is_reserved):
            return True
        return GIT_BLOCK_PRIVATE and ip.is_private

    try:
        return _bad(ipaddress.ip_address(host.strip("[]")))
    except ValueError:
        pass  # not a literal IP — a hostname

    if host in {"localhost", "localhost.localdomain"}:
        return not GIT_ALLOW_LOOPBACK
    if not GIT_BLOCK_PRIVATE:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False  # unresolvable: let git fail on it rather than guess
    return any(_bad(ipaddress.ip_address(i[4][0])) for i in infos)


def _is_safe_git_url(url: str) -> bool:
    """Return True if `url` is safe to hand to git.

    Rejects values that could be read as options (leading '-') or that use
    git's `transport::address` remote-helper form (ext::, file::, ...).

    Also validates the destination. Without this the only transport control
    was GIT_ALLOW_PROTOCOL=http:https:ssh:git, which constrains the *scheme*
    but not the *host* — so `git://10.0.0.50/secrets` or
    `http://169.254.169.254/` was accepted, cloned, embedded into the vector
    and graph stores, and then readable through /search. That is SSRF with
    durable exfiltration, not just a fetch (CWE-918).

    Policy, in order: known scheme -> host present -> host not forbidden ->
    host on TREELOOM_GIT_ALLOWED_HOSTS when one is configured.
    """
    if not url:
        return False
    if url.startswith("-"):
        return False
    if "::" in url:
        return False

    if "://" in url:
        scheme = url.split("://", 1)[0].lower()
        if scheme not in _GIT_URL_SCHEMES:
            return False

    host = _git_url_host(url)
    if not host:
        return False
    if _host_is_forbidden(host):
        return False
    if GIT_ALLOWED_HOSTS and host not in GIT_ALLOWED_HOSTS:
        return False
    return True


def _git_head_sha(path: str) -> str:
    """Return current HEAD SHA for a local git repo at `path`, '' if not a repo."""
    try:
        repo = Repo(path)
        sha = repo.head.commit.hexsha
        repo.close()
        return sha
    except Exception:
        return ""


def _log_progress(job: dict, force: bool = False):
    now = time.time()
    elapsed = now - job["start_time"]
    if not force and now - job.get("last_log_time", 0) < LOG_INTERVAL:
        return
    if not force and job["processed_files"] % LOG_FILE_INTERVAL != 0:
        return
    job["last_log_time"] = now
    remaining = job["total_files"] - job["processed_files"]
    rate = job["processed_files"] / elapsed if elapsed > 0 else 0
    eta = remaining / rate if rate > 0 else 0
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] job={job['job_id'][:8]} "
        f"{job['processed_files']}/{job['total_files']} files "
        f"({remaining} remaining), "
        f"{job['total_chunks']} chunks, "
        f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s, "
        f"current: {job['current_file']}"
    )


async def _graph_index_file(file_path: str, source_id: str):
    with tracer.start_as_current_span(
        "graph.index_file",
        attributes={"treeloom.file_path": file_path, "treeloom.source_id": source_id},
    ) as span:
        try:
            source = read_source(file_path)
        except Exception as exc:
            _mark_span_error(span, exc)
            return
        entities, relationships = extract_graph(file_path, source)
        span.set_attribute("treeloom.entity_count", len(entities))
        span.set_attribute("treeloom.relationship_count", len(relationships))
        if entities:
            await graph_store.store_graph(entities, relationships, source_id=source_id)


def _collect_files(path: str, skip_patterns: list[str] | None = None) -> list[str]:
    import fnmatch
    # Includes markdown extensions (.md/.markdown/.mdx) — see SUPPORTED_EXTENSIONS.
    supported_exts = SUPPORTED_EXTENSIONS
    patterns = (skip_patterns or []) if FEATURE_SKIP_PATTERNS else []
    files = []
    for root, dirs, fnames in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in fnames:
            if fname in SKIP_FILENAMES:
                continue
            ext = os.path.splitext(fname)[1]
            if ext not in supported_exts:
                continue
            full = os.path.join(root, fname)
            if patterns:
                rel = os.path.relpath(full, path)
                if any(fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(rel, p) for p in patterns):
                    continue
            files.append(full)
    return sorted(files)


async def _summaries_for_chunks(chunks: list[dict]) -> list[str | None]:
    keys = [llm.chunk_cache_key(c["text"]) for c in chunks]
    with tracer.start_as_current_span(
        "summaries.cache_lookup",
        attributes={"treeloom.keys_requested": len(keys)},
    ) as span:
        cached = await llm.cache_get_many(keys)
        span.set_attribute("treeloom.cache_hits", len(cached))
        span.set_attribute("treeloom.cache_misses", len(keys) - len(cached))
    out: list[str | None] = [None] * len(chunks)
    pending: list[int] = []
    for i, key in enumerate(keys):
        if key in cached:
            out[i] = cached[key]
        else:
            pending.append(i)

    if pending:
        async def _one(i: int) -> tuple[int, str | None]:
            if OTEL_TRACE_PER_CHUNK:
                with tracer.start_as_current_span(
                    "summaries.generate",
                    attributes={
                        "treeloom.chunk_sha1": keys[i],
                        "treeloom.llm_model": os.environ.get("LLM_MODEL", ""),
                    },
                ):
                    return i, await llm.summarize_with_cache(
                        chunks[i]["text"], chunks[i].get("language", ""), chunks[i]["file_path"]
                    )
            return i, await llm.summarize_with_cache(
                chunks[i]["text"], chunks[i].get("language", ""), chunks[i]["file_path"]
            )

        with tracer.start_as_current_span(
            "summaries.generate_batch",
            attributes={"treeloom.miss_count": len(pending)},
        ):
            results = await asyncio.gather(*(_one(i) for i in pending), return_exceptions=False)
        for i, summary in results:
            out[i] = summary
    return out


async def _process_file(
    file_path: str,
    source_id: str,
    *,
    skip_graph: bool = False,
) -> int:
    from treeloom.infrastructure.retry import retry_with_backoff, SkipError

    with tracer.start_as_current_span("parse_and_chunk") as span:
        try:
            chunks = await retry_with_backoff(
                lambda: asyncio.to_thread(indexer.parse_and_chunk, file_path),
                max_attempts=3,
                base_delay=1.0,
            )
        except SkipError:
            span.set_attribute("treeloom.skipped", True)
            return 0  # UnicodeDecodeError — skip binary file gracefully
        span.set_attribute("treeloom.chunk_count", len(chunks) if chunks else 0)

    if not chunks:
        return 0

    # Truncate oversized chunks to Milvus varchar limit (65535)
    MAX_CHUNK_BYTES = 65535
    max_bytes = 0
    for c in chunks:
        c["source_id"] = source_id
        raw = c["text"].encode("utf-8", errors="replace")
        metrics.chunk_size.observe(len(raw))
        if len(raw) > max_bytes:
            max_bytes = len(raw)
        if len(raw) > MAX_CHUNK_BYTES:
            c["text"] = raw[:MAX_CHUNK_BYTES].decode("utf-8", errors="replace")
            c["truncated"] = True

    texts = [c["text"] for c in chunks]
    with tracer.start_as_current_span(
        "embed.chunks",
        attributes={
            "treeloom.chunk_count": len(chunks),
            "treeloom.max_chunk_bytes": max_bytes,
        },
    ):
        with metrics.embed_latency.time():
            embeddings = await embed(texts)
    metrics.chunks_embedded.inc(len(chunks))

    summary_embs: list[list[float] | None] | None = None
    if USE_SUMMARY_VECTOR:
        summaries = await _summaries_for_chunks(chunks)
        to_embed = [s for s in summaries if s]
        if to_embed:
            with tracer.start_as_current_span(
                "embed.summaries",
                attributes={"treeloom.summary_count": len(to_embed)},
            ):
                embs = await embed(to_embed)
            it = iter(embs)
            summary_embs = [next(it) if s else None for s in summaries]
        else:
            summary_embs = [None] * len(chunks)

    with tracer.start_as_current_span("milvus.init_collection"):
        await init_collection()
    with tracer.start_as_current_span(
        "milvus.insert_chunks",
        attributes={"treeloom.row_count": len(chunks)},
    ):
        await insert_chunks(chunks, embeddings, summary_embeddings=summary_embs)
    if not skip_graph:
        await _graph_index_file(file_path, source_id)
    metrics.files_processed.inc()
    return len(chunks)


async def _walk_and_index(
    path: str, source_id: str, job: dict, skip_count: int = 0,
) -> tuple[int, int]:
    """Index every supported file under `path` with bounded concurrency.

    Concurrency model:
      - Up to INDEX_FILE_CONCURRENCY files in flight at once via a Semaphore.
      - Files complete in arbitrary order; progress counters update under
        a single-event-loop "atomic on each += 1" guarantee.
      - Persistence is serialized through a Lock so concurrent completions
        never race the row's snapshot.

    Resume model (best-effort, replaces the strict `committed_files`
    prefix model that doesn't survive out-of-order completion):
      - On `skip_count > 0` (resume), query Milvus for distinct file_paths
        already indexed under this source_id and skip those.
      - On crash mid-file, that file isn't in Milvus yet (chunks were
        accumulated locally but never inserted), so it'll be re-tried
        on the next resume. This is the same semantics as before for the
        crash case; what's new is that completed files past a crashed
        one are now recognized as done.
    """
    from treeloom.retriever import list_indexed_paths

    files = _collect_files(path, skip_patterns=job.get("skip_patterns"))
    job["total_files"] = len(files)

    if skip_count > 0:
        # Resume: rebuild the "already done" set from Milvus.
        try:
            done_paths = await list_indexed_paths(source_id)
        except Exception:
            logger.exception("[job %s] resume: failed to read Milvus paths, processing everything", job.get("job_id", "?"))
            done_paths = set()
        before = len(files)
        files = [f for f in files if f not in done_paths]
        already_done = before - len(files)
        job["processed_files"] = already_done
        job["committed_files"] = already_done
        total_chunks_ref = [job.get("total_chunks", 0)]
        logger.info(
            "[job %s] resuming: %d files already indexed, %d remaining",
            job.get("job_id", "?"), already_done, len(files),
        )
    else:
        job["processed_files"] = 0
        job["committed_files"] = 0
        job["total_chunks"] = 0
        total_chunks_ref = [0]

    sem = asyncio.Semaphore(INDEX_FILE_CONCURRENCY)
    persist_lock = asyncio.Lock()
    skip_graph = bool(job.get("skip_graph", False))

    async def _process_one(file_path: str) -> None:
        async with sem:
            # `current_file` is racy under concurrency; treat it as
            # "most-recently-started" for human-readable status.
            job["current_file"] = file_path
            file_start = time.time()
            had_error = False
            try:
                file_size = os.path.getsize(file_path)
            except Exception:
                file_size = 0
            language = detect_language(file_path) or ""
            with tracer.start_as_current_span(
                "index_file",
                attributes={
                    "treeloom.file_path": file_path,
                    "treeloom.file_size_bytes": file_size,
                    "treeloom.language": language,
                    "treeloom.source_id": source_id,
                    "treeloom.job_id": job.get("job_id", ""),
                },
            ) as file_span:
                try:
                    n = await asyncio.wait_for(
                        _process_file(file_path, source_id, skip_graph=skip_graph),
                        timeout=INDEX_FILE_TIMEOUT,
                    )
                except asyncio.TimeoutError as exc:
                    _mark_span_error(file_span, exc)
                    file_elapsed = time.time() - file_start
                    logger.error(
                        "[job %s] file timed out after %.1fs (ceiling=%ds) — skipping: %s",
                        job.get("job_id", "?"), file_elapsed, INDEX_FILE_TIMEOUT, file_path,
                    )
                    job["errors"] = (job.get("errors") or 0) + 1
                    n = 0
                    had_error = True
                    await _state._job_file_error_store.record(
                        job_id=job.get("job_id", ""),
                        file_path=file_path,
                        error_kind="timeout",
                        error_message=f"timed out after {file_elapsed:.1f}s",
                        elapsed_s=file_elapsed,
                    )
                except Exception as exc:
                    _mark_span_error(file_span, exc)
                    file_elapsed = time.time() - file_start
                    logger.exception(
                        "[job %s] file raised — skipping: %s (%s)",
                        job.get("job_id", "?"), file_path, exc,
                    )
                    job["errors"] = (job.get("errors") or 0) + 1
                    n = 0
                    had_error = True
                    await _state._job_file_error_store.record(
                        job_id=job.get("job_id", ""),
                        file_path=file_path,
                        error_kind="exception",
                        error_message=f"{type(exc).__name__}: {exc}"[:2000],
                        elapsed_s=file_elapsed,
                    )
                file_span.set_attribute("treeloom.chunk_count", n)

            # asyncio is single-threaded; bare += is atomic between awaits.
            total_chunks_ref[0] += n
            job["processed_files"] += 1
            job["committed_files"] = job["processed_files"]
            job["total_chunks"] = total_chunks_ref[0]
            _log_progress(job)
            if had_error or job["processed_files"] % LOG_FILE_INTERVAL == 0:
                async with persist_lock:
                    await _persist_job(job)

    await asyncio.gather(*(_process_one(f) for f in files), return_exceptions=False)

    async with persist_lock:
        await _persist_job(job)
    _log_progress(job, force=True)
    return total_chunks_ref[0], len(files)


async def _reset_source(source_id: str):
    """Drop existing chunks + entities for a source so re-index is an upsert."""
    try:
        delete_chunks_by_source(source_id)
    except Exception as e:
        print(f"warning: delete_chunks_by_source failed: {e}")
    try:
        await graph_store.delete_source(source_id)
    except Exception as e:
        print(f"warning: delete_source failed: {e}")


def _terminal_status_for_index(
    total_chunks: int, attempted_units: int, errors: int
) -> str:
    """Resolve an index job's terminal status from its outcome counts.

    Returns ``"failed"`` only when the job actually attempted work but *every*
    unit errored and nothing was indexed — i.e. ``attempted_units > 0`` and
    ``errors >= attempted_units`` and ``total_chunks == 0``. Otherwise
    ``"done"``.

    ``attempted_units`` is the count of files the job tried to chunk/embed
    (full/file jobs: all files; incremental: modified/added files, excluding
    pure removals which are successful no-ops). This deliberately keeps two
    legitimate cases as ``done``: a genuine no-op (no units, or zero errors —
    e.g. empty files that yield no chunks) and a *partial* success (some chunks
    landed even though other files errored). It only flips a run that produced
    zero chunks because all of its work failed — the failure mode that
    otherwise reads as success at a glance (a misconfigured/unreachable
    embedding or vector backend).
    """
    if attempted_units > 0 and errors > 0 and errors >= attempted_units and total_chunks == 0:
        return "failed"
    return "done"


async def _finalize_incremental_job(
    job: dict, total_chunks: int, files_processed: int, message: str,
):
    """Finalize an incremental job without touching the source registry.

    Source registry totals (file_count/chunk_count) reflect the full source,
    not this delta. Only persist the job state so dedup + status queries work.

    Status is ``done`` normally, but ``failed`` when every modified/added file
    errored and produced no chunks — pure removals are excluded from the
    attempt count, so a removal-only delta stays ``done``.
    """
    attempted = files_processed - int(job.get("files_removed") or 0)
    errors = int(job.get("errors") or 0)
    status = _terminal_status_for_index(total_chunks, attempted, errors)
    job["status"] = status
    job["finished_at"] = time.time()
    job["processed_files"] = files_processed
    job["total_chunks"] = total_chunks
    job["message"] = message
    if status == "failed":
        job["error"] = job.get("error") or (
            f"All {attempted} changed file(s) failed during incremental "
            f"indexing ({errors} error(s)); 0 chunks produced. See "
            f"GET /jobs/{job.get('job_id', '')}/errors for the per-file cause "
            f"(commonly an unreachable embedding/vector backend)."
        )
        logger.error(
            "incremental job %s finalized FAILED: %d/%d changed files errored, "
            "0 chunks", job.get("job_id", ""), errors, attempted,
        )
    metrics.active_jobs.dec()
    await _persist_job(job)


async def _finalize_job(job: dict, total_chunks: int, total_files: int, message: str):
    # a run that attempted every file, errored on all of them, and produced
    # zero chunks (e.g. the embedding/vector backend was unreachable) must not
    # report success. Mark it failed and skip the source-registry upsert so a
    # fully-broken run isn't recorded as an indexed source.
    errors = int(job.get("errors") or 0)
    if _terminal_status_for_index(total_chunks, total_files, errors) == "failed":
        job["status"] = "failed"
        job["finished_at"] = time.time()
        job["total_chunks"] = total_chunks
        job["total_files"] = total_files
        job["message"] = message
        job["error"] = job.get("error") or (
            f"All {total_files} file(s) failed during indexing "
            f"({errors} error(s)); 0 chunks produced. See "
            f"GET /jobs/{job.get('job_id', '')}/errors for the per-file cause "
            f"(commonly an unreachable embedding/vector backend)."
        )
        logger.error(
            "index job %s finalized FAILED: %d/%d files errored, 0 chunks",
            job.get("job_id", ""), errors, total_files,
        )
        metrics.active_jobs.dec()
        await _persist_job(job)
        return

    job["status"] = "done"
    job["finished_at"] = time.time()
    job["total_chunks"] = total_chunks
    job["total_files"] = total_files
    job["message"] = message
    _persist_job_sync(job)
    # Persist to graph store (Neo4j)
    await graph_store.upsert_source({
        "id": job["source_id"],
        "path": job.get("source_path", ""),
        "url": job.get("source_url", ""),
        "branch": job.get("source_branch", ""),
        "indexed_at": int(job["finished_at"]),
        "file_count": total_files,
        "chunk_count": total_chunks,
        "commit_sha": job.get("commit_sha", ""),
    })
    # Persist to PostgreSQL source registry (graceful degradation)
    try:
        # Two-pass indexing: chunks-only runs leave graph_indexed=False
        # so the operator can later run /index-graph for this source.
        # All-in-one and graph-only runs leave it True.
        skipped_graph = bool(job.get("skip_graph", False)) and job.get("kind") != "graph"
        # Persist how the source was indexed so the fleet auto-refresh loop can
        # re-enqueue the same kind. A graph-only re-pass must NOT relabel
        # the source — preserve the existing kind for it.
        job_kind = job.get("kind") or "repo"
        if job_kind in ("repo", "directory", "file"):
            source_kind = job_kind
        else:
            existing = await _state._source_repo.get_by_id(job["source_id"])
            source_kind = existing.kind if existing is not None else "repo"
        record = SourceRecord(
            id=job["source_id"],
            path=job.get("source_path", ""),
            url=job.get("source_url", ""),
            branch=job.get("source_branch", ""),
            indexed_at=datetime.fromtimestamp(job["finished_at"], tz=timezone.utc),
            file_count=total_files,
            chunk_count=total_chunks,
            commit_sha=job.get("commit_sha", ""),
            created_by=job.get("created_by"),
            kind=source_kind,
            graph_indexed=not skipped_graph,
            graph_indexed_at=(
                None if skipped_graph else
                datetime.fromtimestamp(job["finished_at"], tz=timezone.utc)
            ),
        )
        await _state._source_repo.save(record)
    except Exception:
        pass  # Degraded mode — graph store is the primary persistence
    metrics.active_jobs.dec()
    await _persist_job(job)


def _fail_job(job: dict, exc: BaseException):
    import traceback as tb

    job["status"] = "failed"
    tb_str = tb.format_exc()
    job["error"] = f"{type(exc).__name__}: {exc}"
    if tb_str and tb_str.strip():
        # Append traceback (truncated to fit in the error field) so
        # operators can diagnose without hunting through container logs.
        job["error"] += f"\n{tb_str[-4000:]}"
    job["finished_at"] = time.time()
    _persist_job_sync(job)
    metrics.active_jobs.dec()
    metrics.queue_depth.dec()
    # Structured log
    from treeloom.infrastructure.logging import log as struct_log
    struct_log.error(
        "job_failed",
        job_id=job.get("job_id"),
        source=job.get("source", ""),
        error_type=type(exc).__name__,
        error=str(exc),
        processed_files=job.get("processed_files", 0),
        total_files=job.get("total_files", 0),
    )
    # Increment embed_errors for embed/Milvus failures
    err_type = type(exc).__name__
    if any(k in err_type for k in ("HTTP", "Connect", "Timeout", "Milvus")):
        metrics.embed_errors.inc()


async def _run_index_file_job(job: dict, file_path: str):
    resuming = job.pop("_resume", False)
    job_attrs = {
        "treeloom.job_id": job.get("job_id", ""),
        "treeloom.kind": "file",
        "treeloom.source_id": job.get("source_id", ""),
        "treeloom.file_path": file_path,
        "treeloom.resuming": resuming,
    }
    with tracer.start_as_current_span("index_file", attributes=job_attrs) as span:
        try:
            job["status"] = "running"
            metrics.active_jobs.inc()
            metrics.queue_depth.dec()
            job["total_files"] = 1
            job["current_file"] = file_path
            if resuming and job.get("committed_files", 0) >= 1:
                await _finalize_job(
                    job, job.get("total_chunks", 0), 1,
                    f"Already indexed (resumed) {file_path}",
                )
                return
            if not resuming:
                await _reset_source(job["source_id"])
                job["commit_sha"] = _git_head_sha(os.path.dirname(file_path) or ".")
            # See the note at _process_file: this is CPU-bound tree-sitter
            # work and blocks the event loop when awaited inline.
            chunks = await asyncio.to_thread(indexer.parse_and_chunk, file_path)
            if not chunks:
                job["committed_files"] = 1
                await _finalize_job(job, 0, 0, "No chunks extracted")
                return
            for c in chunks:
                c["source_id"] = job["source_id"]
            texts = [c["text"] for c in chunks]
            embeddings = await embed(texts)
            await init_collection()
            await insert_chunks(chunks, embeddings)
            await _graph_index_file(file_path, job["source_id"])
            job["processed_files"] = 1
            job["committed_files"] = 1
            job["total_chunks"] = len(chunks)
            span.set_attribute("treeloom.chunk_count", len(chunks))
            _log_progress(job, force=True)
            await _finalize_job(
                job, len(chunks), 1,
                f"Indexed {len(chunks)} chunks from {file_path}",
            )
        except Exception as exc:
            _mark_span_error(span, exc)
            _fail_job(job, exc)


async def _run_index_directory_job(job: dict, directory: str, pattern: str):
    resuming = job.pop("_resume", False)
    skip_count = job.get("committed_files", 0) if resuming else 0
    job_attrs = {
        "treeloom.job_id": job.get("job_id", ""),
        "treeloom.kind": "directory",
        "treeloom.source_id": job.get("source_id", ""),
        "treeloom.source_path": directory,
        "treeloom.resuming": resuming,
    }
    try:
        job["status"] = "running"
        metrics.active_jobs.inc()
        metrics.queue_depth.dec()
        if not resuming:
            await _reset_source(job["source_id"])
            job["committed_files"] = 0
            job["processed_files"] = 0
            job["total_chunks"] = 0
            job["commit_sha"] = _git_head_sha(directory)
        root = Path(directory)
        files = []
        for fp in root.rglob(pattern):
            if fp.is_dir() or fp.name.startswith("."):
                continue
            if detect_language(str(fp)):
                files.append(str(fp))
        files.sort()
        job["total_files"] = len(files)
        if resuming:
            job["processed_files"] = min(skip_count, len(files))

        total_chunks = job.get("total_chunks", 0) if resuming else 0
        chunk_buffer: list[dict] = []
        files_in_buffer: list[str] = []

        async def _flush_buffer():
            nonlocal total_chunks
            if not chunk_buffer:
                if files_in_buffer:
                    job["committed_files"] = job.get("committed_files", 0) + len(files_in_buffer)
                    files_in_buffer.clear()
                    _persist_job_sync(job)
                return
            try:
                # Batch flush spans are intentionally root traces (they span
                # multiple files), tagged with job_id so they're discoverable
                # alongside the per-file traces in TraceQL.
                with tracer.start_as_current_span(
                    "embed.chunks",
                    attributes={"treeloom.chunk_count": len(chunk_buffer), **job_attrs},
                ):
                    texts = [c["text"] for c in chunk_buffer]
                    embeddings = await embed(texts)
                metrics.chunks_embedded.inc(len(chunk_buffer))
                with tracer.start_as_current_span(
                    "milvus.init_collection", attributes=job_attrs,
                ):
                    await init_collection()
                with tracer.start_as_current_span(
                    "milvus.insert_chunks",
                    attributes={"treeloom.row_count": len(chunk_buffer), **job_attrs},
                ):
                    await insert_chunks(chunk_buffer, embeddings)
                total_chunks += len(chunk_buffer)
                job["committed_files"] = job.get("committed_files", 0) + len(files_in_buffer)
                job["total_chunks"] = total_chunks
                chunk_buffer.clear()
                files_in_buffer.clear()
                _persist_job_sync(job)
            except Exception as exc:
                # Batch flush failure: log, record files as errored, discard the
                # batch, and continue — do NOT fail the entire job.
                logger.exception(
                    "[job %s] batch flush failed (%d chunks, %d files): %s — discarding batch",
                    job.get("job_id", "?"), len(chunk_buffer), len(files_in_buffer), exc,
                )
                job["errors"] = job.get("errors", 0) + len(files_in_buffer)
                chunk_buffer.clear()
                files_in_buffer.clear()
                _persist_job_sync(job)

        for file_path in files[skip_count:]:
            job["current_file"] = file_path
            with tracer.start_as_current_span(
                "index_file",
                attributes={
                    "treeloom.file_path": file_path,
                    **job_attrs,
                },
            ) as file_span:
                with tracer.start_as_current_span(
                    "parse_and_chunk",
                    attributes={"treeloom.file_path": file_path},
                ) as chunk_span:
                    try:
                        # CPU-bound tree-sitter parse + chunk. Called inline
                        # from this async runner it froze the whole event
                        # loop for its duration — search, /health, the queue
                        # poller and the webhook endpoint all stall
                        # (CWE-400), and this one is inside a per-file loop,
                        # so indexing a repo stalled search for the entire
                        # job. Safe in a thread: _make_chunker builds a fresh
                        # parser per call, which is why _process_file already
                        # does exactly this.
                        chunks = await asyncio.to_thread(
                            indexer.parse_and_chunk, file_path
                        )
                    except Exception as exc:
                        _mark_span_error(chunk_span, exc)
                        _mark_span_error(file_span, exc)
                        logger.warning("Failed to chunk %s: %s", file_path, exc)
                        job["processed_files"] += 1
                        job["errors"] = job.get("errors", 0) + 1
                        files_in_buffer.append(file_path)
                        _log_progress(job)
                        continue
                    chunk_span.set_attribute("treeloom.chunk_count", len(chunks) if chunks else 0)
                if not chunks:
                    file_span.set_attribute("treeloom.chunk_count", 0)
                    job["processed_files"] += 1
                    files_in_buffer.append(file_path)
                    _log_progress(job)
                    continue
                for c in chunks:
                    c["source_id"] = job["source_id"]
                chunk_buffer.extend(chunks)
                files_in_buffer.append(file_path)
                job["processed_files"] += 1
                job["total_chunks"] = total_chunks + len(chunk_buffer)
                file_span.set_attribute("treeloom.chunk_count", len(chunks))
                _log_progress(job)
            if len(chunk_buffer) >= MAX_BATCH_SIZE:
                await _flush_buffer()

        await _flush_buffer()
        metrics.files_processed.inc(job["processed_files"])
        _log_progress(job, force=True)

        if total_chunks > 0:
            try:
                n_comm = await community.run_post_index_signals(job["source_id"])
                logger.info(
                    "post-index signals: %d communities for source %s",
                    n_comm, job["source_id"],
                )
            except Exception:
                # Enrichment only — never fail the index job over it, but log
                # loudly (the old fire-and-forget version swallowed errors and
                # got killed on process exit, leaving zero community signals).
                logger.exception(
                    "post-index signals failed for source %s", job["source_id"],
                )
        await _finalize_job(
            job, total_chunks, len(files),
            f"Indexed {total_chunks} chunks from {len(files)} files",
        )
    except Exception as exc:
        with tracer.start_as_current_span("job.failed", attributes=job_attrs) as err_span:
            _mark_span_error(err_span, exc)
        _fail_job(job, exc)


async def _run_index_repo_job(
    job: dict,
    req: "_RepoJobSpec | Any",
):
    cleanup_path: str | None = None
    resuming = job.pop("_resume", False)
    skip_count = job.get("committed_files", 0) if resuming else 0
    job_attrs = {
        "treeloom.job_id": job.get("job_id", ""),
        "treeloom.kind": "repo",
        "treeloom.source_id": job.get("source_id", ""),
        "treeloom.source_path": req.path or "",
        "treeloom.source_url": req.url or "",
        "treeloom.source_branch": req.branch or "",
        "treeloom.resuming": resuming,
        "treeloom.force": bool(req.force),
    }
    try:
        job["status"] = "running"
        metrics.active_jobs.inc()
        metrics.queue_depth.dec()
        if not resuming:
            await _reset_source(job["source_id"])
            job["committed_files"] = 0

        repo_path = req.path
        branch = req.branch
        if req.url:
            if not _is_safe_git_url(req.url):
                raise ValueError(f"Unsafe git URL: {req.url!r}")
            if branch and branch.startswith("-"):
                raise ValueError(f"Unsafe git branch: {branch!r}")
            repo_path = tempfile.mkdtemp(prefix="treeloom_")
            cleanup_path = repo_path
            kwargs: dict = {
                "depth": 1,
                "env": _git_env(),
                "allow_unsafe_protocols": False,
                "allow_unsafe_options": False,
            }
            if branch:
                kwargs["branch"] = branch
            with tracer.start_as_current_span(
                "git.clone",
                attributes={"treeloom.source_url": req.url, "treeloom.branch": branch or "", **job_attrs},
            ):
                Repo.clone_from(req.url, repo_path, **kwargs)
        else:
            repo_path = os.path.expanduser(req.path)
            try:
                repo = Repo(repo_path)
                branch = branch or repo.active_branch.name
                repo.close()
            except Exception:
                pass

        job["source_branch"] = branch or ""
        job["commit_sha"] = _git_head_sha(repo_path)

        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Path not found: {repo_path}")

        total_chunks, total_files = await _walk_and_index(
            repo_path, job["source_id"], job, skip_count=skip_count,
        )

        if total_chunks > 0:
            try:
                n_comm = await community.run_post_index_signals(job["source_id"])
                logger.info(
                    "post-index signals: %d communities for source %s",
                    n_comm, job["source_id"],
                )
            except Exception:
                # Enrichment only — never fail the index job over it, but log
                # loudly (the old fire-and-forget version swallowed errors and
                # got killed on process exit, leaving zero community signals).
                logger.exception(
                    "post-index signals failed for source %s", job["source_id"],
                )

        branch_info = f" (branch: {branch})" if branch else ""
        await _finalize_job(
            job, total_chunks, total_files,
            f"Indexed {total_chunks} chunks from {total_files} files in {job['source']}{branch_info}",
        )
    except Exception as exc:
        with tracer.start_as_current_span("job.failed", attributes=job_attrs) as err_span:
            _mark_span_error(err_span, exc)
        _fail_job(job, exc)
    finally:
        if cleanup_path:
            shutil.rmtree(cleanup_path, ignore_errors=True)


# ── graph-only job runner (two-pass indexing pass 2) ─────────────
async def _run_index_graph_job(job: dict, source_id: str):
    """Run graph extraction over an already-chunks-indexed source.

    Pass 2 of the two-pass model. Source must have been chunks-indexed
    by `/index-repo skip_graph=true` (or any prior all-in-one run).
    Drops the source's existing graph entities via `delete_source(...)`
    so re-runs are idempotent, then walks the same file set with the
    same Semaphore-bounded concurrency + per-file timeout the regular
    path uses — but only calls `_graph_index_file`, no embed/Milvus.
    """
    job.pop("_resume", False)  # graph job has no resume state to skip files
    job_attrs = {
        "treeloom.job_id": job.get("job_id", ""),
        "treeloom.kind": "graph",
        "treeloom.source_id": source_id,
    }
    try:
        job["status"] = "running"
        metrics.active_jobs.inc()
        metrics.queue_depth.dec()

        record = await _state._source_repo.get_by_id(source_id)
        if record is None:
            raise FileNotFoundError(
                f"source_id={source_id!r} not found in source_records — "
                f"chunks-index it first before running /index-graph."
            )
        target_path = record.path
        if not target_path or not os.path.isdir(target_path):
            raise FileNotFoundError(
                f"Source path {target_path!r} not accessible — cannot graph-index."
            )

        # Idempotent re-run: drop the prior graph for this source.
        await graph_store.delete_source(source_id)

        # Re-collect files. The skip-patterns flag, if any, was stored on
        # the original repo job — for graph jobs we honor it as well.
        files = _collect_files(target_path, skip_patterns=job.get("skip_patterns"))
        job["total_files"] = len(files)
        job["processed_files"] = 0
        job["committed_files"] = 0
        job["total_chunks"] = 0

        sem = asyncio.Semaphore(INDEX_FILE_CONCURRENCY)
        persist_lock = asyncio.Lock()

        async def _graph_one(file_path: str) -> None:
            async with sem:
                job["current_file"] = file_path
                file_start = time.time()
                had_error = False
                with tracer.start_as_current_span(
                    "index_file",
                    attributes={
                        "treeloom.file_path": file_path,
                        **job_attrs,
                    },
                ) as file_span:
                    try:
                        await asyncio.wait_for(
                            _graph_index_file(file_path, source_id),
                            timeout=INDEX_FILE_TIMEOUT,
                        )
                    except asyncio.TimeoutError as exc:
                        _mark_span_error(file_span, exc)
                        file_elapsed = time.time() - file_start
                        logger.error(
                            "[job %s graph] file timed out after %.1fs — skipping: %s",
                            job.get("job_id", "?"), file_elapsed, file_path,
                        )
                        job["errors"] = (job.get("errors") or 0) + 1
                        had_error = True
                        await _state._job_file_error_store.record(
                            job_id=job.get("job_id", ""),
                            file_path=file_path,
                            error_kind="timeout",
                            error_message=f"graph step timed out after {file_elapsed:.1f}s",
                            elapsed_s=file_elapsed,
                        )
                    except Exception as exc:
                        _mark_span_error(file_span, exc)
                        file_elapsed = time.time() - file_start
                        logger.exception(
                            "[job %s graph] file raised — skipping: %s (%s)",
                            job.get("job_id", "?"), file_path, exc,
                        )
                        job["errors"] = (job.get("errors") or 0) + 1
                        had_error = True
                        await _state._job_file_error_store.record(
                            job_id=job.get("job_id", ""),
                            file_path=file_path,
                            error_kind="exception",
                            error_message=f"{type(exc).__name__}: {exc}"[:2000],
                            elapsed_s=file_elapsed,
                        )
                job["processed_files"] += 1
                job["committed_files"] = job["processed_files"]
                _log_progress(job)
                if had_error or job["processed_files"] % LOG_FILE_INTERVAL == 0:
                    async with persist_lock:
                        await _persist_job(job)

        await asyncio.gather(*(_graph_one(f) for f in files), return_exceptions=False)

        # Flip graph_indexed=True only if at least one file succeeded;
        # the operator can re-submit to retry remaining failures.
        had_any_success = job["processed_files"] > (job.get("errors") or 0)
        if had_any_success:
            from datetime import datetime, timezone
            await _state._source_repo.set_graph_indexed(
                source_id, True, datetime.now(timezone.utc),
            )

        await _finalize_job(
            job, 0, len(files),
            f"Graph-indexed {len(files) - (job.get('errors') or 0)} of {len(files)} "
            f"files for source {source_id}",
        )
    except Exception as exc:
        with tracer.start_as_current_span("job.failed", attributes=job_attrs) as err_span:
            _mark_span_error(err_span, exc)
        _fail_job(job, exc)




async def _run_incremental_index(
    source_id: str,
    repo_url: str,
    branch: str,
    changed_files: list[dict[str, str]],
    job: dict,
):
    """Run an incremental index job for a set of changed files.

    Routed through the Postgres job queue so it survives worker restarts —
    ``changed_files`` is persisted in ``job["payload"]`` and can be picked
    up by any indexer process. Re-runs are **idempotent**: each file is
    delete-then-insert, so crashing mid-job and re-running from the start
    is safe. Retries are managed by the queue worker (see postgres_queue.py).

    For each file:
    - ``removed`` → delete chunks + entities.
    - ``modified`` / ``added`` → delete existing chunks, then clone the
      repo, parse the file, embed, and insert fresh chunks.
    """
    job_start = time.time()
    # Recover webhook accept_time from payload for merge→searchable metric.
    payload = job.get("payload") or {}
    accept_time: float | None = payload.get("accept_time")

    cleanup_path: str | None = None
    try:
        job["status"] = "running"
        metrics.active_jobs.inc()
        metrics.queue_depth.dec()

        files_total = len(changed_files)
        files_processed = 0
        files_removed = 0
        files_failed = 0
        chunks_added = 0

        # Separate removed files from to-process files
        removed_paths = [f["path"] for f in changed_files if f["action"] == "removed"]
        process_files = [
            f for f in changed_files if f["action"] in ("modified", "added")
        ]

        # Handle removed files
        for file_path in removed_paths:
            try:
                delete_chunks_by_file(source_id, file_path)
            except Exception:
                pass
            try:
                await delete_entities_by_file(source_id, file_path)
            except Exception:
                pass
            files_removed += 1
            files_processed += 1
            job["files_removed"] = files_removed
            job["files_processed"] = files_processed
            job["committed_files"] = files_processed
            _persist_job_sync(job)

        if not process_files:
            job["total_files"] = files_total
            job["committed_files"] = files_processed
            job["files_removed"] = files_removed
            await _finalize_incremental_job(
                job, chunks_added, files_processed,
                f"Incremental index complete: {files_processed}/{files_total} files "
                f"({files_removed} removed, {chunks_added} chunks added)",
            )
            # Pure-removal delta: always "done" (removals are successful no-ops),
            # but read it back for consistency with the failure-aware finalize.
            _record_incremental_metrics(job_start, accept_time, status=job["status"])
            return

        # Clone the repo for modified/added files
        if not _is_safe_git_url(repo_url):
            raise ValueError(f"Unsafe git URL: {repo_url!r}")
        if branch and branch.startswith("-"):
            raise ValueError(f"Unsafe git branch: {branch!r}")
        repo_path = tempfile.mkdtemp(prefix="treeloom_inc_")
        cleanup_path = repo_path
        kwargs: dict = {
            "depth": 1,
            "env": _git_env(),
            "allow_unsafe_protocols": False,
            "allow_unsafe_options": False,
        }
        if branch:
            kwargs["branch"] = branch
        Repo.clone_from(repo_url, repo_path, **kwargs)

        # Process modified/added files
        for file_info in process_files:
            file_path = file_info["path"]
            full_path = _resolve_inside_clone(repo_path, file_path)
            if full_path is None:
                # The path came from the webhook payload and was joined onto
                # the clone directory unchecked, so `../../etc/passwd` resolved
                # outside it — and the file was then chunked, embedded and made
                # searchable, which turns arbitrary file read into durable
                # exfiltration through /search (CWE-22).
                logger.warning(
                    "Incremental index: refusing path outside the clone: %r",
                    file_path,
                )
                files_failed += 1
                files_processed += 1
                continue

            job["current_file"] = file_path

            # Delete existing chunks for this file
            try:
                delete_chunks_by_file(source_id, file_path)
            except Exception:
                pass

            if not os.path.isfile(full_path):
                files_failed += 1
                files_processed += 1
                continue

            # Parse and index the file
            try:
                # Same reason as the other runners — off the event loop.
                chunks = await asyncio.to_thread(indexer.parse_and_chunk, full_path)
                if not chunks:
                    files_processed += 1
                    continue

                for c in chunks:
                    c["source_id"] = source_id
                    c["file_path"] = file_path

                texts = [c["text"] for c in chunks]
                embeddings = await embed(texts)

                await init_collection()
                await insert_chunks(chunks, embeddings)

                await _graph_index_file(full_path, source_id)

                chunks_added += len(chunks)
                job["total_chunks"] = chunks_added
            except Exception:
                files_failed += 1

            files_processed += 1
            job["files_processed"] = files_processed
            job["committed_files"] = files_processed
            _persist_job_sync(job)

        job["total_files"] = files_total
        job["files_removed"] = files_removed
        job["errors"] = files_failed
        await _finalize_incremental_job(
            job, chunks_added, files_processed,
            f"Incremental index complete: {files_processed}/{files_total} files "
            f"({files_removed} removed, {files_failed} failed, "
            f"{chunks_added} chunks added)",
        )
        # status is "failed" when every changed file errored — keep the
        # incremental metric label consistent with the persisted job status.
        _record_incremental_metrics(job_start, accept_time, status=job["status"])

    except Exception as exc:
        _record_incremental_metrics(job_start, accept_time, status="failed")
        _fail_job(job, exc)
        raise  # Re-raise so the queue worker can apply retry/dead-letter logic.
    finally:
        if cleanup_path:
            shutil.rmtree(cleanup_path, ignore_errors=True)


def _record_incremental_metrics(
    job_start: float, accept_time: float | None, status: str
) -> None:
    """Record duration and merge→searchable histograms + status counter."""
    try:
        now = time.time()
        metrics.incremental_jobs_total.labels(status=status).inc()
        metrics.incremental_job_duration_seconds.observe(now - job_start)
        if accept_time is not None:
            metrics.merge_to_searchable_seconds.observe(now - accept_time)
    except Exception:
        pass


def _git_remote_sha(url: str, branch: str = "") -> str:
    """Resolve HEAD SHA for a remote repo via ls-remote, '' on failure.

    Hardened against argument-injection and transport-smuggling:
    rejects url/branch values starting with '-', rejects '::' transport
    smuggling (ext::/file::), passes '--' to terminate options, and
    restricts allowed transports via GIT_ALLOW_PROTOCOL.
    """
    if not _is_safe_git_url(url):
        return ""
    if branch.startswith("-"):
        return ""
    try:
        from git.cmd import Git
        ref = f"refs/heads/{branch}" if branch else "HEAD"
        g = Git()
        g.update_environment(**_git_env())
        out = g.ls_remote("--", url, ref)
        if not out:
            return ""
        return out.split()[0]
    except Exception:
        return ""


async def compute_source_staleness(src: dict) -> dict:
    """Compute staleness for a source record dict.

    Shared by ``GET /sources/{id}/staleness`` and the freshness sampler
    background task so there is only one place that does the git-SHA
    resolution logic.

    Returns a dict with keys: indexed_sha, current_sha, is_stale (bool|None),
    resolved_via (local|ls-remote|none), indexed_at (float|None).

    Notes:
    - ``git ls-remote`` gives a SHA, not a commit timestamp, so we cannot
      compute "commits/seconds behind HEAD" for URL-backed sources.
    - Use the ``is_stale`` flag combined with ``indexed_at`` age for SLO
      alerting (e.g. stale for more than N minutes).
    """
    indexed_sha = src.get("commit_sha") or ""
    branch = src.get("branch") or ""
    path = src.get("path") or ""
    url = src.get("url") or ""
    indexed_at = src.get("indexed_at")  # may be int/float/None

    if path:
        current_sha = await asyncio.to_thread(_git_head_sha, path)
        resolved_via = "local"
    elif url:
        current_sha = await asyncio.to_thread(_git_remote_sha, url, branch)
        resolved_via = "ls-remote"
    else:
        current_sha = ""
        resolved_via = "none"

    is_stale: bool | None
    if indexed_sha and current_sha:
        is_stale = indexed_sha != current_sha
    else:
        is_stale = None

    return {
        "indexed_sha": indexed_sha,
        "current_sha": current_sha,
        "resolved_via": resolved_via,
        "is_stale": is_stale,
        "indexed_at": indexed_at,
    }


