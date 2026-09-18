"""Fleet-level aggregation — pure functions, no I/O.

The ``GET /fleet`` endpoint and the in-process refresh scheduler both wire live
data (SourceRecord rows, JobStore rows, staleness, entity counts) into these
helpers. Keeping the logic pure makes the rollup and the refresh-selection
behaviour unit-testable without any services.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def _source_label(src: dict) -> str:
    return src.get("url") or src.get("path") or src.get("id") or ""


def _latest_job_by_source(jobs: list[dict]) -> dict[str, dict]:
    """Pick the most-recent job per ``source_id`` (max ``start_time``)."""
    latest: dict[str, dict] = {}
    for job in jobs:
        sid = job.get("source_id") or ""
        if not sid:
            continue
        cur = latest.get(sid)
        if cur is None or (job.get("start_time") or 0) >= (cur.get("start_time") or 0):
            latest[sid] = job
    return latest


def _job_summary(job: Optional[dict]) -> Optional[dict]:
    if job is None:
        return None
    return {
        "job_id": job.get("job_id") or job.get("id") or "",
        "status": job.get("status") or "",
        "error": job.get("error") or "",
        "errors": int(job.get("errors") or 0),
        "finished_at": job.get("finished_at"),
    }


def _has_error(job: Optional[dict]) -> bool:
    if job is None:
        return False
    if (job.get("status") or "") in ("failed", "dead_letter"):
        return True
    return int(job.get("errors") or 0) > 0


def build_fleet_health(
    sources: list[dict],
    jobs: list[dict],
    *,
    staleness_by_id: Optional[dict[str, Optional[bool]]] = None,
    entity_counts: Optional[dict[str, int]] = None,
    staleness_truncated: bool = False,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Roll up per-source health into a single fleet view.

    ``sources`` are SourceRecord-shaped dicts (as ``GET /sources`` returns),
    ``jobs`` are ``Job.to_dict()`` rows. ``staleness_by_id`` / ``entity_counts``
    are optional enrichments — when omitted, their summary/row fields are simply
    absent so the rollup still answers "is the fleet healthy?" with whatever is
    cheaply available.
    """
    now = time.time() if now is None else now
    latest = _latest_job_by_source(jobs)

    rows: list[dict] = []
    total_chunks = 0
    total_files = 0
    total_entities = 0
    sources_with_errors = 0
    sources_graph_missing = 0
    sources_stale = 0
    oldest_age: Optional[float] = None

    for src in sources:
        sid = src.get("id") or src.get("source_id") or ""
        job = latest.get(sid)
        indexed_at = src.get("indexed_at")
        age: Optional[float] = None
        if indexed_at is not None:
            try:
                age = now - float(indexed_at)
            except (TypeError, ValueError):
                age = None

        chunk_count = int(src.get("chunk_count") or 0)
        file_count = int(src.get("file_count") or 0)
        graph_indexed = bool(src.get("graph_indexed", True))

        total_chunks += chunk_count
        total_files += file_count
        if not graph_indexed:
            sources_graph_missing += 1
        if _has_error(job):
            sources_with_errors += 1
        if age is not None and (oldest_age is None or age > oldest_age):
            oldest_age = age

        row: dict[str, Any] = {
            "id": sid,
            "label": _source_label(src),
            "branch": src.get("branch") or "",
            "indexed_at": indexed_at,
            "age_seconds": age,
            "chunk_count": chunk_count,
            "file_count": file_count,
            "graph_indexed": graph_indexed,
            "last_job": _job_summary(job),
        }

        if entity_counts is not None:
            ec = int(entity_counts.get(sid, 0))
            total_entities += ec
            row["entity_count"] = ec

        if staleness_by_id is not None:
            is_stale = staleness_by_id.get(sid)
            row["is_stale"] = is_stale
            if is_stale:
                sources_stale += 1

        rows.append(row)

    summary: dict[str, Any] = {
        "sources_total": len(sources),
        "sources_with_errors": sources_with_errors,
        "sources_graph_missing": sources_graph_missing,
        "total_chunks": total_chunks,
        "total_files": total_files,
        "oldest_index_age_seconds": oldest_age,
    }
    if entity_counts is not None:
        summary["total_entities"] = total_entities
    if staleness_by_id is not None:
        summary["sources_stale"] = sources_stale
        if staleness_truncated:
            summary["staleness_truncated"] = True

    return {"summary": summary, "sources": rows}


_INDEX_KINDS = ("repo", "directory", "file")


def resolve_reindex_kind(*, stored_kind: str | None, has_url: bool, path_is_file: bool) -> str:
    """Decide which job kind to re-enqueue for a source during auto-refresh.

    The auto-refresh loop must NOT blindly re-index every stale source as a
    ``repo`` job: a file-indexed source would fail the runner's
    ``isdir`` check every tick, and a directory-indexed source would be
    silently re-indexed with repo-walk semantics. When the persisted
    ``stored_kind`` is known, trust it. For legacy rows that predate the
    ``kind`` column (``stored_kind`` empty/None), fall back to filesystem
    shape: a URL is a repo, an existing file path is a file, otherwise repo.
    """
    if stored_kind in _INDEX_KINDS:
        return stored_kind  # type: ignore[return-value]
    if has_url:
        return "repo"
    if path_is_file:
        return "file"
    return "repo"


def select_sources_to_refresh(
    sources: list[dict],
    *,
    max_per_tick: int,
) -> list[str]:
    """Choose which sources the auto-refresh loop should re-index this tick.

    Each ``sources`` dict must carry ``id``, ``is_stale`` (bool|None), and
    ``indexed_at``. Only sources known to be stale (``is_stale is True``) are
    selected; non-git / undetermined sources (``None``) are skipped. Oldest-
    indexed first, capped at ``max_per_tick`` to bound the per-tick load.
    """
    if max_per_tick <= 0:
        return []

    candidates = [s for s in sources if s.get("is_stale") is True and (s.get("id"))]
    candidates.sort(key=lambda s: (s.get("indexed_at") is None, s.get("indexed_at") or 0))
    return [s["id"] for s in candidates[:max_per_tick]]
