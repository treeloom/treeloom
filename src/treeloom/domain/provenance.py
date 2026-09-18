"""Provenance building blocks for search results.

Pure stdlib module — no I/O, no network.  Attaches commit/file:line
provenance to search result dicts and produces human-readable citation
strings and repository permalinks.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from treeloom.domain.sources import SourceRecord


# ── Internal helpers ────────────────────────────────────────────────────


def _get(obj: Any, *keys: str) -> Any:
    """Read the first matching key from either an attr-object or a dict.

    Tries each key in order, returning the first truthy value found.
    Returns ``""`` when no key resolves.
    """
    for key in keys:
        try:
            val = getattr(obj, key)
        except AttributeError:
            val = obj.get(key, "") if isinstance(obj, dict) else ""
        if val:
            return val
    return ""


# ── Public API ──────────────────────────────────────────────────────────


def repo_label(source: Any) -> str:
    """Derive a short human-readable label for a source.

    Accepts either a :class:`~treeloom.domain.sources.SourceRecord` (or any
    object with ``.url`` / ``.path`` / ``.id`` attributes) or a plain dict
    with keys ``url``, ``path``, ``id``, or ``source_id``.

    Precedence: url-basename (trailing ``.git`` stripped) → path-basename
    → ``id`` / ``source_id`` → ``""``.
    """
    url = _get(source, "url")
    if url:
        basename = os.path.basename(url.rstrip("/"))
        if basename.endswith(".git"):
            basename = basename[: -len(".git")]
        if basename:
            return basename

    path = _get(source, "path")
    if path:
        basename = os.path.basename(path.rstrip("/"))
        if basename:
            return basename

    return _get(source, "id", "source_id")


def make_citation(
    label: str,
    commit_sha: str,
    file_path: str,
    start: int,
    end: int,
) -> str:
    """Build a compact citation string for a chunk.

    Format: ``"{label}@{sha[:12]}:{file_path}:{start}-{end}"``

    Rules:
    - The ``@{sha}`` segment is omitted when *commit_sha* is falsy.
    - The ``:{start}-{end}`` segment is omitted when *both* start and end
      are 0.
    - The label prefix and colon are omitted when *label* is empty.

    Examples::

        make_citation("my_repo", "a1b2c3d4e5f6aaaa", "src/Foo.tsx", 10, 42)
        # → "my_repo@a1b2c3d4e5f6:src/Foo.tsx:10-42"

        make_citation("", "a1b2c3", "src/Foo.tsx", 0, 0)
        # → "src/Foo.tsx"
    """
    parts: list[str] = []

    # Build the "label@sha" prefix.
    prefix = label or ""
    if commit_sha:
        prefix = f"{prefix}@{commit_sha[:12]}" if prefix else f"@{commit_sha[:12]}"

    if prefix:
        parts.append(prefix)

    parts.append(file_path)

    line_segment = "" if (start == 0 and end == 0) else f"{start}-{end}"
    if line_segment:
        parts.append(line_segment)

    # Join: prefix:file_path[:start-end]
    # When parts = [prefix, file_path, line_segment] we want ":"-separated.
    return ":".join(parts)


def permalink(
    url: str,
    commit_sha: str,
    file_path: str,
    start: int,
    end: int,
) -> str | None:
    """Build a source-hosting permalink for a file range.

    Returns ``None`` unless **both** *url* and *commit_sha* are truthy.
    Strips a trailing ``.git`` from *url* and produces a GitHub/Gitea-compatible
    ``/blob/<sha>/<path>`` URL, appending ``#L{start}-L{end}`` only when start
    and end are not both 0.
    """
    if not url or not commit_sha:
        return None

    base = url.rstrip("/")
    if base.endswith(".git"):
        base = base[: -len(".git")]

    result = f"{base}/blob/{commit_sha}/{file_path}"

    if not (start == 0 and end == 0):
        result += f"#L{start}-L{end}"

    return result


def build_provenance(
    result: dict,
    records_by_id: dict,
    staleness_by_id: dict | None = None,
) -> None:
    """Attach commit/file:line provenance to a search result dict in place.

    Mutates *result* — adds ``commit_sha`` and ``citation`` to each chunk
    whose source is in *records_by_id*, and populates ``result["sources"]``
    with one entry per distinct resolvable source.

    Args:
        result: A search result dict as returned by ``/search`` (has
            ``"chunks"`` list, optional top-level ``"source_id"``).
        records_by_id: Maps source_id → SourceRecord (or duck-typed object).
        staleness_by_id: Optional map of source_id →
            ``{"is_stale": bool, "current_sha": str}``; merged into the
            corresponding ``sources[sid]`` entry when present.
    """
    # Collect all distinct source IDs we encounter (chunks + top-level).
    top_sid: str | None = result.get("source_id") or None
    seen_sids: dict[str, None] = {}  # ordered set (insertion order kept)

    if top_sid:
        seen_sids[top_sid] = None

    for chunk in result.get("chunks", []):
        sid = chunk.get("source_id") or top_sid
        if not sid:
            continue
        rec = records_by_id.get(sid)
        if rec is None:
            continue
        seen_sids[sid] = None

        # Per-chunk fields. commit_sha is NOT written onto the chunk — it's
        # emitted once per source in result["sources"] below (no point repeating
        # the same SHA ~6× per single-source search). It's still needed locally
        # to build the per-chunk citation.
        commit_sha: str = _get(rec, "commit_sha") or ""

        label = repo_label(rec)
        file_path: str = chunk.get("file_path") or ""
        start = int(chunk.get("start_line") or 0)
        end = int(chunk.get("end_line") or 0)
        chunk["citation"] = make_citation(label, commit_sha, file_path, start, end)

    # Build result["sources"] for every resolvable sid.
    sources: dict[str, dict] = {}
    for sid in seen_sids:
        rec = records_by_id.get(sid)
        if rec is None:
            continue

        commit_sha = _get(rec, "commit_sha") or ""
        rec_url: str = _get(rec, "url") or ""
        rec_path: str = _get(rec, "path") or ""
        rec_branch: str = _get(rec, "branch") or ""
        indexed_at_raw = getattr(rec, "indexed_at", None) if not isinstance(rec, dict) else rec.get("indexed_at")
        indexed_at_str = indexed_at_raw.isoformat() if indexed_at_raw is not None else ""

        entry: dict = {
            "commit_sha": commit_sha,
            "indexed_at": indexed_at_str,
            "path": rec_path,
            "url": rec_url,
            "branch": rec_branch,
        }

        # permalink_base only when we can construct a stable URL.
        if rec_url and commit_sha:
            base = rec_url.rstrip("/")
            if base.endswith(".git"):
                base = base[: -len(".git")]
            entry["permalink_base"] = f"{base}/blob/{commit_sha}"

        sources[sid] = entry

    if sources or seen_sids:
        result["sources"] = sources

    # Merge staleness information when provided.
    if staleness_by_id and "sources" in result:
        for sid, stale_info in staleness_by_id.items():
            if sid in result["sources"]:
                result["sources"][sid]["is_stale"] = stale_info.get("is_stale")
                result["sources"][sid]["current_sha"] = stale_info.get("current_sha", "")
