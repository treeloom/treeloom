"""Shared domain types and utility functions for treeloom.

Contains Pydantic models and deterministic source ID generation.
"""
import fnmatch
import hashlib
import os
from typing import Optional
from pydantic import BaseModel, Field


# ── Source helpers ──────────────────────────────────────────────────────


def make_source_id(path: str | None = None, url: str | None = None, branch: str | None = None) -> str:
    """Deterministic source id from canonical path or url + branch.

    Re-indexing the same source produces the same id, enabling dedup.
    """
    parts: list[str] = []
    if url:
        parts.append(url.strip().rstrip("/"))
    elif path:
        try:
            parts.append(os.path.realpath(os.path.expanduser(path)))
        except OSError:
            parts.append(os.path.expanduser(path))
    if branch:
        parts.append(branch.strip())
    # Escape before joining. A bare "|".join is ambiguous: a url of
    # "https://h/r|main" with no branch produced the same key — and therefore
    # the same source_id — as url "https://h/r" on branch "main". source_id is
    # what the dedup short-circuit, the ACL grants and the vector-store filters
    # are all keyed by, so a collision is not cosmetic: a crafted url could
    # take over an existing source's identity (CWE-20).
    #
    # Escaping rather than changing the separator is deliberate. Every input
    # that contains neither "|" nor "\\" — which is every real path, url and
    # branch — hashes to exactly the id it did before, so no existing source
    # needs reindexing and no grant keyed to a source_id breaks. Only the
    # inputs that were ambiguous change, and those had no well-defined id to
    # preserve.
    key = "|".join(_escape_key_part(p) for p in parts) or "anonymous"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _escape_key_part(part: str) -> str:
    """Make the "|" separator unambiguous by escaping it within each part."""
    return part.replace("\\", "\\\\").replace("|", "\\|")


DEFAULT_BRANCH_PATTERNS = ["main", "master"]


def branch_matches(branch: Optional[str], patterns: Optional[list[str]]) -> bool:
    """Check if a branch name matches any of the configured glob patterns.

    Args:
        branch: Branch name. None = assume default branch.
        patterns: List of fnmatch patterns. Empty or None = match everything
                  (opt-in to restrict — no patterns means no filtering).

    Returns:
        True if the branch should be indexed.
    """
    if not patterns:
        return True
    if branch is None:
        branch = "main"
    return any(fnmatch.fnmatch(branch, p) for p in patterns)


# ── Pydantic models ─────────────────────────────────────────────────────


class ChunkHit(BaseModel):
    file_path: str
    start_line: int = 0
    end_line: int = 0
    language: str = ""
    score: float = 0.0
    # snippet is the code body in full mode and the indexed LLM summary in
    # summary_tail mode. It is defaulted (not required) so facet mode can omit
    # the body entirely without the MCP layer's ChunkHit(**c) 500'ing — file_path
    # stays the only required key on the wire contract.
    snippet: str = ""
    source_id: str | None = None
    graph_enhanced: bool = False
    # facet mode surfaces the _chunk_header text as its own field (enclosing
    # class + defined symbols + signature) instead of prepending it to a snippet.
    header: str = ""
    # facet mode only: a stable, location-derived handle
    # ("file_path:start-end") the agent passes to hydrate_chunks to fetch the
    # exact dropped code body. Defaulted "" so full/summary_tail stay
    # byte-identical (exclude_defaults drops it).
    hit_id: str = ""
    # Provenance fields — empty by default so model_dump(exclude_defaults=True)
    # drops them when provenance has not been attached.
    commit_sha: str = ""
    citation: str = ""


class NeighborEntity(BaseModel):
    id: str
    type: str = ""
    name: str = ""
    signature: str = ""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    language: str = ""
    rel_type: str = ""
    rel_direction: str = ""
    target_id: str | None = None


class SearchResponse(BaseModel):
    chunks: list[ChunkHit] = Field(default_factory=list)
    neighbors: list[NeighborEntity] = Field(default_factory=list)
    # Community ids are namespaced strings ("<source_id>#<n>") since
    # detection runs per-source — not integers.
    community_summaries: dict[str, str] = Field(default_factory=dict)
    # Set when every chunk shares one source (always true for repo-scoped
    # searches); per-chunk source_id is omitted in that case.
    source_id: str | None = None
    # Provenance map — keyed by source_id, absent when provenance has not
    # been attached.  Empty default drops off the wire via
    # model_dump(exclude_defaults=True).
    sources: dict = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    status: str
    source_id: str = ""
    source: str = ""
    kind: str = ""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    current_file: str = ""
    start_time: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    message: str = ""


class IndexAck(BaseModel):
    job_id: str
    status: str
    source_id: str
    status_url: str


class SourceInfo(BaseModel):
    id: str
    path: str = ""
    url: str = ""
    branch: str = ""
    branch_patterns: list[str] = Field(default_factory=lambda: DEFAULT_BRANCH_PATTERNS.copy())
    indexed_at: int = 0
    file_count: int = 0
    chunk_count: int = 0


# ── Helper functions ────────────────────────────────────────────────────


def neighbor_from_dict(d: dict) -> NeighborEntity:
    return NeighborEntity(
        id=d.get("id", ""),
        type=d.get("type", ""),
        name=d.get("name", ""),
        signature=d.get("signature", "") or "",
        file_path=d.get("file_path", "") or "",
        start_line=int(d.get("start_line") or 0),
        end_line=int(d.get("end_line") or 0),
        language=d.get("language", "") or "",
        rel_type=d.get("_rel_type", "") or "",
        rel_direction=d.get("_rel_direction", "") or "",
        target_id=d.get("_target_id"),
    )


def make_hit_id(file_path: str, start_line: int, end_line: int) -> str:
    """Stable facet handle: "file_path:start-end".

    Location-derived (not the Milvus PK) so it survives re-index and resolves
    merged hits by line-range overlap. Paths may contain ':' (e.g. Windows
    drives) — `parse_hit_id` splits from the right, so round-tripping is safe.
    """
    return f"{file_path}:{int(start_line)}-{int(end_line)}"


def parse_hit_id(hit_id: str) -> tuple[str, int, int] | None:
    """Inverse of `make_hit_id`. Returns (file_path, start, end) or None if the
    token is malformed (the hydrate endpoint skips bad ids rather than 500)."""
    try:
        path, span = hit_id.rsplit(":", 1)
        lo, hi = span.split("-", 1)
        return path, int(lo), int(hi)
    except (ValueError, AttributeError):
        return None


def chunk_hit_from_milvus(r: dict, graph_enhanced: bool = False) -> ChunkHit:
    entity = r.get("entity", {}) or {}
    # Prefer the graph-rescored final_score (the value the result is ranked
    # by when graph scoring is on), then the reranker's relevance_score,
    # then the raw vector distance.
    if "final_score" in r:
        score = r.get("final_score")
    else:
        score = r.get("relevance_score", r.get("distance", 0))
    return ChunkHit(
        file_path=entity.get("file_path", "") or "",
        start_line=int(entity.get("start_line") or 0),
        end_line=int(entity.get("end_line") or 0),
        language=entity.get("language", "") or "",
        score=float(score or 0),
        snippet=entity.get("chunk_text", "") or "",
        source_id=entity.get("source_id"),
        graph_enhanced=graph_enhanced or bool(entity.get("_graph_enhanced")),
    )
