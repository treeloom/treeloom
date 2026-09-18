import asyncio
import logging
import os
import re

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import ValidationError

from treeloom.config import require_env
from treeloom.domain.shared import (
    ChunkHit,
    IndexAck,
    JobStatus,
    NeighborEntity,
    SearchResponse,
    SourceInfo,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "treeloom",
    instructions="""\
This server provides semantic GraphRAG search over indexed source code.

## When to use these tools (PREFER over grep / glob / Read for code questions)

ALWAYS use `search_code` FIRST whenever the user asks you to:
- understand how something works in an indexed repo
- plan changes to existing code
- trace execution flows, find handlers/publishers/consumers
- locate implementations by behavior rather than exact name
- explore an unfamiliar codebase

Why prefer this over grep:
- grep matches exact strings — useless when the symbol name is unknown
- search_code finds code by *meaning* — "where are feature flag changes propagated"
  surfaces OnFeatureFlagChangedHandler, ControlPlaneFeatureFlagChangePublisher, etc.
- results come pre-ranked with file_path/start_line/end_line — no sifting through
  hundreds of grep hits

## When NOT to use these tools

- Uncommitted local changes — RAG searches the indexed snapshot, not the working
  tree. Use Read for files just modified.
- Exact symbol/string search across a known repo — grep is faster.
- Reading a specific file by path you already know — use Read.
- Repository not yet indexed — call `list_indexed_sources` first; if missing,
  call `index_repo` before searching.

## Multi-query strategy

For complex questions, decompose into 3-5 focused queries rather than one broad
one. Pattern: list_indexed_sources → broad search → read top results → refine
with targeted queries → synthesize. Each query should focus on one concept.

## Discovering scope

Call `list_indexed_sources` first to see what's indexed and get `source_id`
values to scope subsequent searches.
""",
)
INDEXER_URL = require_env("INDEXER_URL")


# ── helpers ──────────────────────────────────────────────────────


def _auth_headers() -> dict:
    """Service-account auth for INDEX/ADMIN tools — TREELOOM_MCP_API_KEY."""
    key = os.environ.get("TREELOOM_MCP_API_KEY", "")
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def _caller_auth_headers(ctx: "Context | None") -> dict:
    """Forward the MCP caller's OWN bearer to the indexer for READ tools.

    Per-principal authorization is enforced at the indexer against
    request.state.user, so a read must carry the *caller's* credential — an API
    key or an OIDC token — not the shared service key. When no inbound bearer is
    present (stdio transport, or an anonymous caller against an open indexer) we
    forward nothing: a missing credential must never silently escalate a read to
    the service account's authority. The forwarded token is opaque here — the
    indexer validates an API key via the user store.

    Per-repo-scoped read tools (search/find/graph-explore) don't call this
    directly — they call `_read_tool_auth_headers()`, which wraps this with
    the stdio env-token fallback while still never touching the service
    account. See that function's docstring for why.
    """
    if ctx is None:
        return {}
    try:
        request = ctx.request_context.request
    except Exception:
        request = None
    if request is None:
        return {}
    auth = request.headers.get("authorization")
    return {"Authorization": auth} if auth else {}


def _service_account_enabled() -> bool:
    """Whether the service account may stand in for an uncredentialed caller.

    OFF by default. Opting in re-creates a confused deputy on any transport
    where callers are not the operator (see `_tool_auth_headers`), so it exists
    for stdio / single-user local setups where the MCP client cannot attach a
    bearer of its own.
    """
    return os.environ.get("TREELOOM_MCP_SERVICE_ACCOUNT", "").lower() == "true"


def _env_token_headers() -> dict:
    """The caller's OWN credential supplied via the environment.

    This is the stdio credential model: there is no HTTP request to read a
    bearer from, and the MCP spec directs stdio servers to "retrieve
    credentials from the environment". Distinct from TREELOOM_MCP_API_KEY,
    which is a shared service account lent to callers who have none.

    Forgiving of a user pasting the full header value (``Bearer xyz``)
    instead of the bare token — without the strip, that would double up
    into ``Authorization: Bearer Bearer xyz``. The prefix match is
    case-insensitive because RFC 7235 makes the scheme name case-insensitive,
    so ``bearer xyz`` is a header value a user can legitimately have copied.
    """
    token = os.environ.get("TREELOOM_MCP_TOKEN", "").strip()
    if token[:7].lower() == "bearer ":
        token = token[7:].strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _tool_auth_headers(ctx: "Context | None") -> dict:
    """Credential to send to the indexer for ANY tool, read or write.

    Precedence: caller bearer -> TREELOOM_MCP_TOKEN -> gated service account
    -> nothing. The caller's own HTTP bearer wins when present (shared HTTP
    transport). Otherwise TREELOOM_MCP_TOKEN — the caller's own credential
    supplied via the environment, the stdio credential model — wins. The
    service account is used only when neither is present AND the operator
    explicitly opted in.

    Why this matters: `_auth_headers()` used to be injected unconditionally on
    every write/admin tool, so an anonymous caller on the HTTP transport
    inherited the service account's authority over the indexer — index any
    path, delete any source, rebuild any graph (CWE-306 / confused deputy).
    Routing writes through the caller's credential makes MCP access exactly as
    privileged as talking to the indexer directly: no escalation path.

    Sending nothing is the safe default, not a failure: the indexer decides.
    With AUTH_ENABLED=true it answers 401; with auth off it was open anyway.
    """
    caller = _caller_auth_headers(ctx)
    if caller:
        return caller
    env_token = _env_token_headers()
    if env_token:
        return env_token
    if _service_account_enabled():
        return _auth_headers()
    return {}


def _read_tool_auth_headers(ctx: "Context | None") -> dict:
    """Credential to send to the indexer for per-repo-scoped READ tools only
    (search_code, search_code_enhanced, hydrate_chunks, explain_code,
    graph_explore, find_definition, find_callers, find_references,
    list_job_errors, list_indexed_sources, source_staleness).

    The last two are reads that the indexer filters per principal rather than
    per repo: GET /sources and its staleness view return only the caller's own
    records (created_by), with admins seeing all. They sat on
    `_tool_auth_headers` because they live among the registry tools, so the
    service-account fallback would have answered them with the whole fleet's
    sources instead of the caller's own. Filtering is filtering — whoever the
    indexer scopes the response to, the credential has to be the caller's.

    Precedence: caller bearer -> TREELOOM_MCP_TOKEN -> nothing. Deliberately
    NOT the same policy as `_tool_auth_headers`: this must never fall through
    to the gated service account, however it's called elsewhere in this
    file — do not "simplify" this to `_tool_auth_headers(ctx)`.

    Why: per-repo search authorization is enforced at the
    indexer against request.state.user, scoping each read to the calling
    principal's own grants. `TREELOOM_MCP_API_KEY` is a *shared* service
    account; if these reads could fall back to it, an operator who opts into
    the service account (`TREELOOM_MCP_SERVICE_ACCOUNT=true`, intended for
    single-user stdio setups where the MCP client has no bearer of its own)
    would silently grant every caller on that transport the service
    account's cross-repo visibility instead of their own scoped grants —
    the exact confused-deputy class `_tool_auth_headers` exists to prevent
    for writes, reopened on the read path. TREELOOM_MCP_TOKEN is safe here
    because it represents the caller's OWN credential (delivered via the
    environment instead of an HTTP header, the stdio model) — not a shared
    credential — so per-repo scoping at the indexer still applies to it
    exactly as it would to a forwarded bearer.
    """
    caller = _caller_auth_headers(ctx)
    if caller:
        return caller
    return _env_token_headers()


def _parse_neighbor(d: dict) -> NeighborEntity:
    """Parse a dict from the indexer into a NeighborEntity.

    The indexer returns graph entity dicts with ``_rel_type``, ``_rel_direction``,
    ``_target_id`` keys (Neo4j convention).  We map them to the Pydantic fields
    ``rel_type``, ``rel_direction``, ``target_id``.
    """
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


def _neighbor_to_dict(ne: NeighborEntity) -> dict:
    """Serialize a NeighborEntity back to a dict with ``_rel_type``/``_rel_direction`` keys.

    The MCP wire format uses the underscore-prefixed field names that the indexer
    produces (Neo4j convention).  Pydantic model_dump() uses the model field names
    (``rel_type``, etc.), so we remap them here.
    """
    # exclude_defaults drops the empty-string/0/None fields most neighbors
    # carry (ExternalModule import targets have only id/type/rel) — they're
    # pure token cost on the MCP wire. `id` has no default, so it's always
    # present.
    d = ne.model_dump(exclude_defaults=True)
    # remap to wire format, keeping only populated values
    for field, wire_key in (
        ("rel_type", "_rel_type"),
        ("rel_direction", "_rel_direction"),
        ("target_id", "_target_id"),
    ):
        v = d.pop(field, None)
        if v:
            d[wire_key] = v
    return d


def _parse_search_response(data: dict) -> SearchResponse:
    """Parse a /search or /graph-explore response dict into a SearchResponse."""
    # Skip rows that fail validation (e.g. missing file_path/snippet) rather
    # than letting one malformed row 500 the whole response.
    chunks = []
    for c in data.get("chunks", []):
        try:
            chunks.append(ChunkHit(**c))
        except ValidationError:
            logger.warning("Skipping malformed chunk in search response: %r", c)
    neighbors = [_parse_neighbor(n) for n in data.get("neighbors", [])]
    # Community ids are namespaced strings ("<source_id>#<n>"); use as-is.
    community_summaries = {
        str(k): v
        for k, v in (data.get("community_summaries", {}) or {}).items()
    }
    return SearchResponse(
        chunks=chunks,
        neighbors=neighbors,
        community_summaries=community_summaries,
        source_id=data.get("source_id"),
        sources=data.get("sources", {}) or {},
    )


def _search_response_to_dict(sr: SearchResponse) -> dict:
    """Convert SearchResponse to a JSON-serializable dict for MCP tool return.

    MCP tools must return plain dicts, not Pydantic models.
    """
    chunks = []
    for c in sr.chunks:
        # exclude_defaults mirrors the indexer's trim — don't re-inflate the
        # empty/False fields it dropped. file_path/snippet have no defaults,
        # so the required keys always survive. score is re-added: 0.0 equals
        # the field default but is a real ranking value.
        d = c.model_dump(exclude_defaults=True)
        d["score"] = c.score
        chunks.append(d)
    out = {
        "chunks": chunks,
        "neighbors": [_neighbor_to_dict(n) for n in sr.neighbors],
        "community_summaries": {str(k): v for k, v in sr.community_summaries.items()},
    }
    if sr.source_id:
        out["source_id"] = sr.source_id
    if sr.sources:
        out["sources"] = sr.sources
    return out


# Repository content is attacker-influenced for any repo an operator indexes,
# and this renderer is the DEFAULT payload for search_code. A fixed ``` fence
# is closable by a snippet that contains one, which drops the remainder into
# prose position in the agent's context. CommonMark closes a fence only on a
# run at least as long as the opener, so opening wider than anything inside
# makes the block unclosable by its own content. Content is unchanged.
_BACKTICK_RUN = re.compile(r"`+")

# Characters that would break a one-line heading into extra prose lines.
_MD_LINE_BREAKERS = re.compile(r"[\r\n\u2028\u2029]+")


def _fence_for(text: str) -> str:
    """The shortest fence that *text* cannot close (minimum three)."""
    longest = max((len(m.group()) for m in _BACKTICK_RUN.finditer(text or "")), default=0)
    return "`" * max(3, longest + 1)


def _md_one_line(value) -> str:
    """Keep a repository-derived value on the single line it belongs on."""
    return _MD_LINE_BREAKERS.sub(" ", str(value))


def _search_response_to_markdown(d: dict) -> str:
    """Render a search-response dict as markdown text.

    Tokenizes 16-22% cheaper than the JSON dict FastMCP would emit (measured
    with tiktoken cl100k/o200k on live featbit responses) — JSON escapes every
    newline/quote in code snippets and repeats key names per chunk. Same
    information, fenced code blocks under a `path:lines (score)` heading.
    """
    parts = []
    if d.get("source_id"):
        parts.append(f"source: {d['source_id']}")
    for c in d.get("chunks", []):
        heading = (
            f"### {_md_one_line(c['file_path'])}:{c.get('start_line', '?')}-"
            f"{c.get('end_line', '?')} (score {c.get('score')})"
        )
        if c.get("citation"):
            heading += f" — {_md_one_line(c['citation'])}"
        parts.append(heading)
        # facet mode: no code body — render the chunk header (the
        # enclosing class / defined symbols) as a one-line "why", and the agent
        # fetches the range via read_file. Fall through to a fence only when a
        # snippet body is present (full mode, or a summary_tail body).
        snippet = c.get("snippet")
        if snippet is None:
            if c.get("header"):
                # facet mode emits this with no fence at all, so it must not
                # be able to become more than the one line it claims to be.
                parts.append(_md_one_line(c["header"]))
            # Surface the hydrate handle explicitly so the agent can pass it
            # verbatim to hydrate_chunks.
            if c.get("hit_id"):
                parts.append(f"hit_id: {c['hit_id']}")
        else:
            fence = _fence_for(snippet)
            parts.append(f"{fence}{c.get('language', '')}\n{snippet}\n{fence}")
    if d.get("neighbors"):
        lines = ["related:"]
        for n in d["neighbors"]:
            loc = f" {n['file_path']}:{n.get('start_line', '')}" if n.get("file_path") else ""
            lines.append(
                f"- {n.get('type', '')} {n.get('name') or n['id']}{loc}"
                f" ({n.get('_rel_type', '')} {n.get('_rel_direction', '')})"
            )
        parts.append("\n".join(lines))
    if d.get("community_summaries"):
        parts.append(
            "communities:\n"
            + "\n".join(f"- {v}" for v in d["community_summaries"].values())
        )
    if d.get("sources"):
        lines = ["provenance:"]
        for sid, meta in d["sources"].items():
            sha = (meta.get("commit_sha") or "")[:12]
            if not sha:
                continue
            line = f"- {sid}: {sha}"
            if meta.get("is_stale"):
                line += " STALE"
            if meta.get("permalink_base"):
                line += f"  {meta['permalink_base']}"
            lines.append(line)
        if len(lines) > 1:
            parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ── indexing tools (async kickoff via indexer service) ───────────


@mcp.tool(
    description="Index a single code file. Parses with tree-sitter, chunks with chonkie, "
    "embeds via TEI, and stores vectors in Milvus + entities in Neo4j. Returns immediately "
    "with a job_id; poll get_index_job to track progress. Set force=true to re-index even "
    "if this file is already indexed and unchanged (otherwise the call no-ops to the prior job)."
)
async def index_file(file_path: str, force: bool = False, ctx: Context | None = None) -> dict:
    """Proxies to POST /index-file. Returns plain dict for MCP serialization."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/index-file",
                json={"file_path": file_path, "force": force},
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "job_id": data["job_id"],
                "status": data["status"],
                "source_id": data.get("source_id", ""),
                "status_url": f"{INDEXER_URL}/jobs/{data['job_id']}",
            }
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Index all code files in a directory (recursive). Returns immediately with "
    "a job_id; poll get_index_job to track progress. Set force=true to re-index even if the "
    "directory is already indexed and unchanged (otherwise the call no-ops to the prior job)."
)
async def index_directory(directory: str, pattern: str = "**/*", force: bool = False, ctx: Context | None = None) -> dict:
    """Proxies to POST /index-directory."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/index-directory",
                json={"directory": directory, "pattern": pattern, "force": force},
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "job_id": data["job_id"],
                "status": data["status"],
                "source_id": data.get("source_id", ""),
                "status_url": f"{INDEXER_URL}/jobs/{data['job_id']}",
            }
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Index a git repository. Accepts a local path (with ~/ expansion) or a remote "
    "git URL. Auto-detects current branch for local repos. Returns immediately with a job_id; "
    "poll get_index_job to track progress. Re-indexing the same source upserts (no duplicates). "
    "By default an already-indexed, unchanged source no-ops to its prior job — set force=true "
    "to force a full re-index."
)
async def index_repo(
    path: str | None = None,
    url: str | None = None,
    branch: str | None = None,
    force: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Proxies to POST /index-repo."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/index-repo",
                json={"path": path, "url": url, "branch": branch, "force": force},
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "job_id": data["job_id"],
                "status": data["status"],
                "source_id": data.get("source_id", ""),
                "status_url": f"{INDEXER_URL}/jobs/{data['job_id']}",
            }
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Build (or rebuild) the Neo4j code graph for an already-indexed source — "
    "pass 2 of two-pass indexing. Use this when chunks/vectors exist but graph-backed tools "
    "(find_definition, find_callers, find_references, graph_explore) return empty, or after a "
    "chunks-only index (skip_graph=true). Drops the source's existing graph entities and "
    "re-extracts; safe to re-run. Returns immediately with a job_id; poll get_index_job. "
    "Get source_id from list_indexed_sources."
)
async def index_graph(source_id: str, ctx: Context | None = None) -> dict:
    """Proxies to POST /index-graph."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/index-graph",
                json={"source_id": source_id},
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "job_id": data["job_id"],
                "status": data["status"],
                "source_id": data.get("source_id", source_id),
                "status_url": f"{INDEXER_URL}/jobs/{data['job_id']}",
            }
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Bulk-build the Neo4j code graph for indexed sources. Lists sources, then "
    "enqueues a graph job (POST /index-graph) for each. Use this once to repair a bulk import "
    "where chunks were indexed but the graph was never built (so find_definition/graph tools "
    "are globally empty). Graph extraction is idempotent (drops + rebuilds), so rebuilding a "
    "healthy source is harmless. Defaults to rebuilding ALL sources — the graph_indexed flag is "
    "NOT a reliable signal for the empty-graph bug (jobs marked themselves indexed while writing "
    "nothing), so only_missing=true (which filters to graph_indexed=false) will skip the very "
    "sources that need repair. Returns one entry per source with its job_id (or error). Jobs run "
    "through the shared queue — this kicks them off and returns; poll "
    "list_index_jobs(status='running') to track."
)
async def rebuild_all_graphs(only_missing: bool = False, ctx: Context | None = None) -> dict:
    """List sources and enqueue a graph job for each (all sources by default)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{INDEXER_URL}/sources",
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            sources = resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}

    results: list[dict] = []
    enqueued = 0
    skipped = 0
    failed = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for src in sources:
            source_id = src.get("id") or src.get("source_id") or ""
            if not source_id:
                continue
            # Only honored when only_missing=true. `graph_indexed` may be absent
            # on legacy/graph-store-fallback records — treat missing as "needs a
            # graph" so a repair pass still covers them.
            if only_missing and src.get("graph_indexed", False):
                skipped += 1
                results.append({"source_id": source_id, "status": "skipped",
                                "reason": "graph_indexed already true"})
                continue
            try:
                r = await client.post(
                    f"{INDEXER_URL}/index-graph",
                    json={"source_id": source_id},
                    headers=_tool_auth_headers(ctx),
                )
                r.raise_for_status()
                data = r.json()
                enqueued += 1
                results.append({"source_id": source_id, "status": data.get("status", "queued"),
                                "job_id": data.get("job_id")})
            except httpx.HTTPError as exc:
                failed += 1
                results.append({"source_id": source_id, "status": "error",
                                "error": str(exc)})

    return {
        "total_sources": len(sources),
        "enqueued": enqueued,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


@mcp.tool(description="Get the current state of an index job (queued/running/done/failed).")
async def get_index_job(job_id: str, ctx: Context | None = None) -> dict:
    """Proxies to GET /jobs/{job_id}."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{INDEXER_URL}/jobs/{job_id}",
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Block until an index job reaches done/failed/cancelled, or until timeout_s "
    "elapses. Polls server-side. Use this when you want a synchronous index call."
)
async def wait_for_index_job(job_id: str, timeout_s: int = 60, ctx: Context | None = None) -> dict:
    """Poll GET /jobs/{job_id} until terminal state or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                resp = await client.get(
                    f"{INDEXER_URL}/jobs/{job_id}",
                    headers=_tool_auth_headers(ctx),
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") in ("done", "failed", "cancelled"):
                    return data
                if asyncio.get_event_loop().time() >= deadline:
                    return data
                await asyncio.sleep(1.0)
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


# ── source registry tools ────────────────────────────────────────


@mcp.tool(description="List all indexed sources (repos, directories, files) with their stats.")
async def list_indexed_sources(ctx: Context | None = None) -> list[dict]:
    """Proxies to GET /sources."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{INDEXER_URL}/sources",
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        return [{"error": f"Indexer unreachable: {exc}"}]


@mcp.tool(
    description="Remove an indexed source. Drops its Milvus chunks and Neo4j entities "
    "(but preserves entities still referenced by other sources)."
)
async def remove_indexed_source(source_id: str, ctx: Context | None = None) -> dict:
    """Proxies to DELETE /sources/{source_id}."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{INDEXER_URL}/sources/{source_id}",
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="List indexing jobs across all sources. Each entry includes "
    "status (queued/running/done/failed), kind (file/directory/repo/graph/incremental), "
    "processed_files / total_files, errors, current_file, and timing. Useful for "
    "answering 'is treeloom busy?' or 'what happened to my last index job?'."
)
async def list_index_jobs(status: str | None = None, ctx: Context | None = None) -> list[dict]:
    """Proxies to GET /jobs. Optional status filter applied client-side."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{INDEXER_URL}/jobs",
                headers=_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            jobs = resp.json()
    except httpx.HTTPError as exc:
        return [{"error": f"Indexer unreachable: {exc}"}]
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    return jobs


@mcp.tool(
    description="List the per-file errors recorded during a specific "
    "indexing job. Each entry includes file_path, error_kind "
    "('timeout' or 'exception'), error_message, elapsed_s, and "
    "occurred_at. Useful for 'which files failed?' and 'why?' without "
    "grepping process logs."
)
async def list_job_errors(
    job_id: str, limit: int = 1000, offset: int = 0, ctx: Context | None = None
) -> dict:
    """Proxies to GET /jobs/{job_id}/errors."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{INDEXER_URL}/jobs/{job_id}/errors",
                params={"limit": limit, "offset": offset},
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Check whether an indexed source is out of date relative to "
    "its current upstream HEAD. For local-path sources, resolves via "
    "`git rev-parse`; for URL sources via `git ls-remote`. Returns "
    "`{indexed_sha, current_sha, is_stale}` — `is_stale` is null for "
    "non-git inputs."
)
async def source_staleness(source_id: str, ctx: Context | None = None) -> dict:
    """Proxies to GET /sources/{source_id}/staleness."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{INDEXER_URL}/sources/{source_id}/staleness",
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


# ── search tools (with scope filters) ────────────────────────────


@mcp.tool(
    description="PRIMARY code-search tool — use FIRST for any question about how indexed code "
    "works, to plan a change, trace a flow, or find an implementation by behavior. Prefer this "
    "over grep/Read for indexed repos: grep needs exact strings, this finds code by meaning. "
    "Embeds the query, vector-searches Milvus, expands via Neo4j graph neighbors (callers/"
    "callees/imports/inheritance) and community summaries, reranks, returns chunks with "
    "file_path, start_line, end_line, snippet, score. Each chunk carries a citation field "
    "(e.g. 'repo@sha12:path') and the response includes a top-level sources map with "
    "commit_sha and permalink_base per source for building file-level permalinks. "
    "top_k defaults to 5 (the benchmarked sweet spot) — raise it only when a first search "
    "shows the answer spans many files. Optional filter: language (e.g. "
    "'python'). SCOPE IS REQUIRED: pass source_id (from list_indexed_sources) or "
    "path_prefix (e.g. '/repo/src/') to scope to one repo, OR cross_repo=true to search "
    "the whole multi-repo index — exactly one, not both, or the search is rejected. "
    "Set check_staleness=true to add an is_stale flag per source (does live git HEAD "
    "resolution — slightly slower). "
    "response_mode (opt-in, default 'full'): 'facet' returns ranked metadata + a one-line "
    "header per hit with NO code body (cheaper — then fetch ranges with read_file); "
    "'summary_tail' keeps the top-2 snippets and replaces lower-ranked bodies with their "
    "indexed one-line summary. "
    "Returns compact markdown by default (~20% fewer tokens); pass "
    "response_format='json' to get the structured dict instead (for programmatic callers)."
)
async def search_code(
    query: str,
    top_k: int = 5,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    cross_repo: bool = False,
    use_hybrid: bool | None = None,
    use_hyde: bool | None = None,
    use_summary_vector: bool | None = None,
    use_graph_scoring: bool | None = None,
    rerank_pool: int | None = None,
    check_staleness: bool = False,
    response_format: str = "markdown",
    response_mode: str = "full",
    include_community_summaries: bool = True,
    adaptive_topk: bool | None = None,
    adaptive_topk_gap: float | None = None,
    strip_imports: bool | None = None,
    query_class_payload: bool | None = None,
    ctx: Context | None = None,
) -> dict | str:
    """Proxy to indexer POST /search. Returns markdown text by default, or dict on request."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "language": language,
                    "path_prefix": path_prefix,
                    "source_id": source_id,
                    "cross_repo": cross_repo,
                    "use_hybrid": use_hybrid,
                    "use_hyde": use_hyde,
                    "use_summary_vector": use_summary_vector,
                    "use_graph_scoring": use_graph_scoring,
                    "rerank_pool": rerank_pool,
                    "check_staleness": check_staleness,
                    "response_mode": response_mode,
                    "include_community_summaries": include_community_summaries,
                    "adaptive_topk": adaptive_topk,
                    "adaptive_topk_gap": adaptive_topk_gap,
                    "strip_imports": strip_imports,
                    "query_class_payload": query_class_payload,
                },
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            sr = _parse_search_response(resp.json())
            out = _search_response_to_dict(sr)
            if response_format == "markdown":
                return _search_response_to_markdown(out)
            return out
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Fetch the full code bodies for facet search hits. After a search_code call "
    "with response_mode='facet' (which returns ranked metadata + headers but NO code), pass "
    "the `hit_id` values of the results you care about here — in ONE batched call — to get "
    "their exact code bodies. Cheaper and more precise than read_file: it returns exactly the "
    "indexed chunk(s), reconstructing merged ranges. Returns compact markdown by default; pass "
    "response_format='json' for the structured dict."
)
async def hydrate_chunks(
    ids: list[str],
    source_id: str | None = None,
    response_format: str = "markdown",
    ctx: Context | None = None,
) -> dict | str:
    """Proxy to indexer POST /hydrate-chunks."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/hydrate-chunks",
                json={"ids": ids, "source_id": source_id},
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            sr = _parse_search_response(resp.json())
            out = _search_response_to_dict(sr)
            if response_format == "markdown":
                return _search_response_to_markdown(out)
            return out
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Use when `search_code` results suggest the answer spans multiple structurally-"
    "related files (e.g. handler/publisher/consumer triads, or a class and its subclasses) and "
    "you want to pull in chunks from those neighbor files in one shot. Same inputs and filters "
    "as `search_code`; does a second-pass Milvus search filtered to neighbor file paths before "
    "reranking. Results carry per-chunk citation fields and a top-level sources map (commit_sha "
    "+ permalink_base). Set check_staleness=true to add an is_stale flag per source (does live "
    "git HEAD resolution — slightly slower). response_mode (opt-in, default 'full'): 'facet' "
    "returns ranked metadata + a one-line header per hit with NO code body (then fetch ranges "
    "with read_file); 'summary_tail' keeps the top-2 snippets and replaces lower-ranked bodies "
    "with their indexed summary. Returns compact markdown by default (~20% fewer "
    "tokens); pass response_format='json' to get the structured dict instead (for programmatic "
    "callers)."
)
async def search_code_enhanced(
    query: str,
    top_k: int = 5,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    cross_repo: bool = False,
    use_hybrid: bool | None = None,
    use_hyde: bool | None = None,
    use_summary_vector: bool | None = None,
    use_graph_scoring: bool | None = None,
    rerank_pool: int | None = None,
    check_staleness: bool = False,
    response_format: str = "markdown",
    response_mode: str = "full",
    include_community_summaries: bool = True,
    adaptive_topk: bool | None = None,
    adaptive_topk_gap: float | None = None,
    strip_imports: bool | None = None,
    query_class_payload: bool | None = None,
    ctx: Context | None = None,
) -> dict | str:
    """Proxy to indexer POST /search (enhanced)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "language": language,
                    "path_prefix": path_prefix,
                    "source_id": source_id,
                    "cross_repo": cross_repo,
                    "use_hybrid": use_hybrid,
                    "use_hyde": use_hyde,
                    "use_summary_vector": use_summary_vector,
                    "use_graph_scoring": use_graph_scoring,
                    "rerank_pool": rerank_pool,
                    "check_staleness": check_staleness,
                    "response_mode": response_mode,
                    "include_community_summaries": include_community_summaries,
                    "adaptive_topk": adaptive_topk,
                    "adaptive_topk_gap": adaptive_topk_gap,
                    "strip_imports": strip_imports,
                    "query_class_payload": query_class_payload,
                },
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            sr = _parse_search_response(resp.json())
            out = _search_response_to_dict(sr)
            if response_format == "markdown":
                return _search_response_to_markdown(out)
            return out
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Use when the user asks you to explain a specific concept or section of code and "
    "wants tight, focused context (e.g. 'explain how X works' rather than 'find everything about "
    "X'). Same as `search_code` but with top_k=5 and an optional file_path filter applied "
    "server-side — useful for narrowing to a single file you already know is relevant."
)
async def explain_code(
    query: str,
    file_path: str | None = None,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    cross_repo: bool = False,
    use_hybrid: bool | None = None,
    use_hyde: bool | None = None,
    use_summary_vector: bool | None = None,
    use_graph_scoring: bool | None = None,
    rerank_pool: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Proxy to indexer POST /search with top_k=5, optional file_path narrowing.

    file_path is passed as the server-side path_prefix so Milvus searches only
    that file's chunks — previously it was filtered client-side AFTER retrieving
    the global top 5, which usually returned nothing when the file wasn't
    already in the top 5. A file scope satisfies the repo-scope requirement,
    so cross_repo is dropped when file_path is set. The exact-match post-filter
    stays as a cheap guard against prefix cousins (/a/b.py vs /a/b.py.bak).
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/search",
                json={
                    "query": query,
                    "top_k": 5,
                    "language": language,
                    "path_prefix": file_path or path_prefix,
                    "source_id": source_id,
                    "cross_repo": False if file_path else cross_repo,
                    "use_hybrid": use_hybrid,
                    "use_hyde": use_hyde,
                    "use_summary_vector": use_summary_vector,
                    "use_graph_scoring": use_graph_scoring,
                    "rerank_pool": rerank_pool,
                },
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            raw = resp.json()
            if file_path:
                raw["chunks"] = [
                    c for c in raw.get("chunks", [])
                    if c.get("file_path") == file_path
                ]
            sr = _parse_search_response(raw)
            return _search_response_to_dict(sr)
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Use when the user wants a structural/architectural view rather than a code-snippet "
    "view — 'what depends on X', 'what does X talk to', 'show me the call graph around Y'. Finds "
    "entities related to the query and traverses their connections (callers, callees, imports, "
    "inheritance) up to `depth` hops. For 'find the code that does X', use `search_code` instead."
)
async def graph_explore(
    query: str,
    depth: int = 1,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    use_hybrid: bool | None = None,
    use_hyde: bool | None = None,
    use_summary_vector: bool | None = None,
    use_graph_scoring: bool | None = None,
    rerank_pool: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Proxy to indexer POST /graph-explore."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{INDEXER_URL}/graph-explore",
                json={
                    "query": query,
                    "depth": depth,
                    "language": language,
                    "path_prefix": path_prefix,
                    "source_id": source_id,
                    "use_hybrid": use_hybrid,
                    "use_hyde": use_hyde,
                    "use_summary_vector": use_summary_vector,
                    "use_graph_scoring": use_graph_scoring,
                    "rerank_pool": rerank_pool,
                },
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            sr = _parse_search_response(resp.json())
            return _search_response_to_dict(sr)
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


# ── symbol-level graph tools ─────────────────────────────────────


@mcp.tool(
    description="Find where a symbol is defined. Looks up Class/Function entities by name. "
    "Returns all matches across files (let the caller disambiguate). Optional `kind` filter "
    "('Class' or 'Function') and `source_id` to scope to one indexed source."
)
async def find_definition(
    name: str,
    kind: str | None = None,
    source_id: str | None = None,
    ctx: Context | None = None,
) -> list[dict]:
    """Proxy to indexer GET /find-definition."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params: dict[str, str] = {"name": name}
            if kind:
                params["kind"] = kind
            if source_id:
                params["source_id"] = source_id
            resp = await client.get(
                f"{INDEXER_URL}/find-definition",
                params=params,
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            entities = resp.json()
            return [_neighbor_to_dict(_parse_neighbor(e)) for e in entities]
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Find callers of a function/method. Accepts either a name (auto-resolves; "
    "returns callers across all matches with target_id set) or a full entity id."
)
async def find_callers(
    name_or_id: str, source_id: str | None = None, ctx: Context | None = None
) -> list[dict]:
    """Proxy to indexer GET /find-callers."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params: dict[str, str] = {"name_or_id": name_or_id}
            if source_id:
                params["source_id"] = source_id
            resp = await client.get(
                f"{INDEXER_URL}/find-callers",
                params=params,
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            callers = resp.json()
            return [_neighbor_to_dict(_parse_neighbor(c)) for c in callers]
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}


@mcp.tool(
    description="Find all references (any incoming relationship — CALLS, INHERITS, IMPORTS, "
    "DEFINES) to a symbol. Accepts a name or a full entity id."
)
async def find_references(
    name_or_id: str, source_id: str | None = None, ctx: Context | None = None
) -> list[dict]:
    """Proxy to indexer GET /find-references."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params: dict[str, str] = {"name_or_id": name_or_id}
            if source_id:
                params["source_id"] = source_id
            resp = await client.get(
                f"{INDEXER_URL}/find-references",
                params=params,
                headers=_read_tool_auth_headers(ctx),
            )
            resp.raise_for_status()
            refs = resp.json()
            return [_neighbor_to_dict(_parse_neighbor(r)) for r in refs]
    except httpx.HTTPError as exc:
        return {"error": f"Indexer unreachable: {exc}"}
