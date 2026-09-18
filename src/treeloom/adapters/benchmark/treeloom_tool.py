"""Benchmark adapter: the treeloom-arm `search_code` tool.

Calls the indexer `POST /search`. In the default (non-lean) mode it renders the
full, provenance-enriched server response with the EXACT production markdown
serializer (`_search_response_to_markdown` from the MCP server) — the same
string a real MCP client's agent sees — so token accounting matches production
instead of understating it. The server already attaches the `sources`
provenance map + per-chunk `citation` before we receive it; we no longer strip
those, so the markdown + provenance token cost is now measured honestly
. The lean mode is the deliberate tripwire arm: it strips the graph
payload and caps snippets, returning a small dict instead.

Records each returned chunk's file_path into `search_log` (the runner uses it
as the treeloom arm's 'retrieved files', in returned order, for recall/MRR) —
this is driven off the raw server `chunks`, independent of the rendered output.
"""
from __future__ import annotations

import os

import httpx

from treeloom.adapters.benchmark.tool_base import Tool

# Fields the lean tripwire arm keeps per chunk (it deliberately strips the
# graph payload + provenance; the non-lean arm renders the full server response).
_CHUNK_FIELDS = ("file_path", "snippet", "start_line", "end_line", "language", "score")


# Lean mode caps each snippet to keep the agent-facing payload small.
_LEAN_SNIPPET_CHARS = 800


def _file_paths(data: dict) -> list[str]:
    """File paths of the returned chunks, in ranked order, skipping any chunk
    with no file_path — this is what feeds `search_log` (recall/MRR), driven off
    the raw server response regardless of how the result is rendered."""
    return [c["file_path"] for c in (data.get("chunks") or []) if c.get("file_path")]


def _lean_shape(data: dict) -> dict:
    # Just the ranked code chunks — drop neighbors + community summaries
    # (graph-rescoring signals, not things an answering agent needs) and cap
    # snippet length. The deliberate lean tripwire arm.
    chunks = []
    for c in data.get("chunks", []) or []:
        if not c.get("file_path"):
            continue
        ch = {k: c.get(k) for k in _CHUNK_FIELDS}
        if ch.get("snippet"):
            ch["snippet"] = ch["snippet"][:_LEAN_SNIPPET_CHARS]
        chunks.append(ch)
    return {"chunks": chunks}


def search_code_tool(
    search_url: str, search_log: list[str], top_k_override: int | None = None,
    lean: bool = False, path_prefix: str | None = None,
    use_graph_scoring: bool = True, response_mode: str = "full",
    trims: dict | None = None,
) -> Tool:
    base = search_url.rstrip("/")

    async def run(query: str = "", top_k: int = 10, **_):
        if not query:
            return {"error": "query is required"}
        # A sweep fixes top_k regardless of what the agent requests, so payload
        # size is the controlled variable.
        k = int(top_k_override) if top_k_override is not None else int(top_k)
        body: dict = {"query": str(query), "top_k": k}
        # Opt-in response-shaping mode: facet / summary_tail arms.
        if response_mode and response_mode != "full":
            body["response_mode"] = response_mode
        # Scope search to the target repo (fair vs the repo-scoped grep arm);
        # otherwise explicitly opt into the whole multi-repo index. The indexer
        # /search requires exactly one of path_prefix/source_id XOR cross_repo.
        if path_prefix:
            body["path_prefix"] = path_prefix
        else:
            body["cross_repo"] = True
        # Graph RESCORING is sent explicitly either way so the benchmark arm's
        # intent is pinned regardless of the server default (ON by default since
        #, but a Cohere-class deployment may set it off). The
        # neighbor/community PAYLOAD is always returned by the server independent
        # of this flag — only the post-rerank rescore is gated.
        body["use_graph_scoring"] = bool(use_graph_scoring)
        # Rejected payload trims (treeloom-trimmed arm): per-request flags
        # merged verbatim, e.g. {include_community_summaries:False,
        # adaptive_topk:True, strip_imports:True}. Empty/None = baseline defaults.
        if trims:
            body.update(trims)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{base}/search", json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            return {"error": f"treeloom search failed: {e}"}
        # Record retrieved file_paths for recall/MRR from the raw server chunks,
        # independent of how we render the result below.
        for fp in _file_paths(data):
            ap = os.path.realpath(fp)
            if ap not in search_log:
                search_log.append(ap)
        if lean:
            return _lean_shape(data)
        # Non-lean: return the EXACT production markdown rendering of the full
        # provenance-enriched server response (chunks/neighbors/community_summaries/
        # sources/source_id are already present in `data`). Import lazily so the
        # benchmark process doesn't trigger mcp_server's import-time env reads /
        # FastMCP init.
        from treeloom.application.mcp_server import _search_response_to_markdown

        return _search_response_to_markdown(data)

    return Tool(
        name="search_code",
        description=(
            "Semantic + graph code search over the indexed repo. args: query "
            "(natural language), top_k (default 10). Returns ranked code chunks "
            "with file_path, line range, and snippet."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    )


def hydrate_chunks_tool(
    search_url: str, search_log: list[str], path_prefix: str | None = None,
) -> Tool:
    """Batch-rehydrate facet hit_ids to full code bodies.

    The companion to response_mode=facet: the agent passes the `hit_id`s it cares
    about (from the facet search markdown) and gets the exact dropped bodies in
    ONE call — bounding compensatory fetch at a fixed +1 turn vs N read_files.
    """
    base = search_url.rstrip("/")

    async def run(ids=None, **_):
        # Tolerate a JSON-string or comma list from the ReAct text protocol.
        if isinstance(ids, str):
            import json as _json
            try:
                ids = _json.loads(ids)
            except ValueError:
                ids = [s.strip() for s in ids.split(",") if s.strip()]
        if not ids or not isinstance(ids, list):
            return {"error": "ids (list of hit_id strings) is required"}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{base}/hydrate-chunks", json={"ids": [str(i) for i in ids]})
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            return {"error": f"treeloom hydrate failed: {e}"}
        for fp in _file_paths(data):
            ap = os.path.realpath(fp)
            if ap not in search_log:
                search_log.append(ap)
        from treeloom.application.mcp_server import _search_response_to_markdown
        return _search_response_to_markdown(data)

    return Tool(
        name="hydrate_chunks",
        description=(
            "Fetch the full code body for facet search hits. After a search_code "
            "result (facet mode: metadata + headers, NO code), pass the hit_id "
            "values you want — as a list, in ONE call — to get their exact code "
            "bodies. Cheaper and more precise than read_file."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ids"],
        },
    )


def treeloom_arm_tools(
    search_url: str, search_log: list[str], top_k_override: int | None = None,
    lean: bool = False, path_prefix: str | None = None,
    use_graph_scoring: bool = True, repo_path: str | None = None,
    response_mode: str = "full", trims: dict | None = None,
    with_hydrate: bool = False,
) -> list[Tool]:
    tools = [search_code_tool(search_url, search_log, top_k_override, lean,
                             path_prefix, use_graph_scoring, response_mode,
                             trims=trims)]
    if with_hydrate:
        tools.append(hydrate_chunks_tool(search_url, search_log, path_prefix))
    if repo_path:
        from treeloom.adapters.benchmark.grep_tools import read_file_tool, glob_tool
        # The hydrate arm deliberately omits read_file: it forces the
        # batch hydrate_chunks tool to be the body-recovery path so the mechanism
        # is actually exercised (with read_file present the agent ignores hydrate
        # entirely and the arm just reproduces treeloom-facet). glob stays for
        # file discovery.
        if not with_hydrate:
            tools.append(read_file_tool(repo_path, search_log))
        tools.append(glob_tool(repo_path))
    return tools
