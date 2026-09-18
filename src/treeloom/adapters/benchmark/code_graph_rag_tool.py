"""Benchmark adapter: the code-graph-rag arm (vitali87's `code-graph-rag`).

Drives the project's real MCP server (`cgr mcp-server`, stdio) like the
claude-context arm. Its retrieval mechanism is text-to-Cypher: the
`query_code_graph` tool sends the natural-language query to a configured
CYPHER_* LLM which generates a Cypher query against a Memgraph knowledge
graph (tree-sitter-extracted Module/Class/Function nodes + relationships);
`get_code_snippet` fetches a definition's source by qualified name. Those
internal LLM calls are server-side and NOT in the agent transcript — same
out-of-band treatment as claude-context's embedding calls.

Environment the spawned server needs (passed through / defaulted here):
  MEMGRAPH_HOST / MEMGRAPH_PORT  — defaults localhost:7688 (NOT 7687: that's
      treeloom's Neo4j bolt port; run e.g.
      `docker run -d --name treeloom-bench-memgraph -p 7688:7687 memgraph/memgraph`)
  CYPHER_PROVIDER / CYPHER_MODEL / CYPHER_API_KEY / CYPHER_ENDPOINT — the
      text-to-Cypher model (e.g. openai / deepseek-chat / sk-... /
      https://api.deepseek.com/v1). ORCHESTRATOR_* mirrors it (some code
      paths validate both).
  TARGET_REPO_PATH — set automatically from the arm's repo.

The package needs Python >= 3.12, so the default launch command runs it in an
isolated env: `uvx --python 3.12 --from code-graph-rag cgr mcp-server`
(override with CGR_MCP_CMD). The server is spawned with cwd in a scratch dir
because it pydantic-parses any `.env` it finds in cwd with extra=forbid —
treeloom's .env crashes it.
"""
from __future__ import annotations

import json
import os
import tempfile

from treeloom.adapters.benchmark.mcp_stdio import MCPStdioClient
from treeloom.adapters.benchmark.tool_base import Tool

# treesitter-full adds JS/TS/Java/Go/Rust/... grammars. NOTE: the released
# package cannot parse C# at all (SupportedLanguage.CSHARP exists but no
# tree_sitter module mapping ships) — on C#-heavy repos the graph only covers
# the other languages; surface that caveat next to any results.
_DEFAULT_CMD = "uvx --python 3.12 --from code-graph-rag[treesitter-full] cgr mcp-server"

_PATH_KEYS = ("file_path", "path", "relative_path")


def _collect_paths(obj, out: list[str]) -> None:
    """Walk a result payload collecting file-path-ish string values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _PATH_KEYS and isinstance(v, str) and v:
                out.append(v)
            else:
                _collect_paths(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_paths(v, out)


class CodeGraphRagClient(MCPStdioClient):
    """One MCP stdio session to a `cgr mcp-server`, reused across queries."""

    def __init__(self, repo: str, command: str | None = None):
        self.repo = os.path.realpath(repo)
        self.project_name = os.path.basename(self.repo)
        env = dict(os.environ)
        env["TARGET_REPO_PATH"] = self.repo
        env.setdefault("MEMGRAPH_HOST", "localhost")
        env.setdefault("MEMGRAPH_PORT", "7688")
        super().__init__(
            command or os.environ.get("CGR_MCP_CMD", _DEFAULT_CMD),
            env=env,
            # Scratch cwd: cgr pydantic-parses ./.env with extra=forbid, so
            # spawning from the treeloom root (or any dir with a foreign .env)
            # kills it at import time.
            cwd=tempfile.mkdtemp(prefix="cgr-mcp-"),
            startup_hint=(
                "code-graph-rag needs a reachable Memgraph "
                "(MEMGRAPH_HOST/MEMGRAPH_PORT, default localhost:7688 here) "
                "and a CYPHER_* LLM config (CYPHER_PROVIDER/MODEL/API_KEY/"
                "ENDPOINT)."
            ),
        )

    async def ensure_indexed(self, *, force: bool = False,
                             timeout_s: float = 3600.0) -> str:
        """Index TARGET_REPO_PATH into the graph unless its project exists.

        `index_repository` clears + re-parses unconditionally, so the skip
        check matters: a featbit-sized repo takes minutes to ingest.
        """
        if not force:
            try:
                listing = await self.call("list_projects", {}, timeout_s=60)
                if self.project_name in listing:
                    return f"project '{self.project_name}' already indexed"
            except RuntimeError:
                pass  # listing failure -> just index
        return await self.call("index_repository", {}, timeout_s=timeout_s)

    async def query(self, natural_language_query: str) -> str:
        return await self.call(
            "query_code_graph",
            {"natural_language_query": natural_language_query},
            timeout_s=180,
        )

    async def snippet(self, qualified_name: str) -> str:
        return await self.call(
            "get_code_snippet", {"qualified_name": qualified_name}, timeout_s=120
        )


def _log_paths_from(text: str, repo: str, search_log: list[str]) -> None:
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return
    found: list[str] = []
    _collect_paths(payload, found)
    for p in found:
        ap = os.path.realpath(p if os.path.isabs(p) else os.path.join(repo, p))
        if ap not in search_log:
            search_log.append(ap)


def query_code_graph_tool(client: CodeGraphRagClient, repo: str,
                          search_log: list[str]) -> Tool:
    async def run(query: str = "", **_):
        if not query:
            return {"error": "query is required"}
        try:
            text = await client.query(str(query))
        except Exception as e:
            return {"error": f"code-graph-rag query failed: {e}"}
        _log_paths_from(text, repo, search_log)
        return {"results": text}

    return Tool(
        name="query_code_graph",
        description=(
            "Query the codebase knowledge graph in natural language (translated "
            "to a Cypher graph query server-side). Returns matching entities "
            "(modules/classes/functions) with qualified names and file paths. "
            "args: query."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def get_code_snippet_tool(client: CodeGraphRagClient, repo: str,
                          search_log: list[str]) -> Tool:
    async def run(qualified_name: str = "", **_):
        if not qualified_name:
            return {"error": "qualified_name is required"}
        try:
            text = await client.snippet(str(qualified_name))
        except Exception as e:
            return {"error": f"code-graph-rag snippet failed: {e}"}
        _log_paths_from(text, repo, search_log)
        return {"results": text}

    return Tool(
        name="get_code_snippet",
        description=(
            "Fetch the source code of a definition by its qualified name from "
            "query_code_graph results (e.g. 'project.module.Class.method'). "
            "args: qualified_name."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {"qualified_name": {"type": "string"}},
            "required": ["qualified_name"],
        },
    )


def code_graph_rag_arm_tools(client: CodeGraphRagClient, repo_path: str,
                             search_log: list[str]) -> list[Tool]:
    """Graph query + snippet + the same read/glob helpers the other arms get."""
    from treeloom.adapters.benchmark.grep_tools import glob_tool, read_file_tool

    return [
        query_code_graph_tool(client, repo_path, search_log),
        get_code_snippet_tool(client, repo_path, search_log),
        read_file_tool(repo_path, search_log),
        glob_tool(repo_path),
    ]
