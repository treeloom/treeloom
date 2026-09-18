"""Benchmark adapter: the claude-context-arm `search_code` tool.

Drives zilliztech/claude-context (the Milvus-backed hybrid BM25+dense code
search MCP server) as a real MCP stdio client: spawns the Node server once per
benchmark run, indexes the repo, and exposes its `search_code` tool to the
agent. The observation fed to the agent is the server's formatted text payload
verbatim, so token accounting reflects what an agent using claude-context
actually pays. `Location: <relpath>:<start>-<end>` lines are parsed (in rank
order) into `search_log` for recall/MRR, mirroring the treeloom arm.

The server is configured entirely by environment variables it reads itself
(passed through from this process): MILVUS_ADDRESS / MILVUS_TOKEN and an
embedding provider (EMBEDDING_PROVIDER, EMBEDDING_MODEL, OPENAI_API_KEY,
OPENAI_BASE_URL for any OpenAI-compatible endpoint). Override the launch
command with CLAUDE_CONTEXT_MCP_CMD (default: npx -y @zilliz/claude-context-mcp@latest).
"""
from __future__ import annotations

import asyncio
import os
import re
import time

from treeloom.adapters.benchmark.mcp_stdio import MCPStdioClient
from treeloom.adapters.benchmark.tool_base import Tool

_DEFAULT_CMD = "npx -y @zilliz/claude-context-mcp@latest"

# Per-result line in the server's search_code text output:
#   Location: src/foo/bar.ts:10-42
_LOCATION_RE = re.compile(r"Location:\s*(.+?):\d+-\d+\s*$", re.MULTILINE)


class ClaudeContextClient(MCPStdioClient):
    """One MCP stdio session to a claude-context server, reused across queries."""

    def __init__(self, command: str | None = None):
        super().__init__(
            command or os.environ.get("CLAUDE_CONTEXT_MCP_CMD", _DEFAULT_CMD),
            # Full env passthrough: the server reads its embedding + Milvus
            # config (MILVUS_ADDRESS, OPENAI_API_KEY, ...) from env.
            env=dict(os.environ),
            startup_hint=(
                "claude-context exits at startup unless its env is set: "
                "MILVUS_ADDRESS (+MILVUS_TOKEN) and an embedding provider "
                "(EMBEDDING_PROVIDER, EMBEDDING_MODEL, OPENAI_API_KEY, "
                "OPENAI_BASE_URL for any OpenAI-compatible endpoint)."
            ),
        )


    async def ensure_indexed(
        self, repo: str, *, force: bool = False,
        timeout_s: float = 3600.0, poll_s: float = 5.0,
    ) -> str:
        """Kick off indexing (no-op if already indexed) and poll status until
        the codebase is searchable. Returns the final status text."""
        repo = os.path.realpath(repo)
        try:
            await self.call(
                "index_codebase", {"path": repo, "force": force}, timeout_s=300
            )
        except RuntimeError as e:
            # "already indexed" comes back as a tool error on some versions;
            # the status poll below is authoritative either way.
            if "already" not in str(e).lower():
                raise
        deadline = time.monotonic() + timeout_s
        while True:
            status = await self.call("get_indexing_status", {"path": repo})
            if "fully indexed" in status or "✅" in status:
                return status
            if "❌" in status or "failed" in status.lower():
                raise RuntimeError(f"claude-context indexing failed: {status}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"claude-context indexing did not finish in {timeout_s:.0f}s: {status}"
                )
            await asyncio.sleep(poll_s)

    async def search(self, repo: str, query: str, limit: int) -> str:
        return await self.call(
            "search_code",
            {"path": os.path.realpath(repo), "query": query, "limit": limit},
            timeout_s=120,
        )


def claude_context_search_tool(
    client: ClaudeContextClient, repo: str, search_log: list[str],
    top_k_override: int | None = None,
) -> Tool:
    async def run(query: str = "", top_k: int = 10, **_):
        if not query:
            return {"error": "query is required"}
        # Same sweep semantics as the treeloom arm: a fixed top_k overrides
        # whatever the agent asks for. claude-context caps limit at 50.
        k = int(top_k_override) if top_k_override is not None else int(top_k)
        k = max(1, min(k, 50))
        try:
            text = await client.search(repo, str(query), k)
        except Exception as e:
            return {"error": f"claude-context search failed: {e}"}
        for rel in _LOCATION_RE.findall(text):
            ap = os.path.realpath(os.path.join(repo, rel.strip()))
            if ap not in search_log:
                search_log.append(ap)
        # Verbatim payload — this is what a claude-context MCP user's agent sees.
        return {"results": text}

    return Tool(
        name="search_code",
        description=(
            "Hybrid (BM25 + dense vector) semantic code search over the indexed "
            "repo. args: query (natural language), top_k (default 10, max 50). "
            "Returns ranked code snippets with file path and line range."
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


def claude_context_arm_tools(
    client: ClaudeContextClient, repo_path: str, search_log: list[str],
    top_k_override: int | None = None,
) -> list[Tool]:
    """search_code (claude-context) + the same read/glob helpers the treeloom
    arm gets, so the arms differ only in their search tool."""
    from treeloom.adapters.benchmark.grep_tools import glob_tool, read_file_tool

    return [
        claude_context_search_tool(client, repo_path, search_log, top_k_override),
        read_file_tool(repo_path, search_log),
        glob_tool(repo_path),
    ]
