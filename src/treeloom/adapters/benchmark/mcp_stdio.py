"""Benchmark adapter: shared stdio MCP client for external-competitor arms.

Spawns a competitor's MCP server as a subprocess and drives it over stdio.
One session per benchmark run; start()/close() must run in the same task
(anyio cancel-scope rule) — run_agentic owns the lifecycle.
"""
from __future__ import annotations

import os
import shlex
from contextlib import AsyncExitStack
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPStdioClient:
    """Minimal lifecycle + call wrapper around one stdio MCP session."""

    def __init__(self, command: str, *, env: dict[str, str] | None = None,
                 cwd: str | None = None, startup_hint: str = ""):
        self._cmd = command
        self._env = env if env is not None else dict(os.environ)
        self._cwd = cwd
        self._startup_hint = startup_hint
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def start(self) -> None:
        argv = shlex.split(self._cmd)
        params = StdioServerParameters(
            command=argv[0], args=argv[1:], env=self._env, cwd=self._cwd
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        try:
            await self._session.initialize()
        except Exception as e:
            # The server exits before the handshake when its config is bad —
            # the generic "Connection closed" hides that.
            await self.close()
            hint = f" {self._startup_hint}" if self._startup_hint else ""
            raise RuntimeError(
                f"MCP server failed to start ({e}).{hint} "
                "Check the server log lines above."
            ) from e

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None

    async def call(self, name: str, args: dict, timeout_s: float = 60.0) -> str:
        """Call a tool and return its text content; raises on tool error."""
        if self._session is None:
            raise RuntimeError(f"{type(self).__name__} not started")
        result = await self._session.call_tool(
            name, args, read_timeout_seconds=timedelta(seconds=timeout_s)
        )
        text = "\n".join(
            t for t in (getattr(c, "text", None) for c in result.content or []) if t
        )
        if getattr(result, "isError", False):
            raise RuntimeError(f"{name} failed: {text or 'unknown MCP error'}")
        return text
