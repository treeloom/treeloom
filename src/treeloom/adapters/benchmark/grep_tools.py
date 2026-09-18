"""Benchmark adapter: grep / glob / read_file tools for the grep-arm agent.

Mirrors what a Claude-Code-style agent does: search with ripgrep, list files
with glob, and read explicit line ranges. All paths are resolved and confined
to the repo root. `read_file` records the resolved absolute path so the runner
can compare what the agent *read* against the query's ground-truth files.
"""
from __future__ import annotations

import os
from pathlib import Path

from treeloom.adapters.benchmark.ripgrep import run_ripgrep
from treeloom.adapters.benchmark.tool_base import Tool


def _resolve_within_repo(repo: str, path: str) -> Path | None:
    """Resolve `path` (absolute or repo-relative) and confine it to `repo`."""
    repo_root = Path(repo).resolve()
    p = Path(path)
    candidate = (p if p.is_absolute() else repo_root / p).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        return None
    return candidate


def grep_tool(repo: str) -> Tool:
    async def run(pattern: str = "", max_matches: int = 50, **_):
        if not pattern:
            return {"error": "pattern is required"}
        rows = run_ripgrep(repo, str(pattern), int(max_matches))
        return {
            "matches": [
                {"file": f, "line": n, "text": t.rstrip()[:300]} for f, n, t in rows
            ],
            "match_count": len(rows),
            "truncated": len(rows) >= int(max_matches),
        }

    return Tool(
        name="grep",
        description=(
            "Search file contents with ripgrep. args: pattern (regex/identifier), "
            "max_matches (default 50). Returns matching {file,line,text}."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_matches": {"type": "integer", "default": 50},
            },
            "required": ["pattern"],
        },
    )


def glob_tool(repo: str) -> Tool:
    async def run(pattern: str = "", limit: int = 100, **_):
        if not pattern:
            return {"error": "pattern is required"}
        root = Path(repo).resolve()
        files: list[str] = []
        try:
            for p in root.rglob(pattern.lstrip("/")):
                if p.is_file():
                    files.append(str(p))
                if len(files) > int(limit):
                    break
        except (ValueError, OSError) as e:
            return {"error": f"glob failed: {e}"}
        truncated = len(files) > int(limit)
        return {"files": files[: int(limit)], "truncated": truncated}

    return Tool(
        name="glob",
        description=(
            "List files matching a glob pattern under the repo (e.g. '**/*.ts'). "
            "args: pattern, limit (default 100)."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["pattern"],
        },
    )


def read_file_tool(repo: str, read_log: list[str]) -> Tool:
    """read_file appends every successfully-read resolved path to `read_log`
    (the runner uses it as the grep arm's 'retrieved files' for recall)."""

    async def run(path: str = "", start_line: int = 1, end_line: int = 200, **_):
        if not path:
            return {"error": "path is required"}
        resolved = _resolve_within_repo(repo, str(path))
        if resolved is None or not resolved.is_file():
            return {"error": f"file not found or outside repo: {path}"}
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return {"error": f"read failed: {e}"}
        s = max(1, int(start_line))
        e = min(len(lines), int(end_line)) if end_line else len(lines)
        if e < s:
            e = min(len(lines), s + 199)
        content = "".join(lines[s - 1 : e])
        abspath = os.path.realpath(str(resolved))
        if abspath not in read_log:
            read_log.append(abspath)
        return {
            "path": abspath,
            "start_line": s,
            "end_line": e,
            "total_lines": len(lines),
            "content": content,
        }

    return Tool(
        name="read_file",
        description=(
            "Read a line range from a file. args: path, start_line (default 1), "
            "end_line (default 200)."
        ),
        run=run,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "default": 200},
            },
            "required": ["path"],
        },
    )


def grep_arm_tools(repo: str, read_log: list[str]) -> list[Tool]:
    return [grep_tool(repo), glob_tool(repo), read_file_tool(repo, read_log)]
