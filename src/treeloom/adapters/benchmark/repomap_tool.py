"""Benchmark adapter: the aider repo-map arm.

Aider's retrieval mechanism is not a search tool — it's ambient context: a
tree-sitter symbol map of the repo, PageRank-ranked and fitted to a token
budget, re-sent with every request. This arm reproduces that faithfully: the
map is built once per run with aider's real RepoMap (in an isolated env via
`uv run --with aider-chat`, so aider's pinned deps never touch this venv) and
injected into the agent's system prompt, where it is re-paid every turn — the
honest token cost of the repo-map approach. The agent gets only read_file +
glob (aider has no grep; the map replaces search), and recall/MRR comes from
what it actually reads, same as the grep arm.

Override the generator command with REPOMAP_PYTHON_CMD (default:
`uv run --no-project --with aider-chat python` — --no-project keeps uv from
resolving/locking the treeloom pyproject it finds in cwd). First invocation
downloads aider-chat into uv's cache; aider also caches tags in
<repo>/.aider.tags.cache.v*.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from treeloom.adapters.benchmark.tool_base import Tool

_DEFAULT_CMD = "uv run --no-project --with aider-chat python"

_MAP_PREAMBLE = (
    "Below is a map of the repository: symbol summaries of the most important "
    "files (ranked by reference-graph centrality). Use it as an index to decide "
    "which files to read — always confirm with read_file before answering; the "
    "map alone is not an observation of file contents.\n\n"
)


def build_repo_map(repo: str, map_tokens: int = 8192, timeout_s: float = 900.0) -> str:
    """Generate aider's repo map for `repo` in an isolated aider-chat env."""
    script = Path(__file__).with_name("gen_repomap_script.py")
    cmd = shlex.split(os.environ.get("REPOMAP_PYTHON_CMD", _DEFAULT_CMD))
    with tempfile.NamedTemporaryFile(mode="r", suffix=".repomap.txt") as out:
        proc = subprocess.run(
            [*cmd, str(script), "--repo", os.path.realpath(repo),
             "--map-tokens", str(map_tokens), "--out", out.name],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"repo-map generation failed (exit {proc.returncode}). "
                f"Command: {' '.join(cmd)} {script.name} ... Stderr tail:\n{tail}"
            )
        repo_map = out.read()
    if not repo_map.strip():
        raise RuntimeError("repo-map generation produced an empty map")
    return _MAP_PREAMBLE + repo_map


def repomap_arm_tools(repo_path: str, read_log: list[str]) -> list[Tool]:
    """read_file + glob only — the map (in the system prompt) replaces search."""
    from treeloom.adapters.benchmark.grep_tools import glob_tool, read_file_tool

    return [read_file_tool(repo_path, read_log), glob_tool(repo_path)]
