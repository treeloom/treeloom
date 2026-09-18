"""Standalone aider repo-map generator — runs INSIDE an aider-chat env.

Invoked by repomap_tool.build_repo_map as a subprocess, e.g.:
    uv run --with aider-chat python gen_repomap_script.py \
        --repo /path/to/repo --map-tokens 8192 --out /tmp/map.txt

Must not import treeloom (it isn't installed in the aider env). Writes the
map to --out (not stdout — aider's Model/IO init prints chatter). The map is
aider's real RepoMap: tree-sitter symbol tags ranked by PageRank over the
def/ref graph, binary-searched down to the token budget. `map_mul_no_files=1`
so --map-tokens IS the budget (aider's own no-files default is 1024 x 8).

Side effect: aider caches tags in <repo>/.aider.tags.cache.v* (speeds re-runs).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--map-tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = os.path.realpath(args.repo)

    from aider.io import InputOutput
    from aider.models import Model
    from aider.repomap import RepoMap

    io = InputOutput(yes=True)
    model = Model("gpt-4o")  # used only for token counting (tiktoken), no API calls
    rm = RepoMap(
        map_tokens=args.map_tokens,
        root=repo,
        main_model=model,
        io=io,
        verbose=False,
        map_mul_no_files=1,
    )
    files = subprocess.run(
        ["git", "-C", repo, "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    other_files = [os.path.join(repo, f) for f in files]

    repo_map = rm.get_repo_map(set(), other_files) or ""
    if not repo_map.strip():
        print("ERROR: aider produced an empty repo map", file=sys.stderr)
        sys.exit(1)
    with open(args.out, "w") as f:
        f.write(repo_map)
    print(f"repo map: {len(repo_map)} chars for {len(other_files)} files", file=sys.stderr)


if __name__ == "__main__":
    main()
