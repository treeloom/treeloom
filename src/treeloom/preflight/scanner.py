"""Filesystem scanner — collects cheap per-file metadata for preflight.

Walks a directory tree, filters to SUPPORTED_EXTENSIONS (code + markdown),
collects size + LOC + language. Respects .gitignore via `git ls-files`
when the path is a git repo; falls back to a manual walk with
configurable skip dirs/patterns.

Output is consumed by treeloom.preflight.model.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from treeloom.adapters.tree_sitter.indexer import SUPPORTED_EXTENSIONS, detect_language

logger = logging.getLogger(__name__)

# Treat clearly-noisy directories as skipped even without an explicit
# `.gitignore` entry. Operator can override via skip_dirs.
DEFAULT_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "vendor",
    ".next", ".nuxt", "dist", "build", "out", "target",
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".tox", ".cache", ".coverage", "htmlcov",
})

# Files that are commonly generated / not worth indexing. The scanner
# itself doesn't refuse these (preflight is read-only); the recommender
# may suggest them as skip_patterns.
NOISY_PATTERNS: tuple[str, ...] = (
    "*.min.js", "*.bundle.js", "*.min.css",
    "package-lock.json", "yarn.lock", "Cargo.lock", "Gemfile.lock",
    "poetry.lock", "pnpm-lock.yaml",
    "*.map",  # source maps
)

# Hard cap on file size we'll bother reading for LOC. Anything bigger
# is reported with size only; LOC defaults to size/40 (rough text guess).
_LOC_READ_CAP_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass(frozen=True)
class FileStat:
    path: str               # absolute
    rel_path: str           # relative to scan root
    size_bytes: int
    loc: int
    language: str           # empty if unsupported (we filter those out)
    max_line_length: int    # 0 if uncounted


_BLOCK_SIZE = 1 << 16


def _is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def _gitignored_files(root: Path) -> set[str] | None:
    """Return the set of rel-paths git considers tracked + untracked-but-
    not-ignored. Returns None if `git ls-files` fails."""
    try:
        result = subprocess.run(
            # `-c` overrides whatever the target repo's .git/config says.
            # git consults core.fsmonitor while enumerating with --others and
            # will EXECUTE the program it names, so a repo the caller planted
            # on a readable path could run code as the indexer (CWE-78).
            # Blanking it, and hooksPath, removes that; GIT_CONFIG_NOSYSTEM
            # stops /etc/gitconfig contributing either. Global config is left
            # alone so core.excludesFile still shapes --exclude-standard.
            ["git",
             "-c", "core.fsmonitor=",
             "-c", "core.hooksPath=/dev/null",
             "-C", str(root), "ls-files",
             "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30, check=True,
            env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
        )
        return set(result.stdout.splitlines())
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _matches_any(rel_path: str, patterns: tuple[str, ...]) -> bool:
    name = os.path.basename(rel_path)
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel_path, p)
               for p in patterns)


def _count_loc_and_max_line(path: str, size_bytes: int) -> tuple[int, int]:
    """Count lines and the longest line, reading in bounded blocks.

    Deliberately NOT `for line in f`. Line iteration is bounded by the next
    newline, not by any byte count, so a stream that never produces one — a
    character device such as /dev/zero, reachable through a symlink — makes
    the first "line" grow until the process dies. Reproduced before this
    change: MemoryError against a repo containing `evil.py -> /dev/zero`, and
    a repo cloned from a URL can contain exactly that, so /preflight's url
    mode reaches it (CWE-835).

    walk_repo now refuses anything that is not a regular file, which is the
    primary fix. This is the independent one: fixed-size reads cannot outrun
    the cap whatever they are pointed at, and they also bound an ordinary
    file that grew between its stat and this read.
    """
    if size_bytes > _LOC_READ_CAP_BYTES:
        # Don't read multi-tens-of-MB files; estimate.
        return (size_bytes // 40, 0)
    loc = 0
    max_line = 0
    run = 0          # bytes since the last newline, across block boundaries
    read = 0
    try:
        with open(path, "rb") as f:
            while read < _LOC_READ_CAP_BYTES:
                block = f.read(min(_BLOCK_SIZE, _LOC_READ_CAP_BYTES - read))
                if not block:
                    break
                read += len(block)
                start = 0
                while True:
                    nl = block.find(b"\n", start)
                    if nl == -1:
                        run += len(block) - start
                        if run > max_line:
                            max_line = run
                        break
                    loc += 1
                    run += nl - start + 1
                    if run > max_line:
                        max_line = run
                    run = 0
                    start = nl + 1
    except OSError:
        return (size_bytes // 40, 0)
    # A trailing fragment with no newline is still a line.
    if run:
        loc += 1
    return (loc, max_line)


def walk_repo(
    path: str,
    *,
    skip_dirs: frozenset[str] | set[str] = DEFAULT_SKIP_DIRS,
    skip_patterns: tuple[str, ...] = (),
    use_gitignore: bool = True,
) -> tuple[list[FileStat], int]:
    """Scan a directory tree and return (supported file stats, unsupported count).

    `unsupported count` = files we walked over but couldn't index because
    the extension is not in SUPPORTED_EXTENSIONS. Useful for the report.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    gitignored: set[str] | None = None
    if use_gitignore and _is_git_repo(root):
        gitignored = _gitignored_files(root)

    supported: list[FileStat] = []
    unsupported_count = 0
    skip_set = set(skip_dirs)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in skip_set and not d.startswith(".")]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)

            if gitignored is not None and rel not in gitignored:
                continue
            if _matches_any(rel, skip_patterns):
                continue

            ext = os.path.splitext(fname)[1]
            if ext not in SUPPORTED_EXTENSIONS:
                unsupported_count += 1
                continue
            lang = detect_language(full) or ""

            try:
                # stat() follows symlinks, which is what we want for the size
                # — but it also means a link to a device or a FIFO arrives
                # here looking like a file. S_ISREG is the check that keeps
                # the reader below off anything that does not end (CWE-835):
                # a character device reports 0 bytes and never yields a
                # newline. A repo cloned from a URL can contain such a link,
                # so this is reachable from /preflight's url mode.
                st = os.stat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            size = st.st_size

            loc, max_line = _count_loc_and_max_line(full, size)
            supported.append(FileStat(
                path=full,
                rel_path=rel,
                size_bytes=size,
                loc=loc,
                language=lang,
                max_line_length=max_line,
            ))

    return supported, unsupported_count
