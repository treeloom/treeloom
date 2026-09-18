"""CLI entry: python -m treeloom.backfill_provenance [options].

Backfill git provenance (commit_sha / url / branch) onto already-indexed
source_records rows that were indexed before the provenance feature
captured HEAD — the classic symptom is /search returning chunks with no
`permalink_base` in the response `sources` map, because `commit_sha` (and
often `url`) are empty.

Current indexing already captures HEAD on every path (`_git_head_sha` in
indexer_service, persisted in `_finalize_job`), so the empty rows are
historical. This tool is a one-shot repair: for each path-backed source
whose stored `commit_sha` is empty and whose on-disk `path` is inside a git
work tree, it resolves the SHA / origin URL / branch from `.git` and UPDATEs
the row (both the Postgres source registry and the Neo4j/sqlite `Source`
node, which also carries `commit_sha`).

Sources that are not git work trees, or have no `origin` remote, are left
untouched (their citations stay in the degraded `repo:path:lines` form with
no permalink). Rows that already have a non-empty `commit_sha` are skipped
unless `--force` is given.

This runs **in-process** against the same Postgres + graph store the indexer
uses — it does NOT go through the HTTP API. It therefore needs the same env
(`DATABASE_URL`, and `GRAPH_STORE` + its service vars) reachable, exactly
like running the indexer on the host.

Examples:
  # Preview only — resolve provenance, print what would change, write nothing
  python -m treeloom.backfill_provenance --dry-run

  # Apply the backfill
  python -m treeloom.backfill_provenance

  # Re-resolve even rows that already have a commit_sha
  python -m treeloom.backfill_provenance --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass

# Reuse the exact HEAD-resolution helper that live indexing uses, so the
# backfilled SHA matches what a fresh index would write.
from treeloom.application.indexer_runners import _git_head_sha

logger = logging.getLogger("treeloom.backfill_provenance")


# ── git resolution (pure-ish, no DB) ─────────────────────────────────────


def normalize_git_url(url: str) -> str:
    """Normalize an origin URL toward the https form provenance permalinks need.

    `domain/provenance.permalink()` builds `<url-without-.git>/blob/<sha>/...`,
    which only yields a clickable link for an https(-shaped) URL. An SSH
    remote (`git@github.com:org/repo.git` or `ssh://git@host/org/repo.git`)
    would produce a broken `git@host:.../blob/...` base, so convert it to
    `https://host/org/repo`. Already-https (or other) URLs are returned with
    only a trailing-`.git`/slash trim applied.
    """
    url = (url or "").strip()
    if not url:
        return ""
    # scp-like syntax: git@host:org/repo(.git)
    if url.startswith("git@") and ":" in url and "://" not in url:
        host, _, path = url[len("git@"):].partition(":")
        url = f"https://{host}/{path}"
    # ssh://git@host/org/repo(.git)
    elif url.startswith("ssh://"):
        rest = url[len("ssh://"):]
        if "@" in rest:
            rest = rest.split("@", 1)[1]
        url = f"https://{rest}"
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def _git_text(path: str, *args: str) -> str:
    """Run `git -C <path> <args...>` and return stripped stdout, '' on error."""
    try:
        from git.cmd import Git

        out = Git(path).execute(["git", *args])
        return (out or "").strip()
    except Exception:
        return ""


def git_origin_url(path: str) -> str:
    """Origin remote URL for the work tree at `path` ('' if none)."""
    return _git_text(path, "remote", "get-url", "origin")


def git_branch(path: str) -> str:
    """Current branch name for `path` ('' if detached or not a repo).

    `rev-parse --abbrev-ref HEAD` returns the literal "HEAD" when detached;
    we map that to '' so a detached checkout records no branch rather than a
    bogus one.
    """
    branch = _git_text(path, "rev-parse", "--abbrev-ref", "HEAD")
    return "" if branch in ("", "HEAD") else branch


@dataclass
class ResolvedProvenance:
    commit_sha: str
    url: str
    branch: str

    @property
    def is_git(self) -> bool:
        return bool(self.commit_sha)


def resolve_provenance(path: str) -> ResolvedProvenance:
    """Resolve commit_sha / origin-url / branch from the on-disk git tree.

    Pure of any DB access — unit-testable against a temp git repo. Returns a
    `ResolvedProvenance` whose `is_git` is False (empty sha) when `path` is
    not a git work tree, in which case the caller should skip the row.
    """
    sha = _git_head_sha(path)
    if not sha:
        return ResolvedProvenance("", "", "")
    return ResolvedProvenance(
        commit_sha=sha,
        url=normalize_git_url(git_origin_url(path)),
        branch=git_branch(path),
    )


# ── per-source backfill (decides what to write, given a resolver) ─────────


@dataclass
class BackfillOutcome:
    source_id: str
    path: str
    action: str  # "update" | "skip-has-sha" | "skip-no-path" | "skip-non-git"
    commit_sha: str = ""
    url: str = ""
    branch: str = ""


def plan_source(
    record,
    resolve=resolve_provenance,
    force: bool = False,
) -> BackfillOutcome:
    """Decide the backfill action for a single SourceRecord. Pure given the
    injected `resolve` callable, so it is fully unit-testable.

    - Skip rows with no local `path` (URL-only sources can't be read locally).
    - Skip rows that already have a `commit_sha` unless `force`.
    - Skip rows whose path is not a git work tree (leave provenance empty).
    - Otherwise: update with the resolved sha/url/branch.
    """
    path = (record.path or "").strip()
    if not path:
        return BackfillOutcome(record.id, "", "skip-no-path")
    if record.commit_sha and not force:
        return BackfillOutcome(record.id, path, "skip-has-sha")
    resolved = resolve(path)
    if not resolved.is_git:
        return BackfillOutcome(record.id, path, "skip-non-git")
    return BackfillOutcome(
        record.id,
        path,
        "update",
        commit_sha=resolved.commit_sha,
        url=resolved.url,
        branch=resolved.branch,
    )


# ── DB / graph wiring ─────────────────────────────────────────────────────


async def _apply_update(record, outcome: BackfillOutcome) -> None:
    """Persist a planned update to both Postgres and the graph store."""
    from dataclasses import replace

    from treeloom import graph_store
    from treeloom.adapters.sources.repository import PostgreSourceRepository

    repo = PostgreSourceRepository()
    updated = replace(
        record,
        commit_sha=outcome.commit_sha,
        url=outcome.url or record.url,
        branch=outcome.branch or record.branch,
    )
    await repo.save(updated)

    # The Source node carries commit_sha too; provenance reads from the PG
    # registry but keep the graph node consistent. Best-effort.
    try:
        await graph_store.upsert_source({
            "id": updated.id,
            "path": updated.path,
            "url": updated.url,
            "branch": updated.branch,
            "indexed_at": int(updated.indexed_at.timestamp()),
            "file_count": updated.file_count,
            "chunk_count": updated.chunk_count,
            "commit_sha": updated.commit_sha,
        })
    except Exception:
        logger.warning("graph upsert_source failed for %s (PG row updated)", updated.id)


async def _run(args: argparse.Namespace) -> int:
    # The asyncpg pool is process-global and lazily None until init_pool() runs;
    # PostgreSourceRepository (and graph_store) resolve it via get_pool(), so the
    # pool must be initialized before any query or list_all() returns empty and
    # the backfill silently no-ops ("No indexed sources found").
    from treeloom.adapters.postgresql.connection import close_pool, init_pool
    from treeloom.adapters.sources.repository import PostgreSourceRepository

    await init_pool()
    try:
        repo = PostgreSourceRepository()
        records = await repo.list_all()
        if not records:
            print("No indexed sources found (or DATABASE_URL unreachable) — nothing to do.")
            return 0

        updated = 0
        skipped = 0
        for record in records:
            outcome = plan_source(record, force=args.force)
            if outcome.action != "update":
                skipped += 1
                if args.verbose:
                    print(f"  {outcome.action:14} {outcome.source_id}  {outcome.path}")
                continue
            if args.dry_run:
                print(
                    f"  would update  {outcome.source_id}  "
                    f"sha={outcome.commit_sha[:12]} branch={outcome.branch or '-'} "
                    f"url={outcome.url or '-'}"
                )
                updated += 1
                continue
            await _apply_update(record, outcome)
            print(
                f"  updated       {outcome.source_id}  "
                f"sha={outcome.commit_sha[:12]} branch={outcome.branch or '-'} "
                f"url={outcome.url or '-'}"
            )
            updated += 1

        verb = "would update" if args.dry_run else "updated"
        print(f"\n{verb} {updated} / skipped {skipped} (of {len(records)} sources).")
        return 0
    finally:
        await close_pool()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m treeloom.backfill_provenance",
        description="Backfill git provenance onto already-indexed sources.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print planned updates without writing anything.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-resolve even rows that already have a commit_sha.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also print skipped rows and the reason.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
