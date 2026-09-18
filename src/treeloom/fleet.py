"""CLI entry: python -m treeloom.fleet <subcommand> [options].

Fleet operations over many indexed sources:

  onboard  Bulk-index every repo in a manifest (YAML/JSON). Idempotent —
           re-running an unchanged manifest is a no-op because /index-repo
           short-circuits already-DONE sources (pass --force to re-index).

  health   Fetch the /fleet rollup and print a fleet-wide health table (or
           raw JSON), so "is the fleet healthy?" is one command, not N calls.

Auth: when AUTH_ENABLED=true on the indexer, /index-repo and /fleet require a
bearer token. Provide it via --key or the TREELOOM_KEY env var (falling back
to TREELOOM_MCP_API_KEY).

Examples:
  python -m treeloom.fleet onboard repos.yaml --wait
  python -m treeloom.fleet onboard repos.yaml --dry-run
  python -m treeloom.fleet health --staleness --entities
  python -m treeloom.fleet health --json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

from treeloom.domain.sources.manifest import parse_manifest

DEFAULT_INDEXER_URL = "http://localhost:8001"
# Terminal job states (mirror rebuild_graphs / wait_for_index_job).
_TERMINAL = {"done", "failed", "cancelled", "dead_letter"}


def _auth_headers(key: str | None) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def _resolve_key(arg_key: str | None, key_file: str | None = None) -> str | None:
    """See infrastructure/cli_auth — --key exposes the token in ps."""
    from treeloom.infrastructure.cli_auth import resolve_key

    return resolve_key(arg_key, key_file)


async def _poll_until_terminal(
    client: httpx.AsyncClient, base: str, headers: dict,
    job_ids: list[str], interval: float,
) -> dict[str, str]:
    """Poll the given jobs until all reach a terminal state. Returns
    {job_id: final_status}."""
    pending = set(job_ids)
    final: dict[str, str] = {}
    while pending:
        await asyncio.sleep(interval)
        for jid in list(pending):
            try:
                resp = await client.get(f"{base}/jobs/{jid}", headers=headers)
                resp.raise_for_status()
                status = resp.json().get("status", "")
            except httpx.HTTPError:
                continue
            if status in _TERMINAL:
                final[jid] = status
                pending.discard(jid)
                print(f"  [{status:9}] job {jid}")
    return final


# ── onboard ────────────────────────────────────────────────────────────────


async def _run_onboard(args: argparse.Namespace) -> int:
    base = args.indexer_url.rstrip("/")
    headers = _auth_headers(_resolve_key(args.key, getattr(args, 'key_file', None)))

    try:
        text = open(args.manifest, encoding="utf-8").read()
    except OSError as exc:
        print(f"error: cannot read manifest {args.manifest!r}: {exc}", file=sys.stderr)
        return 1
    try:
        entries = parse_manifest(text)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("Manifest is empty — nothing to onboard.")
        return 0

    print(f"Manifest has {len(entries)} repo(s).")
    group_label = args.label or os.path.basename(args.manifest)
    enqueued: list[tuple[str, str]] = []  # (label, job_id)
    already_done = 0
    failed = 0

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        # Create one Job (group) for the whole manifest so the submission shows
        # up as a single unit in the operator UI. Best-effort: if the
        # indexer predates job-groups, proceed ungrouped.
        group_id: str | None = None
        if not args.dry_run:
            try:
                gresp = await client.post(
                    f"{base}/job-groups",
                    json={"label": group_label, "kind": "fleet"},
                    headers=headers,
                )
                gresp.raise_for_status()
                group_id = gresp.json().get("id")
                print(f"Job group {group_id} ({group_label}) for {len(entries)} repo(s).")
            except httpx.HTTPError as exc:
                print(f"warning: could not create job group ({exc}); "
                      "proceeding ungrouped", file=sys.stderr)

        for entry in entries:
            body = entry.to_index_request(force_override=args.force)
            if group_id:
                body["group_id"] = group_id
            if args.dry_run:
                print(f"  would index: {entry.label}  {body}")
                continue
            try:
                resp = await client.post(
                    f"{base}/index-repo", json=body, headers=headers
                )
                resp.raise_for_status()
                ack = resp.json()
            except httpx.HTTPError as exc:
                failed += 1
                print(f"  FAILED {entry.label}: {exc}", file=sys.stderr)
                continue
            status = ack.get("status", "")
            job_id = ack.get("job_id", "")
            if status == "done":
                already_done += 1
                print(f"  up-to-date: {entry.label} (job {job_id})")
            else:
                enqueued.append((entry.label, job_id))
                print(f"  queued {status}: {entry.label} (job {job_id})")

        if args.dry_run:
            return 0

        print(
            f"\n{len(enqueued)} enqueued, {already_done} already up-to-date, "
            f"{failed} failed (of {len(entries)} repos)."
        )

        if args.wait and enqueued:
            job_ids = [jid for _, jid in enqueued if jid]
            print(
                f"\nWaiting for {len(job_ids)} jobs to finish "
                f"(polling every {args.poll_interval}s)..."
            )
            final = await _poll_until_terminal(
                client, base, headers, job_ids, args.poll_interval
            )
            n_failed = sum(1 for s in final.values() if s != "done")
            print(f"\nDone. {len(final)} jobs terminal, {n_failed} not 'done'.")
            return 2 if n_failed else 0

    return 1 if failed else 0


# ── health ───────────────────────────────────────────────────────────────


def _fmt_age(seconds) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def _print_health_table(health: dict, *, show_staleness: bool, show_entities: bool) -> None:
    summary = health.get("summary", {})
    rows = health.get("sources", [])

    cols = ["source", "branch", "age", "chunks"]
    if show_entities:
        cols.append("entities")
    cols += ["graph", "last_job"]
    if show_staleness:
        cols.append("stale")

    print("  ".join(f"{c:<10}" if c != "source" else f"{c:<40}" for c in cols))
    for r in rows:
        last = r.get("last_job") or {}
        status = last.get("status", "-")
        errs = last.get("errors", 0)
        last_str = status if not errs else f"{status}({errs}e)"
        cells = [
            f"{(r.get('label') or r.get('id') or ''):<40}",
            f"{(r.get('branch') or '-'):<10}",
            f"{_fmt_age(r.get('age_seconds')):<10}",
            f"{r.get('chunk_count', 0):<10}",
        ]
        if show_entities:
            cells.append(f"{r.get('entity_count', '-'):<10}")
        cells.append(f"{('yes' if r.get('graph_indexed') else 'NO'):<10}")
        cells.append(f"{last_str:<10}")
        if show_staleness:
            is_stale = r.get("is_stale")
            cells.append(
                f"{('STALE' if is_stale else ('fresh' if is_stale is False else '?')):<10}"
            )
        print("  ".join(cells))

    print("\nFleet summary:")
    for k in (
        "sources_total", "sources_stale", "sources_with_errors",
        "sources_graph_missing", "total_chunks", "total_files",
        "total_entities", "staleness_truncated",
    ):
        if k in summary:
            print(f"  {k}: {summary[k]}")
    oldest = summary.get("oldest_index_age_seconds")
    if oldest is not None:
        print(f"  oldest_index_age: {_fmt_age(oldest)}")


async def _run_health(args: argparse.Namespace) -> int:
    base = args.indexer_url.rstrip("/")
    headers = _auth_headers(_resolve_key(args.key, getattr(args, 'key_file', None)))
    params: dict[str, str] = {}
    if args.staleness:
        params["staleness"] = "true"
    if args.entities:
        params["entities"] = "true"

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        try:
            resp = await client.get(f"{base}/fleet", params=params, headers=headers)
            resp.raise_for_status()
            health = resp.json()
        except httpx.HTTPError as exc:
            print(f"error: could not fetch /fleet from {base}: {exc}", file=sys.stderr)
            return 1

    if args.json:
        import json as _json
        print(_json.dumps(health, indent=2, default=str))
        return 0

    _print_health_table(
        health, show_staleness=args.staleness, show_entities=args.entities
    )
    return 0


# ── argparse ─────────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--indexer-url",
        default=os.environ.get("INDEXER_URL") or DEFAULT_INDEXER_URL,
        help=f"Indexer base URL (default: $INDEXER_URL or {DEFAULT_INDEXER_URL}).",
    )
    from treeloom.infrastructure.cli_auth import add_key_arguments

    add_key_arguments(p)
    p.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-request HTTP timeout in seconds (default: 30).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m treeloom.fleet",
        description="Fleet operations over many indexed sources.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_onboard = sub.add_parser("onboard", help="Bulk-index repos from a manifest.")
    p_onboard.add_argument("manifest", help="Path to a YAML/JSON manifest file.")
    p_onboard.add_argument(
        "--force", action="store_true",
        help="Re-index every repo even if already DONE (overrides per-entry force).",
    )
    p_onboard.add_argument(
        "--wait", action="store_true",
        help="Poll until all enqueued jobs reach a terminal state.",
    )
    p_onboard.add_argument(
        "--poll-interval", type=float, default=2.0,
        help="Seconds between status polls when --wait (default: 2.0).",
    )
    p_onboard.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be indexed without calling the indexer.",
    )
    p_onboard.add_argument(
        "--label", default=None,
        help="Label for the Job (group) these repos are submitted under "
             "(default: the manifest filename).",
    )
    _add_common(p_onboard)
    p_onboard.set_defaults(_run=_run_onboard)

    p_health = sub.add_parser("health", help="Print the fleet health rollup.")
    p_health.add_argument(
        "--staleness", action="store_true",
        help="Include live per-source staleness (slower; git HEAD comparison).",
    )
    p_health.add_argument(
        "--entities", action="store_true",
        help="Include per-source graph entity counts.",
    )
    p_health.add_argument(
        "--json", action="store_true", help="Emit raw JSON instead of a table.",
    )
    _add_common(p_health)
    p_health.set_defaults(_run=_run_health)

    args = parser.parse_args(argv)
    return asyncio.run(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main())
