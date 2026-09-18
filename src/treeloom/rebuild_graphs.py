"""CLI entry: python -m treeloom.rebuild_graphs [options].

Bulk-build (or rebuild) the Neo4j code graph for indexed sources by
enqueueing a graph job (POST /index-graph) per source. Use this once to
repair a bulk import where chunks were indexed but the graph was never
built — the classic symptom is find_definition / graph_explore returning
empty across every repo even though search returns chunks.

Graph extraction is idempotent (it drops + re-extracts a source's
entities), so rebuilding a healthy source is harmless.

Why the default rebuilds everything: the `graph_indexed` flag is NOT a
reliable signal for the empty-graph bug. Those jobs ran with
skip_graph=false, so the source was marked graph_indexed=true even though
the graph write silently produced zero entities. A "only graph_indexed=
false" pass would therefore skip exactly the sources that need repair.
Pass --only-missing to opt into the flag-based filter (the normal
two-pass-indexing case).

Auth: when AUTH_ENABLED=true on the indexer, /index-graph requires a
bearer token. Provide it via --key or the TREELOOM_KEY env var (falling
back to TREELOOM_MCP_API_KEY).

Examples:
  # Repair a bulk import — rebuild every source, wait for completion
  python -m treeloom.rebuild_graphs --wait

  # Only sources whose graph was deliberately skipped (two-pass)
  python -m treeloom.rebuild_graphs --only-missing

  # Target a specific indexer + specific sources
  python -m treeloom.rebuild_graphs --indexer-url http://localhost:8001 \
      --source-id 8c4f36e59ccfa521 --source-id 01992f296e83a62c
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

DEFAULT_INDEXER_URL = "http://localhost:8001"
# Terminal job states (mirror wait_for_index_job in mcp_server).
_TERMINAL = {"done", "failed", "cancelled"}


def _auth_headers(key: str | None) -> dict:
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


async def _list_sources(client: httpx.AsyncClient, base: str, headers: dict) -> list[dict]:
    resp = await client.get(f"{base}/sources", headers=headers)
    resp.raise_for_status()
    return resp.json()


async def _enqueue_graph(
    client: httpx.AsyncClient, base: str, headers: dict, source_id: str
) -> dict:
    resp = await client.post(
        f"{base}/index-graph", json={"source_id": source_id}, headers=headers
    )
    resp.raise_for_status()
    return resp.json()


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


async def _run(args: argparse.Namespace) -> int:
    base = args.indexer_url.rstrip("/")
    from treeloom.infrastructure.cli_auth import resolve_key

    headers = _auth_headers(resolve_key(args.key, getattr(args, "key_file", None)))

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if args.source_id:
            targets = [{"id": sid} for sid in args.source_id]
        else:
            try:
                targets = await _list_sources(client, base, headers)
            except httpx.HTTPError as exc:
                print(f"error: could not list sources from {base}: {exc}", file=sys.stderr)
                return 1
            if not targets:
                print("No indexed sources found — nothing to rebuild.")
                return 0

        enqueued: list[tuple[str, str]] = []  # (source_id, job_id)
        skipped = 0
        failed = 0
        for src in targets:
            source_id = src.get("id") or src.get("source_id") or ""
            if not source_id:
                continue
            if args.only_missing and src.get("graph_indexed", False):
                skipped += 1
                continue
            if args.dry_run:
                print(f"  would rebuild: {source_id}")
                enqueued.append((source_id, ""))
                continue
            try:
                ack = await _enqueue_graph(client, base, headers, source_id)
                job_id = ack.get("job_id", "")
                enqueued.append((source_id, job_id))
                print(f"  queued graph job {job_id} for {source_id}")
            except httpx.HTTPError as exc:
                failed += 1
                print(f"  FAILED to enqueue {source_id}: {exc}", file=sys.stderr)

        print(
            f"\n{len(enqueued)} enqueued, {skipped} skipped, {failed} failed "
            f"(of {len(targets)} sources)."
        )
        if args.dry_run:
            return 0

        if args.wait and enqueued:
            job_ids = [jid for _, jid in enqueued if jid]
            print(f"\nWaiting for {len(job_ids)} graph jobs to finish "
                  f"(polling every {args.poll_interval}s)...")
            final = await _poll_until_terminal(
                client, base, headers, job_ids, args.poll_interval
            )
            n_failed = sum(1 for s in final.values() if s != "done")
            print(f"\nDone. {len(final)} jobs terminal, {n_failed} not 'done'.")
            return 2 if n_failed else 0

    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m treeloom.rebuild_graphs",
        description="Bulk-build the Neo4j code graph for indexed sources.",
    )
    parser.add_argument(
        "--indexer-url",
        default=os.environ.get("INDEXER_URL") or DEFAULT_INDEXER_URL,
        help=f"Indexer base URL (default: $INDEXER_URL or {DEFAULT_INDEXER_URL}).",
    )
    from treeloom.infrastructure.cli_auth import add_key_arguments

    add_key_arguments(parser)
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only rebuild sources with graph_indexed=false (two-pass case). "
        "NOT recommended for repairing the empty-graph bug — see module docstring.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Rebuild only this source id (repeatable). Skips listing /sources.",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Poll until all enqueued graph jobs reach a terminal state.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=2.0,
        help="Seconds between status polls when --wait (default: 2.0).",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-request HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be rebuilt without enqueueing anything.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
