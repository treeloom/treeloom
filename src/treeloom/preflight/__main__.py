"""CLI entry: python -m treeloom.preflight <path> [options].

Exit codes:
  0 — green (no warnings, est < 6h)
  1 — yellow (medium warnings, or 6–24h)
  2 — red (high warnings, or >24h)

Useful for CI gating: a new repo added to a benchmark set fails the
build if preflight predicts it would blow up the indexer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from treeloom.preflight.coefficients import (
    fallback_coefficients,
    load_coefficients,
    write_coefficients,
)
from treeloom.preflight.model import estimate_total
from treeloom.preflight.recommender import PreflightReport, build_report
from treeloom.preflight.scanner import walk_repo


def _print_human(report: PreflightReport) -> None:
    print(f"\nPREFLIGHT  {report.path}")
    print(f"  verdict: {report.verdict.upper()}")
    files = report.files_summary
    print(f"  files: {files['total_supported']} supported, {files['total_unsupported']} skipped")
    print(f"  size: {files['size_total_bytes'] / 1024 / 1024:.1f} MB total")
    if files["max_file_bytes"]:
        print(f"  largest: {files['max_file_bytes'] / 1024 / 1024:.1f} MB")
    if files["by_language"]:
        top_langs = sorted(files["by_language"].items(), key=lambda x: -x[1])[:5]
        print(f"  languages: " + ", ".join(f"{l} ({n})" for l, n in top_langs))

    est_on = report.estimates["summary_on"]
    est_off = report.estimates["summary_off"]
    print(f"\n  estimate (summary on): {est_on['wall_time_s']/3600:.1f}h "
          f"@ {est_on['files_per_min']:.1f} files/min")
    print(f"  estimate (summary off): {est_off['wall_time_s']/3600:.1f}h "
          f"@ {est_off['files_per_min']:.1f} files/min")

    if files["hot_files"]:
        print("\n  hot files (top by predicted time):")
        for f in files["hot_files"][:5]:
            print(f"    {f['size_bytes']/1024/1024:>5.1f} MB  "
                  f"{f['estimated_seconds_summary_on']:>6.0f}s  {f['path']}")

    if report.warnings:
        print("\n  warnings:")
        for w in report.warnings:
            tag = {"high": "!!", "medium": "! ", "low": "  "}[w.severity]
            print(f"    {tag} [{w.code}] {w.message}")

    rec = report.recommendations
    if rec.env:
        print("\n  recommended env:")
        for k, v in rec.env.items():
            print(f"    {k}={v}")
    if rec.skip_patterns:
        print("\n  recommended skip_patterns:")
        for p in rec.skip_patterns:
            print(f"    {p}")


async def _scan_and_report(path: str, *, concurrency: int) -> PreflightReport:
    stats, unsupported = walk_repo(path)
    try:
        coefs = await load_coefficients()
    except Exception:
        coefs = fallback_coefficients()
    per_file, total = estimate_total(stats, coefs, file_concurrency=concurrency)
    return build_report(path, stats, unsupported, per_file, total)


async def _run_calibrate(source_id: str) -> int:
    """Update coefficients from an observed completed job."""
    from treeloom.adapters.postgresql.connection import init_pool, get_pool

    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is required for --calibrate.", file=sys.stderr)
        return 2
    await init_pool(os.environ["DATABASE_URL"])
    pool = await get_pool()
    if pool is None:
        print("Could not connect to Postgres.", file=sys.stderr)
        return 2

    async with pool.acquire() as conn:
        job = await conn.fetchrow(
            """
            SELECT id, source, source_path, source_url, processed_files, total_files,
                   total_chunks, start_time, finished_at, errors
              FROM jobs
             WHERE source_id = $1 AND status = 'done'
          ORDER BY finished_at DESC NULLS LAST LIMIT 1
            """,
            source_id,
        )
    if job is None:
        print(f"No completed job for source_id={source_id!r}", file=sys.stderr)
        return 2
    duration = float(job["finished_at"] or 0) - float(job["start_time"] or 0)
    if duration <= 0:
        print("Job duration is non-positive — cannot calibrate.", file=sys.stderr)
        return 2

    target_path = job["source_path"] or job["source"]
    if not target_path or not os.path.isdir(target_path):
        print(f"Source path {target_path!r} not accessible; can't re-scan to calibrate.",
              file=sys.stderr)
        return 2

    stats, _ = walk_repo(target_path)
    coefs = await load_coefficients()
    _, predicted = estimate_total(stats, coefs, file_concurrency=int(
        os.environ.get("INDEX_FILE_CONCURRENCY", "8")
    ))
    predicted_total = predicted.wall_seconds_summary_on  # assume summaries were on

    # Solve a single global scale factor against the observation. This is
    # less granular than per-stage calibration but adequate for v1.
    if predicted_total <= 0:
        print("Predicted total is non-positive; nothing to scale.", file=sys.stderr)
        return 2
    scale = duration / predicted_total

    # Apply scale to the dominant stage coefficients
    updates: dict[tuple[str, str], float] = {
        ("parse_s_per_mb", ""):         coefs.get("parse_s_per_mb") * scale,
        ("embed_s_per_chunk", ""):      coefs.get("embed_s_per_chunk") * scale,
        ("llm_s_per_chunk", ""):        coefs.get("llm_s_per_chunk") * scale,
        ("graph_s_per_file_const", ""): coefs.get("graph_s_per_file_const") * scale,
        ("graph_s_per_entity", ""):     coefs.get("graph_s_per_entity") * scale,
    }
    written = await write_coefficients(updates, source_id=source_id)
    print(f"Calibrated against source_id={source_id}: "
          f"scale={scale:.2f}, {written} coefficient(s) updated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m treeloom.preflight")
    p.add_argument("path", nargs="?", help="Local directory to scan")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Estimate at this INDEX_FILE_CONCURRENCY (default: env or 8)")
    p.add_argument("--calibrate", metavar="SOURCE_ID",
                   help="Recompute coefficients from a completed job")
    args = p.parse_args(argv)

    concurrency = args.concurrency or int(os.environ.get("INDEX_FILE_CONCURRENCY", "8"))

    if args.calibrate:
        return asyncio.run(_run_calibrate(args.calibrate))

    if not args.path:
        p.error("path is required when --calibrate is not used")

    # init pool best-effort so DB-backed coefficients work
    async def _go():
        from treeloom.adapters.postgresql.connection import init_pool
        if os.environ.get("DATABASE_URL"):
            try:
                await init_pool(os.environ["DATABASE_URL"])
            except Exception:
                pass
        return await _scan_and_report(args.path, concurrency=concurrency)
    report = asyncio.run(_go())

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report)

    return {"green": 0, "yellow": 1, "red": 2}[report.verdict]


if __name__ == "__main__":
    sys.exit(main())
