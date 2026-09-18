#!/usr/bin/env python3
"""Assert a 3-query agentic smoke run is sane (internal-reranker-3 Phase 4).

Given a per-query rows JSONL + its _summary.json (as produced by
`python -m treeloom.benchmark agentic`), asserts per model:

  1. Termination: `hit_cap` false on >= 2/3 of rows. (`hit_cap` is stored
     per-row by agentic_runner._arm_row — verified present in current and
     incident-era row files alike.)
  2. Grounding: every row with recall@1 == 1.0 cites a retrieved file
     basename in final_answer (the Phase-0 grounding check).
  3. Non-trivial score: mean judge correctness >= 3.0.
  4. Summary metadata: non-empty `served_models` and
     `agent_protocol` == "native_tools".

Prints a PASS/FAIL line and exits non-zero on any failure. A failure means
alias drift or protocol rot — stop, do not launch the paid cell.

Usage:
    python scripts/agentic_smoke_check.py ROWS.jsonl [--summary SUMMARY.json]
        [--arm treeloom] [--model LABEL]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

MIN_MEAN_CORRECTNESS = 3.0


def check_run(rows: list[dict], summary: dict, arm: str = "treeloom") -> list[str]:
    """Pure check: return a list of failure messages (empty = PASS)."""
    failures: list[str] = []
    if not rows:
        return ["no rows found"]

    arm_rows: list[tuple[str, dict]] = []
    for r in rows:
        a = (r.get("arms") or {}).get(arm)
        if a is None:
            failures.append(f"row {r.get('id')!r}: missing arm {arm!r}")
        else:
            arm_rows.append((str(r.get("id")), a))
    if not arm_rows:
        return failures

    # 1. Termination: hit_cap false on >= ceil(2/3 * n) rows.
    non_cap = sum(1 for _, a in arm_rows if not a.get("hit_cap", False))
    need = math.ceil(2 * len(arm_rows) / 3)
    if non_cap < need:
        failures.append(
            f"termination: hit_cap false on only {non_cap}/{len(arm_rows)} "
            f"rows (need >= {need})"
        )

    # 2. Grounding: recall@1 == 1.0 rows must cite a retrieved basename.
    for qid, a in arm_rows:
        if a.get("recall@1") == 1.0:
            names = [os.path.basename(p) for p in (a.get("retrieved_files") or [])[:5]]
            final = a.get("final_answer") or ""
            if not any(n and n in final for n in names):
                failures.append(
                    f"grounding: {qid} UNGROUNDED — recall@1=1.0 but none of "
                    f"{names} appears in final_answer"
                )

    # 3. Mean correctness across the trio.
    corrs = [
        (a.get("judge") or {}).get("correctness")
        for _, a in arm_rows
    ]
    corrs = [c for c in corrs if isinstance(c, (int, float))]
    mean_corr = sum(corrs) / len(corrs) if corrs else 0.0
    if mean_corr < MIN_MEAN_CORRECTNESS:
        failures.append(
            f"correctness: mean {mean_corr:.2f} < {MIN_MEAN_CORRECTNESS} "
            f"(per-row: {corrs})"
        )

    # 4. Summary metadata: served_models recorded, native protocol confirmed.
    served = summary.get("served_models") or []
    if not served:
        failures.append(
            "summary: served_models missing/empty — vendor-served model was "
            "not recorded (pre-rebuild harness or instrumentation regression)"
        )
    proto = summary.get("agent_protocol")
    if proto != "native_tools":
        failures.append(f"summary: agent_protocol={proto!r} != 'native_tools'")

    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rows", help="Per-query rows .jsonl from an agentic run")
    ap.add_argument("--summary", default=None,
                    help="Matching _summary.json (default: derived from rows path)")
    ap.add_argument("--arm", default="treeloom")
    ap.add_argument("--model", default=None,
                    help="Label for the PASS/FAIL line (default: summary's model)")
    args = ap.parse_args(argv)

    summary_path = args.summary or args.rows.removesuffix(".jsonl") + "_summary.json"
    with open(args.rows) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except FileNotFoundError:
        summary = {}

    label = args.model or summary.get("model") or args.rows
    served = summary.get("served_models") or []
    failures = check_run(rows, summary, arm=args.arm)
    if failures:
        print(f"FAIL {label} (rows={args.rows})")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print(f"PASS {label} (served={','.join(served)}, n={len(rows)}, "
          f"rows={args.rows})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
