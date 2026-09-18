"""Build an answerable, deduplicated benchmark query set from an enhanced query file.

Why: per-query failure analysis (2026-06-10) showed the old 50-query featbit sample's
0.26 recall@5 gap was a benchmark artifact — template queries whose entity name is
non-discriminating ("constructor", "Handle") cannot identify their single ground-truth
file, and 5/50 rows shared identical query text with different ground truths.
Filter validation on that sample: predicted-answerable queries scored 0.919 recall@5,
predicted-unanswerable 0.231.

A row is kept ("answerable") when the query mentions its entity name (verbatim or all
camelCase tokens) AND either the name is discriminating (maps to <=3 distinct files
across the whole query corpus) or the query carries a path/namespace hint.

Usage:
    python benchmarks/make_clean_queries.py \
        --in <raw-generated-queries>.jsonl \
        --out benchmarks/queries/featbit-clean-100.jsonl
    # (originally run against the since-removed featbit-enhanced.jsonl)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from treeloom.domain.benchmark.query_hygiene import filter_answerable

# Featbit-specific stop path parts (repo name, machine path components, etc.)
_FEATBIT_STOP_PARTS = {"home", "user", "source", "featbit"}

QUOTA = {"base": 15, "namespaced": 20, "entity_context": 15, "intent": 20,
         "problem_driven": 15, "cross_cutting": 10, "docstring": 5}
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]

    candidates = filter_answerable(rows, stop_parts=_FEATBIT_STOP_PARTS)
    random.seed(args.seed)
    random.shuffle(candidates)

    quota = dict(QUOTA)
    picked, seen_texts = [], set()
    for r in candidates:
        s = r.get("strategy") or "base"
        if quota.get(s, 0) <= 0:
            continue
        t = r["query"].strip().lower()
        if t in seen_texts:
            continue
        seen_texts.add(t)
        quota[s] -= 1
        picked.append(r)
        if not any(quota.values()):
            break

    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    by_strat = collections.Counter((r.get("strategy") or "base") for r in picked)
    print(f"wrote {len(picked)} rows to {args.out}: {dict(by_strat)}")


if __name__ == "__main__":
    main()
