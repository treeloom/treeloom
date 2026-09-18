"""Run clean-100 benchmark against a search URL and save results.

Usage:
    python benchmarks/run_clean100.py <search_url> <output_prefix>

Runs 3 identical benchmark iterations and saves:
    benchmarks/results/embedding-ab/<output_prefix>_1.json
    benchmarks/results/embedding-ab/<output_prefix>_2.json
    benchmarks/results/embedding-ab/<output_prefix>_3.json

Flags: use_hyde=False, use_graph_scoring=False, use_summary_vector=False
Path prefix: /home/user/source/featbit
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from treeloom.application.benchmark.runner import run_benchmark

QUERIES_FILE = os.path.join(os.path.dirname(__file__), 'queries/featbit-clean-100.jsonl')
PATH_PREFIX = '/home/user/source/featbit'
FLAGS = {'use_hyde': False, 'use_graph_scoring': False, 'use_summary_vector': False}
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results/embedding-ab')


def load_queries():
    with open(QUERIES_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


async def run_one(search_url, queries):
    return await run_benchmark(
        queries,
        search_url=search_url,
        feature_flags=FLAGS,
        path_prefix=PATH_PREFIX,
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmarks/run_clean100.py <search_url> <output_prefix>")
        sys.exit(1)

    search_url = sys.argv[1]
    tag = sys.argv[2]
    queries = load_queries()
    os.makedirs(OUT_DIR, exist_ok=True)

    prev_agg = None
    for i in range(1, 4):
        print(f"  Run {i}/3...", flush=True)
        result = asyncio.run(run_one(search_url, queries))
        agg = result['aggregate']
        print(f"  recall@1={agg['recall@1']:.4f} recall@5={agg['recall@5']:.4f} mrr={agg['mrr']:.4f}")

        out_path = os.path.join(OUT_DIR, f'clean100_{tag}_{i}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

        if prev_agg is not None:
            if agg != prev_agg:
                print(f"  WARNING: Run {i} differs from run {i-1}!")
                print(f"    prev: {prev_agg}")
                print(f"    curr: {agg}")
        prev_agg = agg

    print(f"\nFinal aggregate for {tag}:")
    print(json.dumps(prev_agg, indent=2))


if __name__ == '__main__':
    main()
