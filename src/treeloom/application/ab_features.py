"""A/B-test retrieval feature flags end-to-end.

For each variant in the config, runs benchmark.py with --tool-args set so that
different retrieval features are enabled/disabled.

Features tested:
    - hybrid_search
    - graph_scoring
    - hyde
    - summary_vector
"""
import argparse
import asyncio
import json
import os
import subprocess


def _slug(feature_set: str) -> str:
    """Create a filesystem-safe slug."""
    return feature_set.lower().replace("-", "_").replace(".", "_")


def _summary_from_results(results: list[dict]) -> str:
    """Generate a summary string from benchmark results."""
    if not results:
        return "No results"

    recalls = [r.get("recall", 0) for r in results]
    precisions = [r.get("precision", 0) for r in results]

    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0

    return f"recall={avg_recall:.4f} precision={avg_precision:.4f} n={len(results)}"


def avg(values: list[float]) -> float:
    """Compute average of a list."""
    return sum(values) / len(values) if values else 0.0


def _run_one_variant(
    variant: str,
    repo: str,
    queries: str,
    output: str,
) -> list[dict]:
    """Run benchmark with a single feature variant."""
    # Parse feature flags from variant string
    # e.g., "hybrid+graph" or "none" or "hyde+summary"

    # Build command args
    args = ["python", "-m", "treeloom.benchmark", "--queries", queries, "--repo", repo]

    # Add feature-specific flags based on variant
    # ... (implementation)

    result = subprocess.run(
        args,
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  Failed variant {variant}: {result.stderr[:200]}")
        return []

    # Parse results from output
    # ... (implementation)
    return []


def _print_comparison(variants: list[tuple[str, list[dict]]]) -> None:
    """Print A/B comparison table."""
    print(f"\n{'='*70}")
    print(f"{'Feature Set':<20} {'Recall':>10} {'Precision':>12} {'N':>5}")
    print(f"{'-'*70}")

    for name, results in variants:
        summary = _summary_from_results(results)
        if results:
            recalls = [r.get("recall", 0) for r in results]
            precisions = [r.get("precision", 0) for r in results]
            avg_recall = avg(recalls)
            avg_precision = avg(precisions)
            print(f"{name:<20} {avg_recall:>10.4f} {avg_precision:>12.4f} {len(results):>5}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="A/B-test retrieval feature flags")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", default="ab_features_results.json")
    parser.add_argument("--variants", nargs="+", default=["all"])
    args = parser.parse_args()

    variants = [
        ("none", []),
        ("hybrid_search", []),
        ("graph_scoring", []),
        ("hyde", []),
        ("summary_vector", []),
        ("hybrid+graph", []),
        ("all", []),
    ]

    # Run each variant
    for name, features in variants:
        print(f"\n  Running variant: {name}")
        results = _run_one_variant(name, args.repo, args.queries, args.output)
        variants = [(n, results if n == name else r) for n, r in variants]

    # Print comparison
    _print_comparison(variants)


if __name__ == "__main__":
    main()
