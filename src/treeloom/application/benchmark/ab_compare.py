"""AB comparison — run same queries with different feature flags."""
from __future__ import annotations

from treeloom.application.benchmark.runner import run_benchmark


async def ab_compare(
    queries: list[dict],
    search_url: str,
    configs: list[dict],
    top_k: int = 5,
    path_prefix: str | None = None,
    cross_repo: bool = False,
) -> dict:
    """Run the same queries against multiple feature flag configs.

    Args:
        queries: List of query dicts with ground_truth
        search_url: Base URL of indexer
        configs: [{"name": "baseline", "flags": {}}, ...]
        top_k: K for Precision@K

    Returns:
        {"configs": [{"name": ..., "aggregate": {...}, "per_query": [...]}, ...],
         "comparison": text table}
    """
    results = []
    for cfg in configs:
        result = await run_benchmark(
            queries, search_url, feature_flags=cfg["flags"], top_k=top_k,
            path_prefix=path_prefix, cross_repo=cross_repo,
        )
        results.append({
            "name": cfg["name"],
            "flags": cfg["flags"],
            "aggregate": result["aggregate"],
            "per_query": result["per_query"],
        })

    # Build comparison table
    header = f"{'Config':<25} {'P@5':>6} {'R@5':>6} {'MRR':>6}"
    sep = "-" * len(header)
    rows = [header, sep]
    baseline_p5 = results[0]["aggregate"]["precision@5"] if results else 0
    for r in results:
        agg = r["aggregate"]
        p5 = agg["precision@5"]
        r5 = agg["recall@5"]
        mrr = agg["mrr"]
        delta = ""
        if p5 > baseline_p5 + 0.001:
            delta = f" (+{p5 - baseline_p5:+.3f})"
        elif p5 < baseline_p5 - 0.001:
            delta = f" ({p5 - baseline_p5:+.3f})"
        rows.append(f"{r['name']:<25} {p5:6.3f} {r5:6.3f} {mrr:6.3f}{delta}")

    return {
        "configs": results,
        "comparison": "\n".join(rows),
    }
