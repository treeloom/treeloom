"""Benchmark runner — query search API and compute metrics."""
from __future__ import annotations

from treeloom.domain.benchmark.metrics import canonicalize_retrieved
import httpx

from treeloom.domain.benchmark.metrics import average_precision, ndcg_at_k

# Depth for the standard ranking metrics (BEIR/MTEB convention is @10). For a
# true @10 the search must return ≥10 ranked files — run with top_k >= 10; at a
# smaller top_k the pool is shallower and these report at that depth.
_NDCG_K = 10


async def run_benchmark(
    queries: list[dict],
    search_url: str = "http://localhost:8001",
    feature_flags: dict | None = None,
    top_k: int = 5,
    path_prefix: str | None = None,
    cross_repo: bool = False,
) -> dict:
    """Run queries against search API, compute Precision@5, Recall@5, MRR.

    Args:
        queries: List of {"query": str, "ground_truth": [str, ...], "language": str}
        search_url: Base URL of indexer
        feature_flags: e.g., {"use_hyde": True, "use_hybrid": False}
        top_k: Number of results to consider (default 5)
        path_prefix: scope search to one repo (matches the agentic arm's
            scoping). Default None searches the whole multi-repo index, which
            lets cross-repo symbol collisions corrupt name/definition signals.

    Returns:
        {"aggregate": {"precision@5": ..., "recall@5": ..., "mrr": ...},
         "per_query": [...]}
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = []
        for q in queries:
            body = {"query": q["query"], "top_k": top_k}
            if path_prefix:
                body["path_prefix"] = path_prefix
            elif cross_repo:
                body["cross_repo"] = True
            if feature_flags:
                body.update(feature_flags)
            try:
                resp = await client.post(f"{search_url}/search", json=body)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                results.append({"query": q["query"], "error": str(e),
                                "precision@5": 0, "recall@5": 0, "mrr": 0})
                continue

            chunks = data.get("chunks", [])
            # Extract unique file paths from results
            retrieved = []
            seen = set()
            for c in chunks:
                fp = (c.get("entity", {}).get("file_path", "") or
                      c.get("file_path", ""))
                if fp and fp not in seen:
                    retrieved.append(fp)
                    seen.add(fp)
            # Same canonicalisation as agentic scoring, so `run` and `agentic`
            # cannot disagree about whether a query was answered.
            retrieved = canonicalize_retrieved(
                retrieved, q.get("relevant_file_alternatives"))

            # JSONL query files (benchmarks/queries/) carry the ground truth as
            # `relevant_files` (+ optional `additional_relevant_files`); older
            # ad-hoc sets used `ground_truth`. Accept all three.
            gt = set(q.get("relevant_files") or q.get("ground_truth") or [])
            gt |= set(q.get("additional_relevant_files") or [])
            retrieved_set = set(retrieved[:top_k])

            hits = gt & retrieved_set
            precision = len(hits) / top_k if top_k > 0 else 0

            recall_denom = len(gt)
            recall = len(hits) / recall_denom if recall_denom > 0 else 0

            # recall@1: is the top-ranked file relevant? This is the metric the
            # rerank-normalization targets (graph signals reorder rank 1).
            recall_at_1 = 1.0 if retrieved and retrieved[0] in gt else 0.0

            # MRR: rank of first relevant hit
            mrr = 0.0
            for rank, fp in enumerate(retrieved[:top_k], 1):
                if fp in gt:
                    mrr = 1.0 / rank
                    break

            # nDCG@10 + Average Precision (MAP term): position-discounted
            # ranking-quality metrics over the full ranked file list — more
            # sensitive than binary recall@1.
            gt_list = list(gt)
            ndcg = ndcg_at_k(retrieved, gt_list, _NDCG_K)
            ap = average_precision(retrieved, gt_list, _NDCG_K)

            results.append({
                "query": q["query"],
                "ground_truth": list(gt),
                "retrieved": retrieved[:top_k],
                "recall@1": recall_at_1,
                "precision@5": precision,
                "recall@5": recall,
                "mrr": mrr,
                "ndcg@10": ndcg,
                "ap": ap,
            })

    n = len(results)
    if n == 0:
        return {"aggregate": {"recall@1": 0, "precision@5": 0, "recall@5": 0,
                              "mrr": 0, "ndcg@10": 0, "map": 0},
                "per_query": []}

    scored = [r for r in results if "error" not in r]
    m = len(scored) or 1
    agg = {
        "recall@1": sum(r["recall@1"] for r in scored) / m,
        "precision@5": sum(r["precision@5"] for r in scored) / m,
        "recall@5": sum(r["recall@5"] for r in scored) / m,
        "mrr": sum(r["mrr"] for r in scored) / m,
        "ndcg@10": sum(r["ndcg@10"] for r in scored) / m,
        # MAP = mean of per-query Average Precision.
        "map": sum(r["ap"] for r in scored) / m,
        "total_queries": n,
    }
    return {"aggregate": agg, "per_query": results}
