"""Benchmark domain: metric computation — pure functions, no I/O."""
import math

import tiktoken

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count cl100k tokens in a string."""
    return len(_tokenizer.encode(text))


def compute_recall(ground_truth: list[str], retrieved: list[str], top_k: int = 5) -> float:
    """Compute recall@k — did any ground truth file appear in top-k?"""
    if not ground_truth:
        return 0.0
    relevant = set(ground_truth[:top_k])
    hit = any(r in relevant for r in retrieved[:top_k])
    return 1.0 if hit else 0.0


def compute_precision(ground_truth: list[str], retrieved: list[str], top_k: int = 5) -> float:
    """Compute precision@k — fraction of top-k that are relevant."""
    if not retrieved[:top_k]:
        return 0.0
    relevant = set(ground_truth)
    hits = sum(1 for r in retrieved[:top_k] if r in relevant)
    return hits / top_k


def compute_mrr(retrieved: list[str], relevant: str) -> float:
    """Mean reciprocal rank — 1/rank of the first relevant result."""
    for i, r in enumerate(retrieved):
        if r == relevant:
            return 1.0 / (i + 1)
    return 0.0


def canonicalize_retrieved(
    retrieved: list[str], alternatives: dict[str, list[str]] | None
) -> list[str]:
    """Map each retrieved path that is an ACCEPTED ALTERNATIVE onto its primary
    ground-truth path, then drop repeats (order-preserving).

    `relevant_file_alternatives` expresses "either of these IS the answer" --
    e.g. a C# partial class split across Foo.cs and Foo.Log.cs is one class.
    `additional_relevant_files` cannot express that: recall here is FRACTIONAL
    (hits / len(relevant)), so listing a sibling there means "find BOTH", which
    halves the score of an agent that found the labelled file and caps
    recall@1 at 0.5.

    Canonicalising the retrieved side instead leaves `relevant` -- and so the
    denominator -- untouched: finding either file scores exactly as finding the
    labelled one did. Dropping repeats keeps a second mention of the same class
    from being counted twice or taking a second rank slot.
    """
    if not alternatives:
        return list(retrieved)
    alias = {alt: primary for primary, alts in alternatives.items() for alt in alts}
    out: list[str] = []
    seen: set[str] = set()
    for path in retrieved:
        canon = alias.get(path, path)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Recall@k — fraction of ground truth found in top-k."""
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved[:k] if r in relevant_set)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Precision@k — fraction of top-k that are relevant."""
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved[:k] if r in relevant_set)
    return hits / k if k > 0 else 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """nDCG@k with binary relevance — the standard ranking-quality metric
    (BEIR / MTEB-reranking / TREC). Unlike recall@k it is *position-discounted*:
    a relevant item at rank 1 scores higher than the same item at rank 3, so it
    sees ranking improvements that don't flip the top-1 hit.

        DCG@k  = Σ_{i=1..k} rel_i / log2(i+1)      (rel_i ∈ {0,1})
        IDCG@k = DCG@k of the ideal ranking (all relevant first)
        nDCG@k = DCG@k / IDCG@k

    Returns 0.0 when nothing is relevant (IDCG == 0).
    """
    relevant_set = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2)
              for i, r in enumerate(retrieved[:k]) if r in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Average Precision@k (the per-query term of MAP) with binary relevance:
    the mean of precision@i evaluated at each rank i ≤ k where a relevant item
    is hit, normalized by the number of relevant items (capped at k). Returns
    0.0 when nothing is relevant. Mean of this across queries = MAP.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant_set:
            hits += 1
            score += hits / i
    return score / min(len(relevant_set), k)
