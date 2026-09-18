"""HTTP reranker speaking the TEI `/rerank` protocol (RERANKER_URL).

The default RERANKER_PROVIDER. Works with TEI (bge-reranker), the bundled
services/qwen3_reranker, and anything else that implements
POST /rerank {"query", "texts", "raw_scores"} -> [{"index", "score"}].
"""
import asyncio
import os

import httpx

# Not require_env at import: which reranker vars are mandatory depends on
# RERANKER_PROVIDER, enforced by infrastructure.config.validate_config() at
# startup. A missing URL here still fails loudly at first call.
RERANKER_URL = os.environ.get("RERANKER_URL", "")
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "20"))

# A single keep-alive client reused across requests. A fresh AsyncClient per
# call paid a new TCP (and TLS) handshake on every search. Created lazily on
# first use so it binds to the running event loop (the indexer's uvicorn loop).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def _score_batch(
    client: httpx.AsyncClient, query: str, batch: list[str], offset: int
) -> list[tuple[int, float]]:
    """POST one batch to /rerank; map its local indexes back to global ones."""
    resp = await client.post(
        f"{RERANKER_URL}/rerank",
        json={"query": query, "texts": batch, "raw_scores": True},
    )
    resp.raise_for_status()
    return [(offset + item["index"], float(item["score"])) for item in resp.json()]


async def rerank_texts(
    query: str,
    texts: list[str],
    batch_size: int | None = None,
) -> list[tuple[int, float]]:
    """Score each text against the query; returns (index, score) pairs.

    Indexes are positions in the input list. Batches at RERANK_BATCH_SIZE
    (TEI 422s above ~20) with raw_scores=True so scores are comparable across
    batches. The batches are independent, so they're dispatched CONCURRENTLY
    (asyncio.gather) rather than in a serial loop — on a 50-candidate pool that
    collapses 3 sequential round trips into ~1's wall time. Span tracing showed
    serial rerank batches were ~74% of search latency.
    """
    if not texts:
        return []
    if not RERANKER_URL:
        raise RuntimeError(
            "RERANKER_PROVIDER=http but RERANKER_URL is not set. "
            "Set it in .env or switch RERANKER_PROVIDER=local."
        )
    bs = batch_size or RERANK_BATCH_SIZE
    client = _get_client()
    batches = await asyncio.gather(
        *(
            _score_batch(client, query, texts[i : i + bs], i)
            for i in range(0, len(texts), bs)
        )
    )
    out: list[tuple[int, float]] = []
    for b in batches:
        out.extend(b)
    return out
