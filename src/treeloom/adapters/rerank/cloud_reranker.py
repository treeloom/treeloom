"""Hosted reranking API adapter (RERANKER_PROVIDER=voyage|cohere|jina).

The GPU-less quality option: rerank on a cloud API instead of a local GPU
(TEI/`http`) or a CPU cross-encoder (`local`). Same contract as the other
providers — `async rerank_texts(query, texts, batch_size=None) ->
list[tuple[index, raw_score]]` — so it drops into `treeloom.reranker`'s
dispatch and the rest of the pipeline (RERANK_NORMALIZE fusion, the
dead-reranker fallback) is unchanged.

Currently implements **Voyage** (`rerank-2.5`, `rerank-2.5-lite`). Other
providers (Cohere, Jina) are stubbed with a clear error and slot into the
`_PROVIDERS` registry once their request/response mapping is added.

Voyage notes that shape this adapter:
- One request reranks the whole candidate set (up to 1,000 documents), so
  the default pool of 50 is a single round-trip — no client-side batching
  needed, but we chunk defensively at the 1,000-doc cap with global index
  mapping.
- `truncation: true` lets the API clamp oversized docs server-side, so
  (unlike the embeddings path) we don't pre-truncate with tiktoken.
- `relevance_score` is in [0,1]; downstream `graph_rescore` min-max
  normalizes within the candidate set, so the absolute scale is irrelevant
  to ranking (see the affine-invariance test for the local reranker).
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

RERANKER_PROVIDER = os.environ.get("RERANKER_PROVIDER", "http")
RERANK_RETRIES = int(os.environ.get("RERANK_RETRIES", "3"))

def _voyage_body(model: str, query: str, documents: list[str]) -> dict:
    return {
        "model": model,
        "query": query,
        "documents": documents,
        "truncation": True,
        "return_documents": False,
    }


def _cohere_body(model: str, query: str, documents: list[str]) -> dict:
    # Cohere auto-truncates per max_tokens_per_doc (default 4096); no
    # truncation/return_documents fields.
    return {"model": model, "query": query, "documents": documents}


def _zeroentropy_body(model: str, query: str, documents: list[str]) -> dict:
    body = {"model": model, "query": query, "documents": documents}
    # Optional quality/speed mode: RERANKER_LATENCY=fast|slow (omit = API default).
    latency = os.environ.get("RERANKER_LATENCY")
    if latency:
        body["latency"] = latency
    return body


# Per-provider request/response config. `max_docs` is the hard per-request
# document cap (larger candidate sets are chunked with global index mapping);
# `results_key` is the response array name; `build_body` shapes the request.
# Both Voyage and Cohere return items with `index` + `relevance_score`.
_PROVIDERS: dict[str, dict] = {
    "voyage": {
        "url": "https://api.voyageai.com/v1/rerank",
        "default_model": "rerank-2.5",
        "max_docs": 1000,
        "results_key": "data",
        "build_body": _voyage_body,
        "key_env": "VOYAGE_API_KEY",
    },
    "cohere": {
        "url": "https://api.cohere.com/v2/rerank",
        "default_model": "rerank-v3.5",
        "max_docs": 1000,
        "results_key": "results",
        "build_body": _cohere_body,
        "key_env": "COHERE_API_KEY",
    },
    "zeroentropy": {
        "url": "https://api.zeroentropy.dev/v1/models/rerank",
        "default_model": "zerank-2",
        "max_docs": 1000,
        "results_key": "results",
        "build_body": _zeroentropy_body,
        "key_env": "ZEROENTROPY_API_KEY",
    },
}


def _api_key() -> str:
    """API key for the ACTIVE provider.

    Prefers the generic `RERANKER_API_KEY` (explicit override), then the
    provider-specific var for whichever provider RERANKER_PROVIDER selects.
    Critically NOT a fixed precedence across providers: with both
    VOYAGE_API_KEY and COHERE_API_KEY in the env, `RERANKER_PROVIDER=cohere`
    must use the Cohere key, not whichever appears first.
    """
    cfg = _provider_config()
    return os.environ.get("RERANKER_API_KEY") or os.environ.get(
        cfg["key_env"], ""
    )

_client: httpx.AsyncClient | None = None
_client_key: str | None = None


def _provider_config() -> dict:
    cfg = _PROVIDERS.get(RERANKER_PROVIDER)
    if cfg is None:
        raise RuntimeError(
            f"RERANKER_PROVIDER={RERANKER_PROVIDER!r} has no cloud-rerank "
            f"implementation yet (supported: {', '.join(sorted(_PROVIDERS))})."
        )
    return cfg


def _get_client() -> httpx.AsyncClient:
    """Return the shared client, rebuilding it if the API key changed.

    The key was baked into the Authorization header of a module-level
    singleton on first use, so rotating RERANKER_API_KEY (or the
    provider-specific variable) had no effect until the process restarted —
    including after a key was rotated BECAUSE it leaked, which is the case
    where it matters (CWE-672). Tracking the key the live client was built
    with makes rotation take effect on the next call.
    """
    global _client, _client_key
    key_now = _api_key()
    if _client is not None and key_now and key_now != _client_key:
        logger.info("rerank API key changed; rebuilding the client")
        stale, _client = _client, None
        try:
            import asyncio

            asyncio.get_running_loop().create_task(stale.aclose())
        except RuntimeError:
            pass  # no loop to close it on; it will be collected
    if _client is None:
        key = key_now
        if not key:
            cfg = _provider_config()
            raise RuntimeError(
                f"RERANKER_PROVIDER={RERANKER_PROVIDER} needs an API key — set "
                f"RERANKER_API_KEY or the provider-specific {cfg['key_env']}."
            )
        _client_key = key
        _client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {key}"},
        )
    return _client


def _model() -> str:
    cfg = _provider_config()
    # RERANKER_MODEL is shared with the http provider's env; default per
    # provider. rerank-2.5 is the quality model; rerank-2.5-lite is the
    # cheaper/faster option (set RERANKER_MODEL=rerank-2.5-lite).
    return os.environ.get("RERANKER_MODEL") or cfg["default_model"]


async def _rerank_request(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """One Voyage rerank call over <= max_docs documents (local indices)."""
    from treeloom.infrastructure.tracing import get_tracer

    cfg = _provider_config()
    client = _get_client()
    model = _model()
    body = cfg["build_body"](model, query, documents)
    results_key = cfg["results_key"]
    last_exc: Exception | None = None
    with get_tracer("treeloom.reranker").start_as_current_span(
        "rerank.cloud",
        attributes={
            "treeloom.reranker_provider": RERANKER_PROVIDER,
            "treeloom.reranker_model": model,
            "treeloom.doc_count": len(documents),
        },
    ):
        for attempt in range(RERANK_RETRIES):
            try:
                resp = await client.post(cfg["url"], json=body)
                if resp.status_code == 429 or resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                results = resp.json()[results_key]
                return [
                    (int(item["index"]), float(item["relevance_score"]))
                    for item in results
                ]
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status != 429 and status < 500:
                    logger.error(
                        "rerank API %d: %s", status, exc.response.text[:200]
                    )
                    raise
            except httpx.HTTPError as exc:
                last_exc = exc
            await asyncio.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


async def rerank_texts(
    query: str,
    texts: list[str],
    batch_size: int | None = None,
) -> list[tuple[int, float]]:
    """Score each text against the query; returns (index, relevance) pairs.

    Indexes are positions in the input list. Voyage reranks the whole set in
    one call up to its document cap; larger sets are chunked with the local
    indices offset back to global positions.
    """
    if not texts:
        return []
    cap = _provider_config()["max_docs"]
    chunk = min(batch_size or cap, cap)
    out: list[tuple[int, float]] = []
    for start in range(0, len(texts), chunk):
        batch = texts[start : start + chunk]
        for local_idx, score in await _rerank_request(query, batch):
            out.append((start + local_idx, score))
    return out
