"""In-process CPU cross-encoder reranker (RERANKER_PROVIDER=local).

Runs fastembed's ONNX `TextCrossEncoder` on CPU — no service, no GPU, no
API key. The simple-profile reranker.

Models (fastembed-supported; featbit-clean-100 A/B 2026-06-11, deterministic
flags, vs same-day GPU bge-v2-m3 control at r@1 0.63 / r@5 0.80 / MRR 0.698):
- `jinaai/jina-reranker-v1-turbo-en` (default, ~150MB): 0.58/0.74/0.645 at
  ~2.3 s/query — dominates both MiniLM (0.55/0.70/0.599 @ 1.9s) and
  bge-reranker-base (0.58/0.76/0.637 @ 8.8s).
- `jinaai/jina-reranker-v2-base-multilingual` (~1.1GB): the quality option —
  0.60/0.84/0.689, a statistical TIE with the GPU control (won 15 / lost 14
  per query), but ~16 s/query on CPU.

Scores are raw cross-encoder logits — same semantics as the HTTP path's
`raw_scores: True`, comparable across batches. Downstream graph-signal
fusion min-max normalizes within each candidate set (RERANK_NORMALIZE),
so the absolute logit scale of the chosen model cannot break the
GRAPH_ALPHA..DELTA weights: ranking is invariant under affine score
transforms.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

logger = logging.getLogger(__name__)

RERANKER_LOCAL_MODEL = os.environ.get(
    "RERANKER_LOCAL_MODEL", "jinaai/jina-reranker-v1-turbo-en"
)
RERANK_LOCAL_BATCH_SIZE = int(os.environ.get("RERANK_LOCAL_BATCH_SIZE", "16"))

_encoder = None
_lock = threading.Lock()


def _get_encoder():
    """Lazy singleton — the first call downloads the ONNX model to the HF cache."""
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.warning(
                    "loading local reranker %s — the first run downloads the "
                    "model (~150MB default, ~1.1GB for the quality option) "
                    "and may take a while; subsequent calls are fast",
                    RERANKER_LOCAL_MODEL,
                )
                _encoder = TextCrossEncoder(model_name=RERANKER_LOCAL_MODEL)
    return _encoder


def _rerank_sync(query: str, texts: list[str], batch_size: int) -> list[float]:
    encoder = _get_encoder()
    return [float(s) for s in encoder.rerank(query, texts, batch_size=batch_size)]


async def rerank_texts(
    query: str,
    texts: list[str],
    batch_size: int | None = None,
) -> list[tuple[int, float]]:
    """Score each text against the query; returns (index, raw_logit) pairs.

    Same contract as http_reranker.rerank_texts. Inference is synchronous
    ONNX, so it runs in a worker thread to keep the event loop responsive.
    """
    if not texts:
        return []
    bs = batch_size or RERANK_LOCAL_BATCH_SIZE
    scores = await asyncio.to_thread(_rerank_sync, query, texts, bs)
    return list(enumerate(scores))
