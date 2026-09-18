"""In-process sentence-transformers CrossEncoder reranker (RERANKER_PROVIDER=local-st).

For HF cross-encoders that fastembed's ONNX runtime can't load — notably
`zeroentropy/zerank-*` (sentence-transformers `CrossEncoder` architecture,
zerank-1 is Qwen3-4B-based). Self-hosted: weights from HF, GPU if available
else CPU, no API key.

Same `rerank_texts(query, texts, batch_size=None) -> list[tuple[index,
raw_score]]` contract as the other providers. CrossEncoder scores are raw
(model-dependent range); downstream `graph_rescore` min-max normalizes within
the candidate set (RERANK_NORMALIZE), so absolute scale is irrelevant to
ranking.

Caveats:
- zerank-1 / zerank-2 are **non-commercial licensed** on HF (eval only);
  zerank-1-small is Apache-2.0. Production commercial use → the hosted
  `zeroentropy` cloud provider.
- 4B-class models need ~8GB VRAM; on a GPU already loaded with other services
  they may not fit (use the cloud provider, or a smaller model / CPU).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

logger = logging.getLogger(__name__)

# Reuses RERANKER_LOCAL_MODEL (the local reranker model id) — default is the
# Apache-2.0 small model so the out-of-box path is license-clean.
RERANKER_LOCAL_MODEL = os.environ.get(
    "RERANKER_LOCAL_MODEL", "zeroentropy/zerank-1-small"
)
RERANK_LOCAL_BATCH_SIZE = int(os.environ.get("RERANK_LOCAL_BATCH_SIZE", "32"))
# auto → cuda if available else cpu; override with RERANKER_ST_DEVICE=cpu|cuda
RERANKER_ST_DEVICE = os.environ.get("RERANKER_ST_DEVICE", "")

_encoder = None
_lock = threading.Lock()


def _device() -> str:
    if RERANKER_ST_DEVICE:
        return RERANKER_ST_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_encoder():
    """Lazy singleton — first call downloads the HF model and loads it."""
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                from sentence_transformers import CrossEncoder

                dev = _device()
                logger.warning(
                    "loading sentence-transformers reranker %s on %s — the "
                    "first run downloads the model (can be several GB for "
                    "zerank-1/2) and may take a while; subsequent calls reuse it",
                    RERANKER_LOCAL_MODEL, dev,
                )
                # Force safetensors-only loading. CrossEncoder ultimately calls
                # transformers from_pretrained → torch.load(), whose pickle path
                # is an arbitrary-code-execution primitive on a poisoned
                # pytorch_model.bin. Since RERANKER_LOCAL_MODEL can be set from
                # untrusted deployment config (CI vars, ConfigMaps), use_safetensors
                # makes loading fail closed on a pickle-only repo instead of
                # deserialising it (CWE-502).
                _encoder = CrossEncoder(
                    RERANKER_LOCAL_MODEL,
                    device=dev,
                    model_kwargs={"use_safetensors": True},
                )
    return _encoder


def _rerank_sync(query: str, texts: list[str], batch_size: int) -> list[float]:
    encoder = _get_encoder()
    scores = encoder.predict(
        [(query, t) for t in texts], batch_size=batch_size
    )
    return [float(s) for s in scores]


async def rerank_texts(
    query: str,
    texts: list[str],
    batch_size: int | None = None,
) -> list[tuple[int, float]]:
    """Score each text against the query; returns (index, raw_score) pairs.

    Synchronous torch inference runs in a worker thread to keep the event
    loop responsive.
    """
    if not texts:
        return []
    bs = batch_size or RERANK_LOCAL_BATCH_SIZE
    scores = await asyncio.to_thread(_rerank_sync, query, texts, bs)
    return list(enumerate(scores))
