"""Standalone reranker service — TEI-compatible /rerank for models TEI can't host.

TEI (text-embeddings-inference) refuses jina-reranker-v2-base-multilingual: it
uses custom modeling code (no standard `model_type`), so the normal treeloom
reranker path (TEI on :8081) can't serve it. This wraps the model with
`transformers` + `trust_remote_code` and exposes the exact endpoint
`retriever.rerank` calls:

    POST /rerank  {"query": str, "texts": [str, ...], "raw_scores": bool}
       ->          [{"index": int, "score": float}, ...]   (one per input text)

Point `RERANKER_URL` at this service (default port 8085) instead of TEI to A/B
a different reranker without changing any treeloom code.

Scores come from the model's own `compute_score` (sigmoid, [0,1]). That's fine
for treeloom: `rerank()` only sorts by score, and `graph_rescore` min-max
normalizes within the candidate set (RERANK_NORMALIZE), so bounded sigmoid
scores are if anything more cross-batch-comparable than raw logits. The
`raw_scores` request field is accepted and ignored.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jina_reranker")

MODEL_ID = os.environ.get("MODEL_ID", "jinaai/jina-reranker-v2-base-multilingual")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1024"))
DEVICE = os.environ.get("RERANK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def _load_model():
    from transformers import AutoModelForSequenceClassification

    logger.info("loading reranker %s on %s", MODEL_ID, DEVICE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.eval()
    logger.info("reranker ready: %s", MODEL_ID)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
    yield
    _model = None


app = FastAPI(title="jina-reranker", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    # Accepted for TEI wire-compatibility; ignored (we return model scores).
    raw_scores: bool = True
    truncate: bool | None = None
    truncation_direction: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok" if _model is not None else "loading", "model_id": MODEL_ID}


@app.get("/info")
async def info():
    return {"model_id": MODEL_ID, "max_input_length": MAX_LENGTH}


@app.post("/rerank")
async def rerank(req: RerankRequest):
    if not req.texts:
        return []
    pairs = [[req.query, t] for t in req.texts]
    with torch.no_grad():
        scores = _model.compute_score(pairs, max_length=MAX_LENGTH)
    # compute_score returns a bare float for a single pair, a list otherwise.
    if not isinstance(scores, (list, tuple)):
        scores = [scores]
    return [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
