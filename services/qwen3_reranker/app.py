"""Standalone reranker service — TEI-compatible /rerank for Qwen3-Reranker models.

TEI cannot host Qwen3 rerankers: cuda-1.9 fails with `'classifier' model type
is not supported for Qwen3` (verified 2026-06-10) on both the official
Qwen/Qwen3-Reranker-* models (causal LM, no pooling config) and the
sequence-classification ports. This service wraps the seq-cls port with
`transformers` and exposes the exact endpoint `retriever.rerank` calls:

    POST /rerank  {"query": str, "texts": [str, ...], "raw_scores": bool}
       ->          [{"index": int, "score": float}, ...]   (one per input text)

Point `RERANKER_URL` at this service (default port 8086) instead of TEI.

Qwen3-Reranker scores garbage without its exact prompt template (system
prefix + <Instruct>/<Query>/<Document> + assistant suffix) — see PREFIX /
SUFFIX below; they must not be altered. Scores are the raw classifier logits:
monotone per pair, cross-batch comparable, and treeloom only sorts by score
(graph rescoring min-max normalizes within the candidate set).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import torch
import torch._dynamo
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3_reranker")

MODEL_ID = os.environ.get("MODEL_ID", "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1600"))
BATCH_SIZE = int(os.environ.get("RERANK_INFER_BATCH", "16"))
INSTRUCTION = os.environ.get(
    "RERANK_INSTRUCTION",
    "Given a code search query, retrieve relevant code snippets",
)
DEVICE = os.environ.get("RERANK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# The reranker forward pass is GPU-COMPUTE-bound (measured: latency scales with
# total tokens, not doc count; a single 0.6B model already saturates the SMs and
# replicas don't add throughput). Both knobs below attack that compute directly:
#   - RERANK_BUCKET: sort the candidate set by token length before batching so a
#     batch of short chunks isn't padded up to one long chunk's length. With
#     padding=True the per-batch cost is (batch_size x longest_doc), so length
#     bucketing eliminates cross-padding FLOP waste. Scores are mapped back to
#     the caller's original order.
#   - RERANK_COMPILE: torch.compile(dynamic=True) the forward; fuses kernels and
#     raises FLOP efficiency on the same GPU. dynamic=True avoids a recompile per
#     padded shape (bucketing produces many distinct lengths).
#   - RERANK_ATTN: attn_implementation (sdpa = PyTorch scaled_dot_product_attention,
#     faster than HF eager). Falls back to eager if the model rejects it.
RERANK_BUCKET = os.environ.get("RERANK_BUCKET", "1") == "1"
# torch.compile is DEFAULT-OFF: it gave +27% on fixed shapes but RECOMPILES on
# every new padded shape, and real code chunks have continuously varied lengths
# (bucketing makes this worse) — measured 34s recompile spikes on live /search,
# a net loss. Enable only with fixed-bucket sequence padding (round seq length
# up to a small fixed set) so inductor compiles a handful of graphs and stops.
RERANK_COMPILE = os.environ.get("RERANK_COMPILE", "0") == "1"
RERANK_ATTN = os.environ.get("RERANK_ATTN", "sdpa")

# Required Qwen3-Reranker template (matches the official model's chat format;
# the seq-cls port was verified score-identical to the original with it).
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_model = None
_tokenizer = None


def _load():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("loading reranker %s on %s", MODEL_ID, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    kwargs = {"dtype": torch.float16 if DEVICE == "cuda" else torch.float32}
    if RERANK_ATTN:
        kwargs["attn_implementation"] = RERANK_ATTN
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **kwargs)
    except (ValueError, ImportError) as e:
        # Model/build rejects the requested attention impl — fall back to eager.
        logger.warning("attn_implementation=%s failed (%s); using default", RERANK_ATTN, e)
        kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **kwargs)
    model.to(DEVICE)
    model.eval()
    if RERANK_COMPILE and DEVICE == "cuda":
        try:
            # torch.compile is lazy — failures (e.g. no C compiler for triton)
            # surface on the FIRST forward, not here. suppress_errors makes that
            # first compiled call fall back to eager + warn instead of 500-ing
            # the request; build-essential in the Dockerfile is what makes the
            # compile actually succeed.
            torch._dynamo.config.suppress_errors = True
            # dynamic=True: one compiled graph across the variable padded shapes
            # that bucketing produces (else inductor recompiles per shape).
            model = torch.compile(model, dynamic=True)
            logger.info("torch.compile enabled (dynamic=True)")
        except Exception as e:  # inductor/triton unavailable -> run eager
            logger.warning("torch.compile failed (%s); running eager", e)
    logger.info(
        "reranker ready: %s (attn=%s compile=%s bucket=%s)",
        MODEL_ID, kwargs.get("attn_implementation", "default"),
        RERANK_COMPILE, RERANK_BUCKET,
    )
    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    _model, _tokenizer = _load()
    yield
    _model = None
    _tokenizer = None


app = FastAPI(title="qwen3-reranker", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    # Accepted for TEI wire-compatibility; we always return raw logits.
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
    inputs = [
        f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {req.query}\n<Document>: {t}{SUFFIX}"
        for t in req.texts
    ]
    # Tokenize once without padding to get per-doc lengths and reusable token
    # ids; pad per-batch below so each batch only pays for its own longest doc.
    enc = _tokenizer(inputs, truncation=True, max_length=MAX_LENGTH)
    n = len(inputs)
    # Length bucketing: batch similar-length docs together so a batch of short
    # chunks isn't padded up to a long chunk elsewhere in the pool. Returns
    # scores in the caller's original order regardless.
    if RERANK_BUCKET:
        order = sorted(range(n), key=lambda i: len(enc["input_ids"][i]))
    else:
        order = list(range(n))

    scores = [0.0] * n
    with torch.no_grad():
        for b in range(0, n, BATCH_SIZE):
            idxs = order[b : b + BATCH_SIZE]
            features = {k: [enc[k][i] for i in idxs] for k in enc}
            batch = _tokenizer.pad(features, padding=True, return_tensors="pt").to(DEVICE)
            logits = _model(**batch).logits.squeeze(-1)
            vals = logits.float().tolist()
            if not isinstance(vals, list):  # single-element batch -> scalar
                vals = [vals]
            for j, i in enumerate(idxs):
                scores[i] = vals[j]
    return [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
