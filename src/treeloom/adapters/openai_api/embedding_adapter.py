"""OpenAI-compatible embedding adapter (EMBEDDING_PROVIDER=openai).

POSTs to `{EMBEDDING_BASE_URL}/embeddings` with Bearer auth — works against
api.openai.com and any compatible endpoint (OpenRouter, TEI's /v1, vLLM).
Exposes the same `embed` / `embed_query` / `MAX_BATCH_SIZE` surface as the
TEI proxy so `treeloom.embedder` can dispatch between them; the Postgres
`embedding_backends` registry is intentionally bypassed (it models TEI
fleets: NOTIFY/LISTEN reload, GPU/CPU classes, the bare `/embed` protocol).

Unlike TEI, OpenAI hard-fails inputs over 8,192 tokens and large requests,
so each input is truncated with tiktoken and batches are token-budgeted.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
).rstrip("/")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# EMBEDDING_API_KEY wins so the LLM and embeddings can use different keys;
# falls back to OPENAI_API_KEY (the simple-profile single key).
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY") or os.environ.get(
    "OPENAI_API_KEY", ""
)
MAX_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "128"))
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "4"))
# OpenAI rejects inputs > 8,192 tokens; leave headroom under the limit.
EMBED_MAX_TOKENS_PER_INPUT = int(os.environ.get("EMBED_MAX_TOKENS_PER_INPUT", "8000"))
# Stay under the ~300k tokens/request ceiling with margin to spare.
EMBED_MAX_TOKENS_PER_REQUEST = int(
    os.environ.get("EMBED_MAX_TOKENS_PER_REQUEST", "120000")
)
EMBED_RETRIES = int(os.environ.get("EMBED_RETRIES", "3"))
EMBEDDING_QUERY_PREFIX = os.environ.get("EMBEDDING_QUERY_PREFIX", "")

_client: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None
_encoder = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        if not EMBEDDING_API_KEY:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai needs EMBEDDING_API_KEY or "
                "OPENAI_API_KEY set."
            )
        _client = httpx.AsyncClient(
            timeout=120.0,
            headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        )
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)
    return _semaphore


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken

        try:
            _encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            # Non-OpenAI model name on a compatible endpoint — cl100k_base is
            # only used for truncation/budgeting, not exact tokenization.
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _truncate(text: str) -> tuple[str, int]:
    """Clamp one input under the per-input token limit; returns (text, tokens).

    TEI degraded gracefully on oversized chunks — OpenAI 400s the whole
    batch instead, so a single huge chunk must not poison its batch.
    """
    if not text:
        # The API rejects empty strings; a single space embeds fine.
        return " ", 1
    enc = _get_encoder()
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= EMBED_MAX_TOKENS_PER_INPUT:
        return text, len(toks)
    logger.warning(
        "truncating %d-token input to %d for the embeddings API",
        len(toks), EMBED_MAX_TOKENS_PER_INPUT,
    )
    return (
        enc.decode(toks[:EMBED_MAX_TOKENS_PER_INPUT]),
        EMBED_MAX_TOKENS_PER_INPUT,
    )


def _build_batches(texts: list[str]) -> list[list[str]]:
    """Split into batches respecting both count and per-request token caps."""
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        clamped, n = _truncate(t)
        if cur and (
            len(cur) >= MAX_BATCH_SIZE
            or cur_tokens + n > EMBED_MAX_TOKENS_PER_REQUEST
        ):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(clamped)
        cur_tokens += n
    if cur:
        batches.append(cur)
    return batches


async def _embed_batch(batch: list[str]) -> list[list[float]]:
    from treeloom.infrastructure.tracing import get_tracer

    client = _get_client()
    last_exc: Exception | None = None
    with get_tracer("treeloom.embedding").start_as_current_span(
        "openai.embed",
        attributes={
            "treeloom.embedding_model": EMBEDDING_MODEL,
            "treeloom.batch_size": len(batch),
        },
    ):
        for attempt in range(EMBED_RETRIES):
            try:
                resp = await client.post(
                    f"{EMBEDDING_BASE_URL}/embeddings",
                    json={"model": EMBEDDING_MODEL, "input": batch},
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()["data"]
                # The API documents order-preservation but indexes are
                # authoritative — sort by them rather than trusting order.
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status != 429 and status < 500:
                    logger.error(
                        "embeddings API %d: %s", status, exc.response.text[:200]
                    )
                    raise
            except httpx.HTTPError as exc:
                last_exc = exc
            await asyncio.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts via the OpenAI-compatible API, preserving input order."""
    if not texts:
        return []
    sem = _get_semaphore()

    async def _run(batch: list[str]) -> list[list[float]]:
        async with sem:
            return await _embed_batch(batch)

    results = await asyncio.gather(*(_run(b) for b in _build_batches(texts)))
    return [vec for batch_vecs in results for vec in batch_vecs]


async def embed_query(texts: list[str]) -> list[list[float]]:
    """Embed search queries, applying the instruction prefix if configured.

    OpenAI embedding models need no prefix — EMBEDDING_QUERY_PREFIX should
    stay empty for them; honored anyway for compatible endpoints serving
    instruction-prefix models.
    """
    if EMBEDDING_QUERY_PREFIX:
        texts = [EMBEDDING_QUERY_PREFIX + t for t in texts]
    return await embed(texts)
