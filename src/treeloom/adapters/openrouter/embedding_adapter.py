"""OpenRouter embedding adapter — drop-in replacement for TEI embedder.

Uses OpenRouter's unified embeddings API with model routing.
Implements the same `embed(texts) -> list[list[float]]` interface
as the TEI adapter so the rest of Treeloom needs zero changes.

Model: openai/text-embedding-3-small (1536-dim, $0.02/1M tokens)
Endpoint: POST https://openrouter.ai/api/v1/embeddings
"""

from __future__ import annotations

import os
import sys

import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBED_MODEL = os.environ.get(
    "OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small"
)
MAX_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=120.0,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
        )
    return _client


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenRouter's embeddings API.

    Batches into groups of MAX_BATCH_SIZE for efficiency.
    Returns one embedding vector per input text, preserving order.
    """
    if not texts:
        return []

    client = _get_client()
    import asyncio

    all_embeddings: list[list[float]] = []
    total = len(texts)
    done = 0

    for i in range(0, total, MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        try:
            resp = await client.post(
                OPENROUTER_EMBED_URL,
                json={
                    "model": OPENROUTER_EMBED_MODEL,
                    "input": batch,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
            done += len(batch)
            if total > 32:  # Only log progress for large jobs
                print(
                    f"[embed] openrouter {done}/{total} ({done * 100 // total}%)",
                    file=sys.stderr,
                )
        except httpx.HTTPStatusError as exc:
            print(
                f"[embed] openrouter HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}",
                file=sys.stderr,
            )
            raise
        except httpx.HTTPError as exc:
            print(
                f"[embed] openrouter error: {exc}",
                file=sys.stderr,
            )
            raise

    return all_embeddings
