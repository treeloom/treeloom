"""Reranker dispatch — RERANKER_PROVIDER env selects the backend.

- `http` (default): TEI-protocol POST to RERANKER_URL/rerank (TEI bge,
  services/qwen3_reranker, anything /rerank-compatible).
- `local`: in-process CPU cross-encoder via fastembed/ONNX — no service,
  no GPU, no API key (the simple-profile choice).
- `local-st`: in-process sentence-transformers `CrossEncoder` — loads any HF
  cross-encoder fastembed can't (e.g. zeroentropy/zerank-*); uses a GPU if
  available, else CPU. Self-hosted, no API key.
- `voyage`/`cohere`/`zeroentropy` (and future `jina`): hosted rerank API —
  no GPU, no local model, an API key. The GPU-less quality option.

All expose `async rerank_texts(query, texts, batch_size=None)
-> list[tuple[index, raw_score]]`.
"""
import os

_provider = os.environ.get("RERANKER_PROVIDER", "http")

# Hosted rerank providers (cloud_reranker.py). Keep in sync with its
# _PROVIDERS registry.
_CLOUD_PROVIDERS = {"voyage", "cohere", "zeroentropy", "jina"}

if _provider == "local":
    from treeloom.adapters.rerank.local_cross_encoder import rerank_texts  # noqa: F401
elif _provider == "local-st":
    from treeloom.adapters.rerank.st_cross_encoder import rerank_texts  # noqa: F401
elif _provider in _CLOUD_PROVIDERS:
    from treeloom.adapters.rerank.cloud_reranker import rerank_texts  # noqa: F401
else:
    from treeloom.adapters.rerank.http_reranker import rerank_texts  # noqa: F401
