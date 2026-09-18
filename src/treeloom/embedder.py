"""Embedder dispatch — EMBEDDING_PROVIDER env selects the backend.

- `tei` (default): cost-aware proxy routing batches across TEI backends
  registered in Postgres (`embedding_backends`). See
  `treeloom.adapters.tei.embedding_proxy` and
  `docs/adr-001-cost-aware-embedding-proxy.md`.
- `openai`: any OpenAI-compatible `/embeddings` endpoint (api.openai.com or
  a compatible base URL) — the simple-profile choice; bypasses the Postgres
  registry entirely.

Both expose `embed`, `embed_query`, and `MAX_BATCH_SIZE`.
"""
import os

_provider = os.environ.get("EMBEDDING_PROVIDER", "tei")

if _provider == "openai":
    from treeloom.adapters.openai_api.embedding_adapter import (  # noqa: F401
        MAX_BATCH_SIZE,
        embed,
        embed_query,
    )
else:
    from treeloom.adapters.tei.embedding_proxy import (  # noqa: F401
        MAX_BATCH_SIZE,
        embed,
        embed_query,
        get_proxy,
        set_proxy,
    )
