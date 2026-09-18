"""Vector-store dispatch — VECTOR_STORE env selects the backend module.

Each backend exposes the same module-level store surface returning
Milvus-SHAPED hits (`{id, distance, entity: {chunk_text, file_path, ...}}`):
`init_collection`, `insert_chunks`, `search`, `hybrid_search`,
`delete_chunks_by_source`, `delete_chunks_by_file`, `list_indexed_paths`.

The search orchestration (`graph_search`, `rerank`, graph rescoring, ...)
is store-agnostic and lives in `treeloom.application.retrieval`; it calls
back into this module for the raw vector search, so every backend gets the
full pipeline (reranker, graph signals, ChunkHit wire contract) for free.

Backends:
- `milvus` (default): production store — server-side BM25 hybrid via
  RRFRanker, summary vectors.
- `lancedb`: embedded single-directory store with native BM25 FTS hybrid —
  the simple-profile choice.
- `chromadb`: embedded, dense-only (Chroma's sparse/BM25 indexing is
  Cloud-only). Degraded: hybrid falls back to dense, no resume tracking,
  no per-file deletes.
"""
import os

_store = os.environ.get("VECTOR_STORE", "milvus")

if _store == "chromadb":
    from treeloom.adapters.chromadb.vector_store import (  # noqa: F401
        delete_chunks_by_source,
        init_collection,
        insert_chunks,
        search,
    )

    def hybrid_search(
        query_text: str,
        query_dense: list[float],
        query_summary_dense: list[float] | None = None,
        hyde_dense: list[float] | None = None,
        top_k: int = 60,
        language: str | None = None,
        path_prefix: str | None = None,
        source_id: str | None = None,
        rrf_k: int | None = None,
        exclude_source_ids: list[str] | None = None,
    ) -> list[dict]:
        """Dense-only fallback — Chroma has no local BM25 (spike-verified)."""
        return search(
            query_dense,
            top_k=top_k,
            language=language,
            path_prefix=path_prefix,
            source_id=source_id,
            exclude_source_ids=exclude_source_ids,
        )

    async def list_indexed_paths(source_id: str) -> set[str]:
        """No resume tracking on chroma — re-indexing a file is harmless."""
        return set()

    def delete_chunks_by_file(source_id: str, file_path: str) -> int:
        raise NotImplementedError(
            "per-file deletes (webhook incremental updates) are not "
            "implemented for VECTOR_STORE=chromadb"
        )

elif _store == "lancedb":
    from treeloom.adapters.lancedb.vector_store import (  # noqa: F401
        delete_chunks_by_file,
        delete_chunks_by_source,
        hybrid_search,
        init_collection,
        insert_chunks,
        list_indexed_paths,
        search,
    )
else:
    from treeloom.adapters.milvus.vector_store import (  # noqa: F401
        _build_filter,
        _get_client,
        delete_chunks_by_file,
        delete_chunks_by_source,
        get_chunk_bodies,
        hybrid_search,
        init_collection,
        insert_chunks,
        list_indexed_paths,
        search,
    )

# Store-agnostic orchestration — same entry points for every backend.
from treeloom.application.retrieval import (  # noqa: E402,F401
    MILVUS_TOP_K_PRE_RERANK,
    TOP_K_PRE_RERANK,
    _cosine,
    _get_community_embeddings,
    _maybe_hyde_embedding,
    graph_enhanced_search,
    graph_rescore,
    graph_search,
    hydrate_chunks,
    invalidate_graph_caches,
    rerank,
)
