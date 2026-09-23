import logging
import os

from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from treeloom.config import require_env
from treeloom.domain.indexing import VectorStorePort

logger = logging.getLogger(__name__)


MILVUS_HOST = require_env("MILVUS_HOST")
MILVUS_PORT = require_env("MILVUS_PORT")
MILVUS_URI = os.environ.get(
    "MILVUS_URI",
    f"https://{MILVUS_HOST}:{MILVUS_PORT}",
)
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "treeloom_chunks")
VECTOR_DIM = int(require_env("VECTOR_DIM"))
RRF_K = int(os.environ.get("RRF_K", "60"))

_client: MilvusClient | None = None
_default_adapter: "MilvusAdapter | None" = None


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_URI)
    return _client


class MilvusAdapter(VectorStorePort):
    """Milvus implementation of VectorStorePort.

    Manages a Milvus collection for storing and searching code chunk embeddings.
    Each instance owns its own MilvusClient connection.
    """

    def __init__(
        self,
        host: str,
        port: str,
        collection_name: str | None = None,
        vector_dim: int | None = None,
    ):
        self.host = host
        self.port = port
        self.uri = f"https://{host}:{port}"
        self.collection_name = collection_name or COLLECTION_NAME
        self.vector_dim = vector_dim or VECTOR_DIM
        self._client: MilvusClient | None = None

    def _get_client(self) -> MilvusClient:
        """Lazily create and return the MilvusClient for this adapter."""
        if self._client is None:
            self._client = MilvusClient(self.uri)
        return self._client

    def _reconnect(self) -> MilvusClient:
        """Force a new MilvusClient connection after channel errors."""
        self._client = MilvusClient(self.uri)
        return self._client

    async def _execute_with_reconnect(self, method, *args, max_retries: int = 3, **kwargs):
        """Execute a Milvus operation with automatic reconnect on channel errors."""
        import asyncio
        from treeloom.infrastructure.tracing import get_tracer
        _tracer = get_tracer("treeloom.milvus")
        op_name = method.__name__ if hasattr(method, "__name__") else str(method)
        with _tracer.start_as_current_span(
            f"milvus.{op_name}",
            attributes={"db.system": "milvus"},
        ) as _span:
            last_exc = None
            for attempt in range(max_retries):
                mc = self._get_client()
                try:
                    return await asyncio.to_thread(method, mc, *args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    if "closed channel" in msg or "Cannot invoke RPC" in msg:
                        try:
                            _span.add_event("reconnect", {"attempt": attempt + 1})
                        except Exception:
                            pass
                        self._reconnect()
                        continue
                    try:
                        from opentelemetry.trace import Status, StatusCode
                        _span.set_status(Status(StatusCode.ERROR, str(exc)))
                        _span.record_exception(exc)
                    except Exception:
                        pass
                    raise
            from treeloom.infrastructure.logging import log as struct_log
            struct_log.error(
                "milvus_reconnect_exhausted",
                method=op_name,
                retries=max_retries,
                error=str(last_exc),
            )
            try:
                from opentelemetry.trace import Status, StatusCode
                _span.set_status(Status(StatusCode.ERROR, str(last_exc)))
                if last_exc is not None:
                    _span.record_exception(last_exc)
            except Exception:
                pass
            raise last_exc

    # ------------------------------------------------------------------
    # VectorStorePort implementation
    # ------------------------------------------------------------------

    async def init_collection(self):
        """Create the Milvus collection with the treeloom schema if it does not exist."""
        if await self._execute_with_reconnect(MilvusClient.has_collection, self.collection_name):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="chunk_text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(
                name="summary_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim,
            ),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="start_line", dtype=DataType.INT64),
            FieldSchema(name="end_line", dtype=DataType.INT64),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=128),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        schema.add_function(
            Function(
                name="bm25_chunk_text",
                function_type=FunctionType.BM25,
                input_field_names=["chunk_text"],
                output_field_names=["sparse_vector"],
            )
        )

        mc = self._get_client()
        index_params = mc.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="summary_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": 1.2, "bm25_b": 0.75},
        )

        await self._execute_with_reconnect(
            MilvusClient.create_collection,
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    async def insert(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        summary_embeddings: list[list[float] | None] | None = None,
    ):
        """Insert chunks and their embeddings into the Milvus collection.

        Handles batching (max 100 per insert) and text truncation (max 50000 chars).
        """
        mc = self._get_client()
        MAX_TEXT_LEN = 50000
        MAX_INSERT_BATCH = 100
        data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Accept both dicts and Chunk dataclass instances
            if hasattr(chunk, "text"):
                text = chunk.text
                row = {
                    "vector": emb,
                    "chunk_text": text[:MAX_TEXT_LEN] if len(text) > MAX_TEXT_LEN else text,
                    "file_path": chunk.file_path,
                    "language": getattr(chunk, "language", ""),
                    "start_line": int(getattr(chunk, "start_line", 0)),
                    "end_line": int(getattr(chunk, "end_line", 0)),
                    "source_id": getattr(chunk, "source_id", ""),
                }
            else:
                text = chunk["text"]
                if len(text) > MAX_TEXT_LEN:
                    text = text[:MAX_TEXT_LEN]
                row = {
                    "vector": emb,
                    "chunk_text": text,
                    "file_path": chunk["file_path"],
                    "language": chunk.get("language", ""),
                    "start_line": int(chunk.get("start_line", 0)),
                    "end_line": int(chunk.get("end_line", 0)),
                    "source_id": chunk.get("source_id", ""),
                }
            if summary_embeddings is not None and summary_embeddings[i] is not None:
                row["summary_vector"] = summary_embeddings[i]
            else:
                row["summary_vector"] = [0.0] * self.vector_dim
            data.append(row)
        # pymilvus 3.0 can hit recursion depth on large batch inserts.
        # `_execute_with_reconnect` already wraps the call in
        # `asyncio.to_thread`, so the per-batch helper must be SYNC —
        # passing an async function would just create a coroutine that
        # never gets awaited and silently drop the insert.
        for offset in range(0, len(data), MAX_INSERT_BATCH):
            batch = data[offset : offset + MAX_INSERT_BATCH]

            def _insert_batch(mc, coll, b):
                return mc.insert(coll, b)

            await self._execute_with_reconnect(
                _insert_batch,
                self.collection_name,
                batch,
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        language: str | None = None,
        path_prefix: str | None = None,
        source_id: str | None = None,
        exclude_source_ids: list[str] | None = None,
    ) -> list[dict]:
        """Dense vector search with optional metadata filters."""
        mc = self._get_client()
        kwargs: dict = {
            "data": [query_embedding],
            "limit": top_k,
            "output_fields": SEARCH_OUTPUT_FIELDS,
            "anns_field": "vector",
        }
        expr, params = _build_filter(
            language, path_prefix, source_id, exclude_source_ids
        )
        if expr:
            kwargs["filter"] = expr
            if params:
                kwargs["filter_params"] = params
        results = mc.search(self.collection_name, **kwargs)
        return results[0]

    def hybrid_search(
        self,
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
        """Hybrid search combining dense, summary, HyDE, and BM25 sparse vectors."""
        mc = self._get_client()
        expr, expr_params = _build_filter(
            language, path_prefix, source_id, exclude_source_ids
        )
        expr = expr or None
        # AnnSearchRequest takes the template bindings as expr_params. Every
        # sub-request must carry them: an unbound placeholder is a parse error,
        # and a sub-request that silently dropped the ACL clause would return
        # denied rows for the ranker to fuse in.
        req_params: dict = {"expr_params": expr_params} if expr_params else {}
        ef = max(200, top_k)

        reqs: list[AnnSearchRequest] = [
            AnnSearchRequest(
                data=[query_dense],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": ef}},
                limit=top_k,
                expr=expr,
                **req_params,
            )
        ]
        if hyde_dense is not None:
            reqs.append(
                AnnSearchRequest(
                    data=[hyde_dense],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"ef": ef}},
                    limit=top_k,
                    expr=expr,
                    **req_params,
                )
            )
        if query_summary_dense is not None:
            reqs.append(
                AnnSearchRequest(
                    data=[query_summary_dense],
                    anns_field="summary_vector",
                    param={"metric_type": "COSINE", "params": {"ef": ef}},
                    limit=top_k,
                    expr=expr,
                    **req_params,
                )
            )
        reqs.append(
            AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}},
                limit=top_k,
                expr=expr,
                **req_params,
            )
        )

        results = mc.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=RRFRanker(k=rrf_k or RRF_K),
            limit=top_k,
            output_fields=SEARCH_OUTPUT_FIELDS,
        )
        return results[0]

    def delete_by_filter(self, field: str, value: str):
        """Delete chunks matching a field == value filter expression.

        Implements the VectorStorePort.delete_by_filter interface.
        Escapes quotes in values to prevent injection.
        """
        mc = self._get_client()
        if not mc.has_collection(self.collection_name):
            return
        # `field` is a code-supplied column name, never caller input; `value`
        # is bound server-side rather than escaped.
        mc.delete(
            self.collection_name,
            filter=f"{field} == {{f_value}}",
            filter_params={"f_value": value},
        )

    def delete_chunks_by_source(self, source_id: str):
        """Delete all chunks for a given source_id (convenience method)."""
        self.delete_by_filter("source_id", source_id)

    def list_indexed_paths(self, source_id: str) -> set[str]:
        """Return the set of distinct file_paths already indexed for a source.

        Used by the indexer's file-level concurrency loop to filter out
        files that completed in a prior run so resume doesn't re-process
        them. Returns an empty set if the collection doesn't exist yet.
        """
        mc = self._get_client()
        if not mc.has_collection(self.collection_name):
            return set()
        # query_iterator is the one call here with no filter_params support
        # (pymilvus builds a QueryIterator that never forwards expr_params),
        # so this clause stays interpolated and must be escaped properly.
        paths: set[str] = set()
        it = mc.query_iterator(
            collection_name=self.collection_name,
            filter=f'source_id == "{_escape_literal(source_id)}"',
            output_fields=["file_path"],
            batch_size=10000,
        )
        try:
            while True:
                batch = it.next()
                if not batch:
                    break
                for row in batch:
                    p = row.get("file_path") or ""
                    if p:
                        paths.add(p)
        finally:
            it.close()
        return paths

    def get_chunk_bodies(
        self, locations: list[tuple[str, int, int]],
        source_id: str | None = None,
        exclude_source_ids: list[str] | None = None,
    ) -> list[dict]:
        """Reconstruct the code bodies for facet `hit_id` locations.

        `locations` is a list of (file_path, start_line, end_line). For each, the
        chunk(s) whose own range falls inside [start_line, end_line] are fetched
        and their `chunk_text` concatenated in line order — so a MERGED facet hit
        (range spanning several stored chunks) rehydrates to the same body the
        agent would have seen in full mode. Returns one dict per location (in
        input order) shaped like a Milvus search hit's entity, with the empty
        ones dropped (a bad/stale location yields no row).
        """
        mc = self._get_client()
        if not mc.has_collection(self.collection_name) or not locations:
            return []
        out: list[dict] = []
        for file_path, start, end in locations:
            # file_path is derived from an agent-supplied hit_id, so it is the
            # injection boundary here; bind it rather than escape it. start/end
            # are int()-cast, which is its own sufficient guard.
            filt = (
                "file_path == {f_file_path} and start_line >= "
                f"{int(start)} and end_line <= {int(end)}"
            )
            filt_params: dict = {"f_file_path": file_path}
            if source_id:
                filt += " and source_id == {f_source_id}"
                filt_params["f_source_id"] = source_id
            if exclude_source_ids:
                # The caller's denied sources. Without this, a hit_id is just
                # a file path and line range, so a caller could hydrate a body
                # out of a source their grants exclude.
                filt += " and source_id not in {f_exclude}"
                filt_params["f_exclude"] = list(exclude_source_ids)
            rows = mc.query(
                collection_name=self.collection_name,
                filter=filt,
                filter_params=filt_params,
                output_fields=SEARCH_OUTPUT_FIELDS,
            )
            if not rows:
                continue
            rows.sort(key=lambda r: int(r.get("start_line") or 0))
            body = "\n".join((r.get("chunk_text") or "") for r in rows)
            first = rows[0]
            out.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "language": first.get("language", "") or "",
                "chunk_text": body,
                "source_id": first.get("source_id"),
            })
        return out

    def delete_chunks_by_file(self, source_id: str, file_path: str) -> int:
        """Delete all chunks for a specific file within a source.

        Returns:
            Number of deleted chunks, or 0 if the collection does not exist.
        """
        mc = self._get_client()
        if not mc.has_collection(self.collection_name):
            return 0
        result = mc.delete(
            self.collection_name,
            filter="source_id == {f_source_id} and file_path == {f_file_path}",
            filter_params={"f_source_id": source_id, "f_file_path": file_path},
        )
        return result.get("delete_count", 0)


# ---------------------------------------------------------------------------
# Module-level free functions — thin wrappers around the default adapter.
# Application code that imports these directly must not break.
# ---------------------------------------------------------------------------


_default_adapter: MilvusAdapter | None = None


def _get_adapter() -> MilvusAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = MilvusAdapter(host=MILVUS_HOST, port=MILVUS_PORT)
    return _default_adapter


async def init_collection():
    return await _get_adapter().init_collection()


async def insert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    summary_embeddings: list[list[float] | None] | None = None,
):
    return await _get_adapter().insert(chunks, embeddings, summary_embeddings=summary_embeddings)


def delete_chunks_by_source(source_id: str):
    return _get_adapter().delete_chunks_by_source(source_id)


async def list_indexed_paths(source_id: str) -> set[str]:
    """Async wrapper — runs the sync Milvus query off the event loop."""
    import asyncio as _asyncio
    return await _asyncio.to_thread(_get_adapter().list_indexed_paths, source_id)


async def get_chunk_bodies(
    locations: list[tuple[str, int, int]],
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Async wrapper — reconstruct facet hit_id bodies off the event loop."""
    import asyncio as _asyncio
    return await _asyncio.to_thread(
        _get_adapter().get_chunk_bodies, locations, source_id, exclude_source_ids)


def delete_chunks_by_file(source_id: str, file_path: str) -> int:
    """Delete all chunks for a specific file within a source.

    Args:
        source_id: The source identifier.
        file_path: The file path to delete chunks for.

    Returns:
        Number of deleted chunks, or 0 if the collection does not exist.
    """
    return _get_adapter().delete_chunks_by_file(source_id, file_path)


SEARCH_OUTPUT_FIELDS = [
    "chunk_text",
    "file_path",
    "language",
    "start_line",
    "end_line",
    "source_id",
]


def _escape_literal(value: str) -> str:
    r"""Escape a value for inlining into a Milvus double-quoted string literal.

    Backslash FIRST, then the quote — the order is the whole point. `\` is an
    escape character in Milvus expressions, so a value ending in one swallows
    the closing quote if only the quote is escaped: `sid\` becomes
    `"sid\"`, and the expression no longer terminates. Three call sites in
    this module previously did `.replace('"', '\\"')` alone and each one
    404s or mis-deletes on a path containing a backslash (Windows paths, or
    any file whose name contains one).

    Prefer template parameters where Milvus supports them; this exists for
    the `like` operand, which it does not.
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _build_filter(
    language: str | None,
    path_prefix: str | None,
    source_id: str | None,
    exclude_source_ids: list[str] | None = None,
) -> tuple[str, dict]:
    r"""Build a Milvus boolean expression and its template parameters.

    Returns ``(expression, params)``. Pass *params* as ``filter_params=`` to
    MilvusClient.search/query/delete, or as ``expr_params=`` to
    AnnSearchRequest. An empty expression means "no filter".

    Every value that Milvus can bind server-side is bound, not interpolated.
    This used to interpolate all four, unescaped, and `path_prefix` is
    caller-supplied on /search — so a prefix of::

        %" or source_id == "secret-repo

    produced::

        file_path like "%" or source_id == "secret-repo%"
          and source_id not in ["secret-repo"]

    `and` binds tighter than `or`, so the left disjunct matched every row and
    the exclusion list never applied. That list is the per-repo ACL prefilter
— the deny grants it carries were voidable by any
    caller who could set a path prefix. Verified against Milvus 2.5.4: the
    interpolated form returns the denied source's rows; the form below
    returns none.

    LIMITATION — `path_prefix` is still interpolated, because Milvus 2.5.4
    cannot template a `like` operand (``file_path like {p}`` is a parse
    error, checked on a live server). It is escaped instead, backslash FIRST
    then quote, which is the order that actually holds: `\` is an escape
    character in Milvus string literals, so escaping only the quote leaves a
    trailing-backslash value able to swallow its own closing quote. Should a
    later Milvus template `like`, this clause should move too.

    Second limitation: Milvus `like` offers no way to escape `%`, so a literal
    `%` inside a path prefix acts as a wildcard and widens the match. It
    cannot cross the ACL — the exclusion clause is a separate, templated
    conjunct — so this only widens a caller's query within the scope they are
    already permitted.

    Each clause is parenthesised so that no future escaping slip can reorder
    the conjunction and detach the ACL clause the way the original did.
    """
    parts: list[str] = []
    params: dict = {}
    if language:
        parts.append("(language == {f_language})")
        params["f_language"] = language
    if path_prefix:
        parts.append(f'(file_path like "{_escape_literal(path_prefix)}%")')
    if source_id:
        parts.append("(source_id == {f_source_id})")
        params["f_source_id"] = source_id
    if exclude_source_ids:
        # Exclude denied sources from a shared-index (all_access) query.
        parts.append("(source_id not in {f_exclude})")
        params["f_exclude"] = list(exclude_source_ids)
    return " and ".join(parts), params


def search(
    query_embedding: list[float],
    top_k: int = 20,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    return _get_adapter().search(
        query_embedding, top_k, language, path_prefix, source_id,
        exclude_source_ids,
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
    return _get_adapter().hybrid_search(
        query_text, query_dense, query_summary_dense, hyde_dense,
        top_k, language, path_prefix, source_id, rrf_k,
        exclude_source_ids,
    )




# ---------------------------------------------------------------------------
# Backward-compatible re-exports — the search orchestration moved verbatim to
# treeloom.application.retrieval (store-agnostic). Existing imports of these
# names from this module keep working.
# ---------------------------------------------------------------------------
from treeloom.application.retrieval import (  # noqa: E402,F401
    GRAPH_ALPHA,
    GRAPH_BETA,
    GRAPH_DELTA,
    GRAPH_EPSILON,
    GRAPH_GAMMA,
    MILVUS_TOP_K_PRE_RERANK,
    RERANK_NORMALIZE,
    SYMBOL_PROMOTE,
    USE_GRAPH_SCORING,
    USE_HYBRID,
    USE_HYDE,
    USE_SUMMARY_VECTOR,
    _adaptive_cut,
    _apply_response_budget,
    _chunk_header,
    _cosine,
    _display_score,
    _file_defines_query_symbol,
    _get_community_embeddings,
    _maybe_hyde_embedding,
    _merge_adjacent_hits,
    _minmax,
    _name_in_query,
    _promote_definition,
    _query_symbols,
    _rank_neighbors,
    _splice_snippets,
    graph_enhanced_search,
    graph_rescore,
    graph_search,
    invalidate_graph_caches,
    rerank,
)
