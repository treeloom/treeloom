"""Shared pytest fixtures for treeloom unit tests.

Mock only out-of-process dependencies (TEI, Milvus, Neo4j, LLM API).
Never mock internal treeloom classes — Principle VIII (Detroit-style).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Hermetic environment
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_auth_env(monkeypatch):
    """Default AUTH_ENABLED off for every unit test.

    The indexer loads a developer's `.env` (dotenv searches upward, so a
    repo-root `.env` is picked up even when tests run from a nested git
    worktree). If that `.env` sets AUTH_ENABLED=true, the AuthMiddleware would
    401 every endpoint test that doesn't explicitly opt into auth. Pin it off
    here so the suite is hermetic; tests that exercise auth set it true
    themselves via ``mocker.patch.dict(os.environ, {"AUTH_ENABLED": "true"})``.
    """
    monkeypatch.setenv("AUTH_ENABLED", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(dim: int = 768, count: int = 1) -> list[list[float]]:
    """Deterministic fake embedding vectors."""
    vectors: list[list[float]] = []
    for n in range(count):
        # Seed each vector so results are deterministic and unique per index.
        vec = [((n * 17 + i * 13) % 1000) / 1000.0 for i in range(dim)]
        vectors.append(vec)
    return vectors


def _fake_neo4j_entity(
    *,
    id: str = "entity-1",
    type: str = "Function",
    name: str = "my_func",
    signature: str = "def my_func(x: int) -> str",
    file_path: str = "src/main.py",
    start_line: int = 10,
    end_line: int = 25,
    language: str = "python",
) -> dict[str, Any]:
    """Build a Neo4j entity dict matching the treeloom Entity schema."""
    return {
        "id": id,
        "type": type,
        "name": name,
        "signature": signature,
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": language,
    }


# ---------------------------------------------------------------------------
# TEI Embedding mock  (httpx.AsyncClient → /embed)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tei_embed(mocker) -> AsyncMock:
    """Mock httpx.AsyncClient for TEI embedding service.

    Dynamically generates fake embeddings proportional to batch size.
    Returns 768-dim fake embeddings.
    """
    mock_client = AsyncMock()

    async def _post_side(url, json=None, **_kw):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if json and isinstance(json, dict):
            inputs = json.get("inputs", [])
            resp.json.return_value = _fake_embedding(count=len(inputs))
        else:
            resp.json.return_value = _fake_embedding(count=1)
        return resp

    mock_client.post = AsyncMock(side_effect=_post_side)
    return mock_client


# ---------------------------------------------------------------------------
# TEI Reranker mock  (httpx.AsyncClient → /rerank)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tei_rerank(mocker) -> AsyncMock:
    """Mock httpx.AsyncClient for TEI reranker service.

    Simulates POST {RERANKER_URL}/rerank → [{"index": 0, "score": 0.95}, ...].
    """
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    def _rerank_response(query: str, texts: list[str], **kwargs) -> list[dict]:
        return [{"index": i, "score": 0.95 - i * 0.01} for i in range(len(texts))]

    mock_response.json.side_effect = (
        lambda: _rerank_response("", [])
    )
    mock_response.json.return_value = [
        {"index": 0, "score": 0.95},
        {"index": 1, "score": 0.87},
        {"index": 2, "score": 0.72},
    ]
    mock_client.post.return_value = mock_response

    return mock_client


# ---------------------------------------------------------------------------
# Milvus collection mock  (pymilvus.Collection / MilvusClient)
# ---------------------------------------------------------------------------

def _make_milvus_search_hit(
    *,
    file_path: str = "src/main.py",
    start_line: int = 10,
    end_line: int = 25,
    language: str = "python",
    source_id: str = "src-1",
    chunk_text: str = "def hello():\\n    return 'world'",
    score: float = 0.92,
) -> dict[str, Any]:
    """Build a single Milvus search-hit dict matching the treeloom schema."""
    return {
        "id": 1,
        "distance": score,
        "relevance_score": score,
        "entity": {
            "chunk_text": chunk_text,
            "file_path": file_path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "source_id": source_id,
        },
    }


@pytest.fixture
def mock_milvus_collection(mocker) -> MagicMock:
    """Mock pymilvus.MilvusClient for vector search.

    search() returns a list of hit dicts.
    hybrid_search() returns similarly-structured results.
    """
    mock_mc = MagicMock()
    mock_mc.has_collection.return_value = True

    def _search_results(*args, **kwargs):
        limit = kwargs.get("limit", 3)
        hits = [_make_milvus_search_hit() for _ in range(min(limit, 3))]
        return [hits]

    mock_mc.search.side_effect = _search_results
    mock_mc.hybrid_search.side_effect = _search_results
    mock_mc.insert.return_value = {"insert_count": 1}
    mock_mc.delete.return_value = {}
    mock_mc.prepare_index_params.return_value = MagicMock()

    # Patch the module-level _client used by treeloom.adapters.milvus.vector_store.
    mocker.patch.object(
        __import__("treeloom.adapters.milvus.vector_store", fromlist=["_get_client"]),
        "_get_client",
        return_value=mock_mc,
    )
    # Reset cached default adapter so it picks up the patched _get_client
    vs = __import__("treeloom.adapters.milvus.vector_store", fromlist=["_get_adapter"])
    vs._get_adapter.__wrapped__ if hasattr(vs._get_adapter, "__wrapped__") else None
    # Directly patch _get_client on the class so new instances get the mock
    mocker.patch.object(
        vs.MilvusAdapter, "_get_client", return_value=mock_mc
    )
    return mock_mc


@pytest.fixture
def mock_milvus_search_hit() -> dict[str, Any]:
    """Single pre-built Milvus search hit for convenience."""
    return _make_milvus_search_hit()


# ---------------------------------------------------------------------------
# Neo4j driver mock  (neo4j.AsyncDriver / AsyncSession)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_neo4j_driver(mocker) -> AsyncMock:
    """Mock neo4j.AsyncDriver with execute_query() returning fake entities.

    Uses the _fake_neo4j_entity helper so results match the treeloom schema.
    """
    mock_driver = AsyncMock()
    mock_session = AsyncMock()

    # Default: return a single entity
    default_entity = _fake_neo4j_entity()
    mock_record = MagicMock()
    mock_record.__getitem__ = MagicMock(return_value=default_entity)
    mock_result = MagicMock()
    mock_result.fetch = AsyncMock(return_value=[mock_record])
    mock_session.run = AsyncMock(return_value=mock_result)

    # Make session() return an async context manager wrapping mock_session
    _session_ctx = MagicMock()
    _session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    _session_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_driver.session = MagicMock(return_value=_session_ctx)

    # Patch graph_store's _driver
    mocker.patch.object(
        __import__("treeloom.adapters.neo4j.graph_store", fromlist=["_driver"]),
        "_driver",
        mock_driver,
    )
    return mock_driver


# ---------------------------------------------------------------------------
# LLM httpx mock  (httpx.AsyncClient → /chat/completions)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_httpx(mocker) -> AsyncMock:
    """Mock httpx.AsyncClient for LLM API calls.

    Simulates POST {LLM_URL}/chat/completions with an OpenAI-compatible response.
    """
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "def example():\n    return 'fake LLM output'",
                }
            }
        ]
    }
    mock_client.post.return_value = mock_response

    # Patch the module-level _client in treeloom.adapters.llm_api.llm_adapter.
    mocker.patch.object(
        __import__("treeloom.adapters.llm_api.llm_adapter", fromlist=["_get_client"]),
        "_get_client",
        return_value=mock_client,
    )
    return mock_client
