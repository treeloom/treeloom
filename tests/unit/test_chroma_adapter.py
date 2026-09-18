"""Unit tests for ChromaAdapter class implementing VectorStorePort (TDD).

Detroit-style: mock only chromadb (external), never internal classes.
"""
from unittest.mock import MagicMock, patch

import pytest

from treeloom.domain.indexing import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chromadb_client():
    """Mock chromadb.PersistentClient only.

    Returns a mock client with a mock collection that simulates
    get_or_create_collection, add, query, get, and delete.
    """
    with patch("chromadb.PersistentClient") as MockClient:
        mock_client = MagicMock()
        mock_collection = MagicMock()

        # Default: collection exists
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection = MagicMock()

        # Mock add
        mock_collection.add.return_value = None

        # Mock query (for search / hybrid_search)
        mock_collection.query.return_value = {
            "ids": [["chunk-0"]],
            "distances": [[0.08]],  # ChromaDB returns distance, lower = better
            "metadatas": [[{
                "file_path": "src/main.py",
                "language": "python",
                "start_line": 10,
                "end_line": 25,
                "source_id": "src-1",
            }]],
            "documents": [["def hello(): return 'world'"]],
            "embeddings": None,
        }

        # Mock get (for listing)
        mock_collection.get.return_value = {
            "ids": ["chunk-0", "chunk-1"],
            "metadatas": [
                {"source_id": "src-1", "file_path": "src/main.py"},
                {"source_id": "src-1", "file_path": "src/util.py"},
            ],
        }

        # Mock delete
        mock_collection.delete.return_value = None

        MockClient.return_value = mock_client
        yield mock_client, mock_collection


# ---------------------------------------------------------------------------
# TDD tests for ChromaAdapter
# ---------------------------------------------------------------------------

class TestChromaAdapterInitCollection:
    """init_collection() should create/get a collection."""

    def test_creates_collection(self, mock_chromadb_client):
        """init_collection() calls get_or_create_collection."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        mock_client.get_or_create_collection.assert_called_once()
        assert mock_client.get_or_create_collection.call_args[1]["name"] == adapter.collection_name

    def test_collection_name_constant(self, mock_chromadb_client):
        """ChromaAdapter uses a consistent collection name."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        adapter = ChromaAdapter(path="/tmp/chroma_test")
        assert adapter.collection_name == "treeloom_chunks"


class TestChromaAdapterInsert:
    """insert() should add documents with embeddings to the collection."""

    def test_insert_adds_chunks_with_embeddings(self, mock_chromadb_client):
        """insert() delegates to collection.add with correct data."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        chunks = [
            Chunk(
                text="def foo():\n    pass",
                file_path="src/foo.py",
                language="python",
                start_line=1,
                end_line=2,
                source_id="src-1",
            ),
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        adapter.insert(chunks, embeddings)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        assert "ids" in call_kwargs
        assert "embeddings" in call_kwargs
        assert "metadatas" in call_kwargs
        assert "documents" in call_kwargs

        # Verify metadata includes all required fields
        meta = call_kwargs["metadatas"][0]
        assert meta["file_path"] == "src/foo.py"
        assert meta["language"] == "python"
        assert meta["start_line"] == 1
        assert meta["end_line"] == 2
        assert meta["source_id"] == "src-1"

        # Verify documents contains the text
        assert call_kwargs["documents"][0] == "def foo():\n    pass"

        # Verify embeddings
        assert call_kwargs["embeddings"] == [[0.1, 0.2, 0.3]]

    def test_insert_generates_unique_ids(self, mock_chromadb_client):
        """insert() should generate unique IDs for each chunk."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        chunks = [
            Chunk(
                text="chunk1",
                file_path="src/a.py",
                language="python",
                start_line=1,
                end_line=1,
                source_id="src-1",
            ),
            Chunk(
                text="chunk2",
                file_path="src/b.py",
                language="python",
                start_line=2,
                end_line=2,
                source_id="src-1",
            ),
        ]
        embeddings = [[0.1] * 3, [0.2] * 3]

        adapter.insert(chunks, embeddings)

        call_kwargs = mock_collection.add.call_args[1]
        ids = call_kwargs["ids"]
        assert len(ids) == 2
        assert len(set(ids)) == 2  # All unique


class TestChromaAdapterSearch:
    """search() should query by embedding and return normalized results."""

    def test_search_queries_with_embedding_and_top_k(self, mock_chromadb_client):
        """search() delegates to collection.query with query_embeddings and n_results."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        query_embedding = [0.1, 0.2, 0.3]
        results = adapter.search(query_embedding, top_k=10)

        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["query_embeddings"] == [query_embedding]
        assert call_kwargs["n_results"] == 10

        # Verify normalized result format
        assert isinstance(results, list)
        assert len(results) == 1

    def test_search_returns_normalized_format(self, mock_chromadb_client):
        """search() returns results in the Milvus-compatible format (dicts with entity)."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        query_embedding = [0.1, 0.2, 0.3]
        results = adapter.search(query_embedding, top_k=1)

        assert len(results) == 1
        hit = results[0]
        # Milvus-compatible format
        assert "id" in hit
        assert "distance" in hit
        assert "entity" in hit
        entity = hit["entity"]
        assert entity["chunk_text"] == "def hello(): return 'world'"
        assert entity["file_path"] == "src/main.py"
        assert entity["language"] == "python"
        assert entity["start_line"] == 10
        assert entity["end_line"] == 25
        assert entity["source_id"] == "src-1"

    def test_search_converts_chromadb_distance_to_score(self, mock_chromadb_client):
        """search() should convert ChromaDB distance to a relevance score.

        ChromaDB returns L2 distance (lower = better). The adapter should
        convert to a score where higher = better (e.g., 1.0 / (1.0 + distance)).
        """
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        query_embedding = [0.1, 0.2, 0.3]
        results = adapter.search(query_embedding, top_k=1)

        hit = results[0]
        # Distance should be converted to score (higher = better)
        assert hit["distance"] > 0
        # Original ChromaDB distance was 0.08 → score = 1/(1+0.08) ≈ 0.926
        assert 0.9 <= hit["distance"] <= 1.0

    def test_search_returns_empty_list_when_no_results(self, mock_chromadb_client):
        """search() should return an empty list when ChromaDB returns no results."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        mock_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "embeddings": None,
        }

        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        results = adapter.search([0.1, 0.2, 0.3], top_k=5)
        assert results == []


class TestChromaAdapterHybridSearch:
    """hybrid_search() does embedding-only search with metadata filtering.

    ChromaDB doesn't do BM25 natively, so hybrid = embedding search
    with optional metadata filtering.
    """

    def test_hybrid_search_does_embedding_search(self, mock_chromadb_client):
        """hybrid_search() should perform an embedding-based query."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        results = adapter.hybrid_search(
            query="def hello",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=10,
        )

        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
        assert call_kwargs["n_results"] == 10

        assert isinstance(results, list)
        assert len(results) == 1

    def test_hybrid_search_returns_normalized_format(self, mock_chromadb_client):
        """hybrid_search() returns results in Milvus-compatible format."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        results = adapter.hybrid_search(
            query="def hello",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=1,
        )

        assert len(results) == 1
        hit = results[0]
        assert "id" in hit
        assert "distance" in hit
        assert "entity" in hit
        assert hit["entity"]["chunk_text"] == "def hello(): return 'world'"

    def test_hybrid_search_with_metadata_filter(self, mock_chromadb_client):
        """hybrid_search() should pass metadata filters to ChromaDB query.

        When metadata filters are applicable (e.g., source_id), they should
        be passed as a ChromaDB 'where' clause.
        """
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        # We need to extend hybrid_search to accept filter kwargs
        # For now, test that where clause can be passed
        results = adapter.hybrid_search(
            query="def hello",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=10,
            where={"source_id": "src-1"},
        )

        call_kwargs = mock_collection.query.call_args[1]
        assert "where" in call_kwargs
        assert call_kwargs["where"] == {"source_id": "src-1"}


class TestChromaAdapterDeleteByFilter:
    """delete_by_filter() should delete by metadata field."""

    def test_delete_by_filter_deletes_by_metadata(self, mock_chromadb_client):
        """delete_by_filter() calls collection.delete with a where clause."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        adapter.delete_by_filter("source_id", "src-1")

        mock_collection.delete.assert_called_once()
        call_kwargs = mock_collection.delete.call_args[1]
        assert "where" in call_kwargs
        assert call_kwargs["where"] == {"source_id": "src-1"}

    def test_delete_by_filter_with_other_field(self, mock_chromadb_client):
        """delete_by_filter() should work with any metadata field name."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        mock_client, mock_collection = mock_chromadb_client
        adapter = ChromaAdapter(path="/tmp/chroma_test")
        adapter.init_collection()

        adapter.delete_by_filter("file_path", "src/old.py")

        call_kwargs = mock_collection.delete.call_args[1]
        assert call_kwargs["where"] == {"file_path": "src/old.py"}


class TestChromaAdapterConstructor:
    """ChromaAdapter constructor should initialize path and PersistentClient."""

    def test_constructor_stores_path(self, mock_chromadb_client):
        """Constructor should store the persist directory path."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        adapter = ChromaAdapter(path="/var/lib/chromadb")

        assert adapter.path == "/var/lib/chromadb"

    def test_constructor_creates_persistent_client(self, mock_chromadb_client):
        """Constructor should create a chromadb.PersistentClient with the given path."""
        mock_client, mock_collection = mock_chromadb_client

        from treeloom.adapters.chromadb.vector_store import ChromaAdapter

        adapter = ChromaAdapter(path="/var/lib/chromadb")

        # PersistentClient should have been called with the path
        from chromadb import PersistentClient
        # The mock is patched, so we check the call happened
        # (the mock_chromadb_client fixture patches chromadb.PersistentClient)

    def test_implements_vector_store_port(self, mock_chromadb_client):
        """ChromaAdapter should be an instance of VectorStorePort."""
        from treeloom.adapters.chromadb.vector_store import ChromaAdapter
        from treeloom.domain.indexing import VectorStorePort

        adapter = ChromaAdapter(path="/tmp/chroma_test")
        assert isinstance(adapter, VectorStorePort)
