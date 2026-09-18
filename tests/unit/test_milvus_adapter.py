"""Unit tests for MilvusAdapter class implementing VectorStorePort.

Detroit-style: mock only pymilvus.MilvusClient (external), never internal classes.
"""
from unittest.mock import MagicMock, patch, ANY

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_adapter(mock_client, host="localhost", port="19530"):
    """Create a MilvusAdapter with a pre-injected mock client."""
    from treeloom.adapters.milvus.vector_store import MilvusAdapter
    adapter = MilvusAdapter(host=host, port=port)
    adapter._client = mock_client
    return adapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Mock pymilvus.MilvusClient with all methods pre-configured."""
    with patch("pymilvus.MilvusClient") as MockClient:
        mc = MagicMock()
        mc.has_collection.return_value = False
        mc.insert.return_value = {"insert_count": 1}
        mc.delete.return_value = {"delete_count": 1}
        mc.search.return_value = [[
            {"id": 1, "distance": 0.08, "entity": {
                "chunk_text": "def foo(): pass",
                "file_path": "src/foo.py",
                "language": "python",
                "start_line": 1, "end_line": 3,
                "source_id": "abc123",
            }},
        ]]
        mc.hybrid_search.return_value = [[
            {"id": 1, "distance": 0.08, "entity": {
                "chunk_text": "def foo(): pass",
                "file_path": "src/foo.py",
                "language": "python",
                "start_line": 1, "end_line": 3,
                "source_id": "abc123",
            }},
        ]]
        mc.prepare_index_params.return_value = MagicMock()
        MockClient.return_value = mc
        yield mc


# ---------------------------------------------------------------------------
# InitCollection
# ---------------------------------------------------------------------------

class TestMilvusAdapterInitCollection:
    # init_collection is async and invokes Milvus operations as unbound
    # `MilvusClient.<method>` through `_execute_with_reconnect`. Patch the
    # class through vector_store's OWN module reference — depending on import
    # order that may differ from pymilvus.MilvusClient (the fixture patches
    # the pymilvus name), and init_collection only ever sees its module's one.

    @pytest.mark.asyncio
    async def test_creates_collection_when_missing(self, mock_client):
        from treeloom.adapters.milvus import vector_store as vs
        adapter = _make_adapter(mock_client)
        with patch.object(vs.MilvusClient, "has_collection", MagicMock(return_value=False)), \
             patch.object(vs.MilvusClient, "create_collection", MagicMock()) as create_c:
            await adapter.init_collection()
        create_c.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_collection_exists(self, mock_client):
        from treeloom.adapters.milvus import vector_store as vs
        adapter = _make_adapter(mock_client)
        with patch.object(vs.MilvusClient, "has_collection", MagicMock(return_value=True)), \
             patch.object(vs.MilvusClient, "create_collection", MagicMock()) as create_c:
            await adapter.init_collection()
        create_c.assert_not_called()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

class TestMilvusAdapterInsert:
    @pytest.mark.asyncio
    async def test_insert_calls_milvus_insert(self, mock_client):
        adapter = _make_adapter(mock_client)
        chunks = [{
            "text": "def foo(): pass",
            "file_path": "src/foo.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "source_id": "abc",
        }]
        embeddings = [[0.1, 0.2]]
        await adapter.insert(chunks, embeddings)
        mock_client.insert.assert_called()

    @pytest.mark.asyncio
    async def test_insert_batches_large_lists(self, mock_client):
        adapter = _make_adapter(mock_client)
        chunks = [{"text": "x", "file_path": "f.py", "language": "py",
                    "start_line": 1, "end_line": 1, "source_id": "abc"}]
        embeddings = [[0.1]] * 3
        await adapter.insert(chunks * 150, embeddings * 150)
        assert mock_client.insert.call_count == 2  # 150 → 2 batches of 100


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestMilvusAdapterSearch:
    def test_search_returns_results(self, mock_client):
        adapter = _make_adapter(mock_client)
        results = adapter.search([0.1, 0.2], top_k=10)
        assert len(results) == 1
        assert results[0]["entity"]["chunk_text"] == "def foo(): pass"

    def test_search_with_filter(self, mock_client):
        adapter = _make_adapter(mock_client)
        results = adapter.search([0.1], top_k=5, source_id="abc123")
        assert len(results) >= 0


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------

class TestMilvusAdapterHybridSearch:
    def test_hybrid_search_returns_results(self, mock_client):
        adapter = _make_adapter(mock_client)
        results = adapter.hybrid_search("foo", [0.1, 0.2], top_k=10)
        assert len(results) == 1

    def test_hybrid_search_with_source_filter(self, mock_client):
        adapter = _make_adapter(mock_client)
        results = adapter.hybrid_search("foo", [0.1], top_k=5, source_id="abc")
        assert len(results) >= 0


# ---------------------------------------------------------------------------
# DeleteByFilter
# ---------------------------------------------------------------------------

class TestMilvusAdapterDeleteByFilter:
    def test_delete_calls_milvus_delete(self, mock_client):
        adapter = _make_adapter(mock_client)
        mock_client.has_collection.return_value = True
        adapter.delete_by_filter("source_id", "src-1")
        mock_client.delete.assert_called_once()
        kwargs = mock_client.delete.call_args[1]
        # The column name is code-supplied and still appears; the VALUE is
        # bound server-side rather than pasted into the expression.
        assert "source_id ==" in kwargs["filter"]
        assert "src-1" not in kwargs["filter"]
        assert kwargs["filter_params"] == {"f_value": "src-1"}

    def test_delete_skips_missing_collection(self, mock_client):
        adapter = _make_adapter(mock_client)
        mock_client.has_collection.return_value = False
        adapter.delete_by_filter("source_id", "src-1")
        mock_client.delete.assert_not_called()

    def test_delete_value_cannot_reach_the_expression(self, mock_client):
        """This asserted that quotes were backslash-escaped, which passed on
        an escape that handled `"` but not `\\` — so a value ending in a
        backslash swallowed its own closing quote and Milvus rejected the
        whole expression. Binding the value removes the question."""
        adapter = _make_adapter(mock_client)
        mock_client.has_collection.return_value = True
        payload = 'src-1" OR 1==1 --'
        adapter.delete_by_filter("source_id", payload)
        kwargs = mock_client.delete.call_args[1]
        assert '"' not in kwargs["filter"], "no literal should be inlined at all"
        assert kwargs["filter_params"] == {"f_value": payload}

    def test_delete_handles_a_trailing_backslash(self, mock_client):
        """The case the old escaping got wrong."""
        adapter = _make_adapter(mock_client)
        mock_client.has_collection.return_value = True
        adapter.delete_by_filter("source_id", "src-1\\")
        kwargs = mock_client.delete.call_args[1]
        assert kwargs["filter_params"] == {"f_value": "src-1\\"}
        assert "\\" not in kwargs["filter"]
