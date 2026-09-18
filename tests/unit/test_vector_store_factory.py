"""Unit tests for vector_store_factory.create_vector_store().

Detroit-style: mock only external adapter classes (MilvusAdapter, ChromaAdapter),
never internal classes.
"""

from unittest.mock import patch, MagicMock

import pytest

from treeloom.infrastructure.config import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_adapter(name):
    """Create a mock adapter class that records instantiation."""
    mock_cls = MagicMock(name=name)
    mock_instance = MagicMock(name=f"{name}Instance")
    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


# ---------------------------------------------------------------------------
# Tests: create_vector_store
# ---------------------------------------------------------------------------

class TestCreateVectorStore:
    """Tests for factory function create_vector_store."""

    def test_backend_milvus(self):
        """Input config with backend='milvus' returns MilvusAdapter instance."""
        mock_milvus, mock_instance = _make_mock_adapter("MilvusAdapter")
        mock_chroma, _ = _make_mock_adapter("ChromaAdapter")

        with patch(
            "treeloom.infrastructure.vector_store_factory.MilvusAdapter", mock_milvus
        ), patch(
            "treeloom.infrastructure.vector_store_factory.ChromaAdapter", mock_chroma
        ):
            from treeloom.infrastructure.vector_store_factory import create_vector_store

            config = {"backend": "milvus", "milvus": {"host": "localhost", "port": "19530"}}
            result = create_vector_store(config)

            assert result is mock_instance
            mock_milvus.assert_called_once_with(host="localhost", port="19530")
            mock_chroma.assert_not_called()

    def test_backend_chromadb(self):
        """Input config with backend='chromadb' returns ChromaAdapter instance."""
        mock_milvus, _ = _make_mock_adapter("MilvusAdapter")
        mock_chroma, mock_instance = _make_mock_adapter("ChromaAdapter")

        with patch(
            "treeloom.infrastructure.vector_store_factory.MilvusAdapter", mock_milvus
        ), patch(
            "treeloom.infrastructure.vector_store_factory.ChromaAdapter", mock_chroma
        ):
            from treeloom.infrastructure.vector_store_factory import create_vector_store

            config = {"backend": "chromadb", "chromadb": {"path": "./chroma_data"}}
            result = create_vector_store(config)

            assert result is mock_instance
            mock_chroma.assert_called_once_with(path="./chroma_data")
            mock_milvus.assert_not_called()

    @pytest.mark.parametrize("backend", ["pgvector", "qdrant"])
    def test_removed_backends_are_rejected(self, backend):
        """pgvector and Qdrant adapters were removed: nothing in the runtime could
        select them (VECTOR_STORE=qdrant silently fell through to Milvus), so they
        only advertised support that did not exist. They must now fail loudly."""
        from treeloom.infrastructure.vector_store_factory import create_vector_store

        with pytest.raises(ValueError, match="Unknown vector store backend"):
            create_vector_store({"backend": backend})

    def test_backend_unknown_raises_value_error(self):
        """Input config with backend='unknown' raises ValueError."""
        from treeloom.infrastructure.vector_store_factory import create_vector_store

        config = {"backend": "unknown"}
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            create_vector_store(config)

    def test_missing_backend_raises_config_error(self):
        """Missing backend section raises ConfigError."""
        from treeloom.infrastructure.vector_store_factory import create_vector_store

        config = {}
        with pytest.raises(ConfigError):
            create_vector_store(config)
