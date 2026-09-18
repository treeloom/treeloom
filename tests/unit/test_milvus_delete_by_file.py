"""Unit tests for Milvus delete_chunks_by_file adapter (T006).

Detroit-style: mock only pymilvus (external), never internal classes.
"""
from unittest.mock import MagicMock, ANY

import pytest


# ---------------------------------------------------------------------------
# T006 – delete_chunks_by_file
# ---------------------------------------------------------------------------

def test_delete_chunks_by_file_constructs_correct_filter(mock_milvus_collection):
    """delete_chunks_by_file should build a filter with both source_id and file_path."""
    from treeloom.adapters.milvus.vector_store import delete_chunks_by_file, COLLECTION_NAME

    delete_chunks_by_file(source_id="src-1", file_path="src/main.py")

    mock_milvus_collection.delete.assert_called_once()
    call_kwargs = mock_milvus_collection.delete.call_args
    assert call_kwargs[0][0] == COLLECTION_NAME
    filter_str = call_kwargs[1]["filter"]
    assert "source_id ==" in filter_str
    assert "file_path ==" in filter_str
    assert " and " in filter_str.lower()
    # Both values are bound, not interpolated — a file path is the one thing
    # here most likely to contain a quote or a backslash.
    assert "src-1" not in filter_str
    assert "src/main.py" not in filter_str
    assert call_kwargs[1]["filter_params"] == {
        "f_source_id": "src-1", "f_file_path": "src/main.py",
    }


def test_delete_chunks_by_file_returns_delete_count(mock_milvus_collection):
    """delete_chunks_by_file should return the number of deleted entities."""
    from treeloom.adapters.milvus.vector_store import delete_chunks_by_file

    # Simulate Milvus returning delete_count
    mock_milvus_collection.delete.return_value = {"delete_count": 5}

    result = delete_chunks_by_file(source_id="src-1", file_path="src/main.py")

    assert result == 5


def test_delete_chunks_by_file_handles_missing_collection(mock_milvus_collection):
    """delete_chunks_by_file should return 0 when collection does not exist."""
    from treeloom.adapters.milvus.vector_store import delete_chunks_by_file

    mock_milvus_collection.has_collection.return_value = False
    mock_milvus_collection.delete.reset_mock()

    result = delete_chunks_by_file(source_id="src-1", file_path="src/main.py")

    assert result == 0
    mock_milvus_collection.delete.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        'src-1" OR 1==1 --',      # quote breakout
        "src-1\\",                # trailing backslash — the old escape's blind spot
        'a"b\\c"d',               # both, interleaved
    ],
)
def test_delete_chunks_by_file_binds_values_rather_than_escaping_them(payload):
    """This asserted only that a backslash-quote appeared somewhere in the
    expression, which the old `.replace('"', '\\"')` satisfied while still
    being wrong: `\\` is itself an escape character in Milvus, so a value
    ending in one swallowed the closing quote and the server rejected the
    expression (error 1100). Binding removes the class."""
    from treeloom.adapters.milvus.vector_store import MilvusAdapter
    from unittest.mock import MagicMock

    mock_mc = MagicMock()
    mock_mc.has_collection.return_value = True
    mock_mc.delete.return_value = {"delete_count": 0}

    adapter = MilvusAdapter(host="localhost", port="19530")
    adapter._client = mock_mc

    adapter.delete_chunks_by_file(source_id=payload, file_path=payload)

    kwargs = mock_mc.delete.call_args[1]
    assert '"' not in kwargs["filter"] and "\\" not in kwargs["filter"]
    assert kwargs["filter_params"] == {
        "f_source_id": payload, "f_file_path": payload,
    }
