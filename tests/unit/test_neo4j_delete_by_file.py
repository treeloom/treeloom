"""Unit tests for Neo4j delete_entities_by_file adapter (T007).

Detroit-style: mock only neo4j (external), never internal classes.
"""
from unittest.mock import AsyncMock, MagicMock, ANY

import pytest


# ---------------------------------------------------------------------------
# T007 – delete_entities_by_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_entities_by_file_runs_cypher(mock_neo4j_driver):
    """delete_entities_by_file should execute a Cypher DETACH DELETE with params."""
    from treeloom.adapters.neo4j.graph_store import delete_entities_by_file

    # Set up the mock result so consume() works
    session_ctx = mock_neo4j_driver.session.return_value
    mock_session = session_ctx.__aenter__.return_value

    mock_result = MagicMock()
    mock_summary = MagicMock()
    mock_counters = MagicMock()
    mock_counters.nodes_deleted = 3
    mock_summary.counters = mock_counters
    mock_result.consume = AsyncMock(return_value=mock_summary)
    mock_session.run.return_value = mock_result

    result = await delete_entities_by_file(source_id="src-1", file_path="src/main.py")

    assert result == 3

    # Verify driver.session() was called
    mock_neo4j_driver.session.assert_called()

    # Verify session.run was called with correct params
    mock_session.run.assert_called()
    call_args = mock_session.run.call_args
    cypher = call_args[0][0]
    params = call_args[1]

    assert "DETACH DELETE" in cypher
    assert "source_id" in cypher.lower() or "$source_id" in cypher
    assert "file_path" in cypher.lower() or "$file_path" in cypher
    assert params["source_id"] == "src-1"
    assert params["file_path"] == "src/main.py"


@pytest.mark.asyncio
async def test_delete_entities_by_file_returns_count(mock_neo4j_driver):
    """delete_entities_by_file should return the number of deleted nodes."""
    from treeloom.adapters.neo4j.graph_store import delete_entities_by_file

    session_ctx = mock_neo4j_driver.session.return_value
    mock_session = session_ctx.__aenter__.return_value

    # Simulate Neo4j returning a count summary
    mock_result = MagicMock()
    mock_summary = MagicMock()
    mock_counters = MagicMock()
    mock_counters.nodes_deleted = 7
    mock_summary.counters = mock_counters
    mock_result.consume = AsyncMock(return_value=mock_summary)
    mock_session.run.return_value = mock_result

    result = await delete_entities_by_file(source_id="src-2", file_path="lib/utils.py")

    assert result == 7


@pytest.mark.asyncio
async def test_delete_entities_by_file_no_match_returns_zero(mock_neo4j_driver):
    """delete_entities_by_file should return 0 when no entities match."""
    from treeloom.adapters.neo4j.graph_store import delete_entities_by_file

    session_ctx = mock_neo4j_driver.session.return_value
    mock_session = session_ctx.__aenter__.return_value

    mock_result = MagicMock()
    mock_summary = MagicMock()
    mock_counters = MagicMock()
    mock_counters.nodes_deleted = 0
    mock_summary.counters = mock_counters
    mock_result.consume = AsyncMock(return_value=mock_summary)
    mock_session.run.return_value = mock_result

    result = await delete_entities_by_file(
        source_id="nonexistent", file_path="ghost.py"
    )

    assert result == 0
