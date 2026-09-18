"""Adapter tests for graph_store — Neo4j driver integration."""
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_store_graph_calls_neo4j():
    """store_graph should execute Cypher MERGE statements via Neo4j session."""
    from treeloom.graph_store import store_graph
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    entities = [
        {"id": "func://s/test.py::f", "name": "f", "type": "Function",
         "file_path": "test.py", "signature": "def f(): pass", "language": "python",
         "source_id": "s"},
    ]
    with patch("treeloom.adapters.neo4j.graph_store._get_session", return_value=mock_session):
        await store_graph(entities, [], source_id="s")
    assert mock_session.run.called


@pytest.mark.asyncio
async def test_store_graph_merges_source_node():
    """Regression: the entity-creation query must MERGE the Source node, not
    MATCH it. The Source node isn't created until _finalize_job runs (after
    all files are processed), so a MATCH finds nothing and silently drops
    every Entity — leaving find_definition/graph traversal empty."""
    from treeloom.graph_store import store_graph
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    entities = [
        {"id": "func://s/test.py::f", "name": "f", "type": "Function",
         "file_path": "test.py", "signature": "def f(): pass", "language": "python",
         "source_id": "s"},
    ]
    with patch("treeloom.adapters.neo4j.graph_store._get_session", return_value=mock_session):
        await store_graph(entities, [], source_id="s")
    entity_query = mock_session.run.call_args_list[0].args[0]
    assert "MERGE (s:Source {id: $source_id})" in entity_query
    assert "MATCH (s:Source" not in entity_query


@pytest.mark.asyncio
async def test_clear_all_calls_neo4j(mock_neo4j_driver):
    """clear_all() should execute a Cypher DELETE."""
    from treeloom.graph_store import clear_all
    await clear_all()


@pytest.mark.asyncio
async def test_get_entities_by_file(mock_neo4j_driver):
    """get_entities_by_file should query by file_path."""
    from treeloom.graph_store import get_entities_by_file
    result = await get_entities_by_file("/test/file.py", source_id="s")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_traverse_returns_list(mock_neo4j_driver):
    """traverse should return entities within specified depth."""
    from treeloom.graph_store import traverse
    result = await traverse("func://s/test.py::start", depth=2)
    assert isinstance(result, list)
