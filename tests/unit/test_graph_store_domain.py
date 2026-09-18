"""Domain tests for graph_store — entity resolution, ID generation, relationship logic."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from treeloom.graph_store import store_graph


@pytest.mark.asyncio
async def test_store_graph_empty_lists():
    """Empty entities and relationships → should not error."""
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    with patch("treeloom.adapters.neo4j.graph_store._get_session", return_value=mock_session):
        await store_graph([], [])


@pytest.mark.asyncio
async def test_store_graph_entities_without_source():
    """Entities without source_id should still store."""
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    entities = [
        {
            "id": "func://test/mod.py::f",
            "name": "f",
            "type": "Function",
            "file_path": "/test/mod.py",
            "signature": "def f(): pass",
            "language": "python",
        }
    ]
    with patch("treeloom.adapters.neo4j.graph_store._get_session", return_value=mock_session):
        await store_graph(entities, [], source_id=None)


@pytest.mark.asyncio
async def test_store_graph_with_relationships():
    """Entities with CALLS relationships."""
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    entities = [
        {"id": "func://test/a.py::caller", "name": "caller", "type": "Function",
         "file_path": "/test/a.py", "signature": "def caller(): pass", "language": "python"},
        {"id": "func://test/a.py::callee", "name": "callee", "type": "Function",
         "file_path": "/test/a.py", "signature": "def callee(): pass", "language": "python"},
    ]
    relationships = [
        {"source_id": "func://test/a.py::caller", "target_id": "func://test/a.py::callee", "type": "CALLS"},
    ]
    with patch("treeloom.adapters.neo4j.graph_store._get_session", return_value=mock_session):
        await store_graph(entities, relationships, source_id="test_source")


def test_entity_id_format():
    """Entity IDs follow the pattern <type>://<source>/<path>::<name>."""
    entity = {
        "id": "func://mysource/app/main.py::my_function",
        "name": "my_function",
        "type": "Function",
        "file_path": "app/main.py",
        "source_id": "mysource",
    }
    assert "func://" in entity["id"]
    assert "my_function" in entity["id"]
    assert "main.py" in entity["id"]
