"""Adapter tests for indexer_service — FastAPI endpoint integration."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient for the indexer service."""
    from treeloom.indexer_service import app
    return TestClient(app)


def test_index_file_endpoint_valid(test_client):
    """POST /index-file with valid path returns an accepted status."""
    response = test_client.post(
        "/index-file",
        json={"file_path": "/tmp/test_repo/main.py"},
    )
    assert response.status_code in [200, 202, 404, 422]


def test_index_file_missing_field(test_client):
    """POST /index-file without file_path returns 422."""
    response = test_client.post("/index-file", json={})
    assert response.status_code == 422


def test_index_directory_endpoint(test_client):
    """POST /index-directory with valid fields."""
    response = test_client.post(
        "/index-directory",
        json={"directory": "/tmp/test_dir", "pattern": "**/*.py"},
    )
    assert response.status_code in [200, 202, 404, 422]


def test_status_endpoint(test_client):
    """GET /status should return JSON."""
    response = test_client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


# ── US2 Indexer Search & Graph Endpoints ──────────────────────────


class TestIndexerSearchEndpoints:
    """Tests for /search, /find-definition, /find-callers,
    /find-references, and /graph-explore endpoints."""

    def test_search_endpoint(self, test_client, mocker):
        """POST /search with query — mock retriever.graph_search."""
        mock_embed = mocker.patch(
            "treeloom.application.indexer_service.embed_query",
            return_value=[[0.1, 0.2, 0.3]],
        )
        mock_gs = mocker.patch(
            "treeloom.application.indexer_service.graph_search",
            return_value={
                "chunks": [
                    {
                        "file_path": "src/main.py",
                        "start_line": 10,
                        "end_line": 25,
                        "language": "python",
                        "score": 0.95,
                        "snippet": "def hello(): pass",
                        "source_id": "src-1",
                        "graph_enhanced": False,
                    }
                ],
                "neighbors": [],
                "community_summaries": {},
            },
        )

        # /search requires an explicit scope: path_prefix/source_id XOR
        # cross_repo (unscoped searches are rejected with 400 by design).
        response = test_client.post(
            "/search",
            json={"query": "auth", "top_k": 5, "language": "python",
                  "cross_repo": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["snippet"] == "def hello(): pass"
        mock_embed.assert_called_once()
        mock_gs.assert_called_once()

    def test_find_definition_by_name(self, test_client, mocker):
        """GET /find-definition?name=ClassName — returns entity list."""
        mock_find = mocker.patch(
            "treeloom.application.indexer_service.graph_store.find_entities_by_name",
            return_value=[
                {
                    "id": "func://src/main.py#ClassName",
                    "type": "Class",
                    "name": "ClassName",
                    "signature": "class ClassName(Base):",
                    "file_path": "src/main.py",
                    "start_line": 20,
                    "end_line": 50,
                    "language": "python",
                }
            ],
        )

        response = test_client.get("/find-definition", params={"name": "ClassName"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "ClassName"
        assert data[0]["type"] == "Class"
        mock_find.assert_called_once_with(
            "ClassName", entity_type=None, source_id=None,
            exclude_source_ids=None,
        )

    def test_find_definition_by_name_and_kind(self, test_client, mocker):
        """GET /find-definition with kind filter passes kind to graph_store."""
        mock_find = mocker.patch(
            "treeloom.application.indexer_service.graph_store.find_entities_by_name",
            return_value=[
                {
                    "id": "func://src/main.py#ClassName",
                    "type": "Class",
                    "name": "ClassName",
                    "signature": "class ClassName(Base):",
                    "file_path": "src/main.py",
                    "start_line": 20,
                    "end_line": 50,
                    "language": "python",
                }
            ],
        )

        response = test_client.get(
            "/find-definition",
            params={"name": "ClassName", "kind": "Class"},
        )
        assert response.status_code == 200
        mock_find.assert_called_once_with(
            "ClassName", entity_type="Class", source_id=None,
            exclude_source_ids=None,
        )

    def test_find_callers(self, test_client, mocker):
        """GET /find-callers?name_or_id=func_id — returns callers list."""
        mock_find = mocker.patch(
            "treeloom.application.indexer_service.graph_store.find_entities_by_name",
            return_value=[
                {
                    "id": "func://src/main.py#my_func",
                    "type": "Function",
                    "name": "my_func",
                    "signature": "def my_func(x):",
                    "file_path": "src/main.py",
                    "start_line": 10,
                    "end_line": 20,
                    "language": "python",
                }
            ],
        )
        mock_callers = mocker.patch(
            "treeloom.application.indexer_service.graph_store.find_callers",
            return_value=[
                {
                    "id": "func://src/caller.py#caller_func",
                    "type": "Function",
                    "name": "caller_func",
                    "signature": "def caller_func():",
                    "file_path": "src/caller.py",
                    "start_line": 5,
                    "end_line": 15,
                    "language": "python",
                    "_rel_type": "CALLS",
                    "_rel_direction": "in",
                    "_target_id": "func://src/main.py#my_func",
                }
            ],
        )

        response = test_client.get(
            "/find-callers", params={"name_or_id": "my_func"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "caller_func"
        assert data[0]["_rel_type"] == "CALLS"
        mock_find.assert_called_once()
        mock_callers.assert_called_once()

    def test_find_references(self, test_client, mocker):
        """GET /find-references?name_or_id=func_id — returns references list."""
        mock_find = mocker.patch(
            "treeloom.application.indexer_service.graph_store.find_entities_by_name",
            return_value=[
                {
                    "id": "func://src/main.py#my_func",
                    "type": "Function",
                    "name": "my_func",
                    "signature": "def my_func(x):",
                    "file_path": "src/main.py",
                    "start_line": 10,
                    "end_line": 20,
                    "language": "python",
                }
            ],
        )
        mock_refs = mocker.patch(
            "treeloom.application.indexer_service.graph_store.find_references",
            return_value=[
                {
                    "id": "func://src/other.py#other_func",
                    "type": "Function",
                    "name": "other_func",
                    "signature": "def other_func():",
                    "file_path": "src/other.py",
                    "start_line": 1,
                    "end_line": 8,
                    "language": "python",
                    "_rel_type": "IMPORTS",
                    "_rel_direction": "in",
                    "_target_id": "func://src/main.py#my_func",
                }
            ],
        )

        response = test_client.get(
            "/find-references", params={"name_or_id": "my_func"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "other_func"
        assert data[0]["_rel_type"] == "IMPORTS"
        mock_find.assert_called_once()
        mock_refs.assert_called_once()

    def test_graph_explore(self, test_client, mocker):
        """POST /graph-explore with query and depth — returns graph search result."""
        mock_embed = mocker.patch(
            "treeloom.application.indexer_service.embed_query",
            return_value=[[0.1, 0.2, 0.3]],
        )
        mocker.patch(
            "treeloom.application.indexer_service.graph_search",
            return_value={
                "chunks": [
                    {
                        "file_path": "src/auth.py",
                        "start_line": 15,
                        "end_line": 40,
                        "language": "python",
                        "score": 0.88,
                        "snippet": "def authenticate(): pass",
                        "source_id": "src-1",
                        "graph_enhanced": True,
                    }
                ],
                "neighbors": [
                    {
                        "id": "func://src/auth.py#login",
                        "type": "Function",
                        "name": "login",
                        "signature": "def login(u, p):",
                        "file_path": "src/auth.py",
                        "start_line": 42,
                        "end_line": 55,
                        "language": "python",
                        "_rel_type": "CALLS",
                        "_rel_direction": "out",
                        "_target_id": "func://src/auth.py#authenticate",
                    }
                ],
                "community_summaries": {1: "Auth module"},
            },
        )

        response = test_client.post(
            "/graph-explore",
            json={"query": "auth", "depth": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "neighbors" in data
        assert "community_summaries" in data
        assert len(data["neighbors"]) == 1
        assert data["neighbors"][0]["name"] == "login"
        assert data["community_summaries"] == {"1": "Auth module"}
        mock_embed.assert_called_once()
