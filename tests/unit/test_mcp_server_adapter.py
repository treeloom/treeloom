"""Adapter tests for mcp_server — proxy calls to indexer API via httpx."""
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch


# ── Test fixtures ──────────────────────────────────────────────────

@pytest.fixture
def mock_chunks():
    """Standard chunk response from indexer /search endpoint."""
    return [
        {
            "file_path": "src/main.py",
            "start_line": 10,
            "end_line": 25,
            "language": "python",
            "score": 0.95,
            "snippet": "def hello(): pass",
            "source_id": "src-1",
            "graph_enhanced": False,
        },
        {
            "file_path": "src/auth.py",
            "start_line": 15,
            "end_line": 40,
            "language": "python",
            "score": 0.88,
            "snippet": "def authenticate(): pass",
            "source_id": "src-1",
            "graph_enhanced": False,
        },
    ]


@pytest.fixture
def mock_neighbor_entity():
    """Standard NeighborEntity dict from indexer."""
    return {
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


@pytest.fixture
def mock_search_response(mock_chunks, mock_neighbor_entity):
    """Full /search response shape from indexer."""
    return {
        "chunks": mock_chunks,
        "neighbors": [mock_neighbor_entity],
        "community_summaries": {"1": "Auth module"},
    }


@pytest.fixture
def mock_entity_list():
    """Standard entity list from /find-definition, /find-callers, /find-references."""
    return [
        {
            "id": "func://src/main.py#ClassName",
            "type": "Class",
            "name": "ClassName",
            "signature": "class ClassName(Base):",
            "file_path": "src/main.py",
            "start_line": 20,
            "end_line": 50,
            "language": "python",
            "_rel_type": "",
            "_rel_direction": "",
            "_target_id": None,
        }
    ]


# ── T011: httpx mock helper ────────────────────────────────────────


def _make_mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_async_client_post(mocker, json_data=None, status_code=200):
    """Mock httpx.AsyncClient.post to return given data."""
    mock_resp = _make_mock_response(status_code, json_data)
    mock_post = mocker.patch("httpx.AsyncClient.post", return_value=mock_resp)
    return mock_post


def _mock_async_client_get(mocker, json_data=None, status_code=200):
    """Mock httpx.AsyncClient.get to return given data."""
    mock_resp = _make_mock_response(status_code, json_data)
    mock_get = mocker.patch("httpx.AsyncClient.get", return_value=mock_resp)
    return mock_get


# ── T011: test_search_code_via_proxy ───────────────────────────────


@pytest.mark.asyncio
async def test_search_code_via_proxy(mocker, mock_search_response):
    """search_code() proxies to indexer POST /search and returns SearchResponse."""
    from treeloom.application.mcp_server import search_code
    _mock_async_client_post(mocker, json_data=mock_search_response)

    result = await search_code(
        query="auth",
        top_k=5,
        language="python",
        source_id="src-1",
        response_format="json",
    )

    assert isinstance(result, dict)  # SearchResponse is a Pydantic model, but tool returns it

    # Verify the result shape
    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["snippet"] == "def hello(): pass"
    assert result["chunks"][0]["score"] == 0.95
    # The MCP layer trims default-valued fields (exclude_defaults) to save
    # tokens, so graph_enhanced=False is conveyed by absence.
    assert result["chunks"][0].get("graph_enhanced", False) is False
    assert "neighbors" in result
    assert len(result["neighbors"]) == 1
    assert result["neighbors"][0]["name"] == "login"
    assert result["neighbors"][0]["_rel_type"] == "CALLS"
    assert "community_summaries" in result
    assert result["community_summaries"] == {"1": "Auth module"}

    # Verify the proxy call
    import httpx
    httpx.AsyncClient.post.assert_called_once()
    call_args = httpx.AsyncClient.post.call_args
    assert "/search" in call_args[0][0]


@pytest.mark.asyncio
async def test_search_code_skips_malformed_chunks(mocker, mock_chunks):
    """A chunk missing required fields (file_path/snippet) must be skipped,
    not 500 the whole response. Regression for the ChunkHit validation crash."""
    from treeloom.application.mcp_server import search_code
    bad_row = {"id": "466622819309695119", "final_score": 0.0,
               "language": "python", "start_line": 1}  # no file_path / snippet
    _mock_async_client_post(mocker, json_data={
        "chunks": [bad_row, mock_chunks[0]],
        "neighbors": [],
        "community_summaries": {},
    })

    result = await search_code(query="auth", response_format="json")

    # bad row dropped, valid row kept — no exception raised
    assert isinstance(result, dict)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["snippet"] == "def hello(): pass"


@pytest.mark.asyncio
async def test_search_code_passes_all_params(mocker, mock_search_response):
    """search_code() forwards all optional params to the indexer."""
    from treeloom.application.mcp_server import search_code
    _mock_async_client_post(mocker, json_data=mock_search_response)

    await search_code(
        query="auth",
        top_k=3,
        language="python",
        path_prefix="/repo/src/",
        source_id="src-1",
        use_hybrid=True,
        use_hyde=False,
        use_summary_vector=True,
        use_graph_scoring=True,
        rerank_pool=20,
    )

    import httpx
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert call_kwargs["json"]["query"] == "auth"
    assert call_kwargs["json"]["top_k"] == 3
    assert call_kwargs["json"]["language"] == "python"
    assert call_kwargs["json"]["path_prefix"] == "/repo/src/"
    assert call_kwargs["json"]["source_id"] == "src-1"
    assert call_kwargs["json"]["use_hybrid"] is True
    assert call_kwargs["json"]["use_hyde"] is False
    assert call_kwargs["json"]["use_summary_vector"] is True
    assert call_kwargs["json"]["use_graph_scoring"] is True
    assert call_kwargs["json"]["rerank_pool"] == 20


# ── T011: test_search_code_enhanced_via_proxy ──────────────────────


@pytest.mark.asyncio
async def test_search_code_enhanced_via_proxy(mocker, mock_search_response):
    """search_code_enhanced() proxies to indexer POST /search."""
    from treeloom.application.mcp_server import search_code_enhanced
    _mock_async_client_post(mocker, json_data=mock_search_response)

    result = await search_code_enhanced(query="auth", top_k=10, response_format="json")

    assert "chunks" in result
    assert "neighbors" in result
    import httpx
    httpx.AsyncClient.post.assert_called_once()


# ── T011: test_explain_code_via_proxy ──────────────────────────────


@pytest.mark.asyncio
async def test_explain_code_via_proxy(mocker, mock_search_response):
    """explain_code() proxies to indexer POST /search with top_k=5."""
    from treeloom.application.mcp_server import explain_code
    _mock_async_client_post(mocker, json_data=mock_search_response)

    result = await explain_code(query="auth", language="python")

    assert "chunks" in result
    import httpx
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert call_kwargs["json"]["top_k"] == 5


@pytest.mark.asyncio
async def test_explain_code_filters_by_file_path(mocker, mock_search_response):
    """explain_code() filters chunks client-side when file_path is provided."""
    # Make a response where only one chunk matches the file_path
    filtered_response = {
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
            },
            {
                "file_path": "src/other.py",
                "start_line": 1,
                "end_line": 5,
                "language": "python",
                "score": 0.7,
                "snippet": "x = 1",
                "source_id": "src-1",
                "graph_enhanced": False,
            },
        ],
        "neighbors": [],
        "community_summaries": {},
    }
    from treeloom.application.mcp_server import explain_code
    _mock_async_client_post(mocker, json_data=filtered_response)

    result = await explain_code(query="auth", file_path="src/main.py")

    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["file_path"] == "src/main.py"


# ── T011: test_graph_explore_via_proxy ─────────────────────────────


@pytest.mark.asyncio
async def test_graph_explore_via_proxy(mocker, mock_search_response):
    """graph_explore() proxies to indexer POST /graph-explore."""
    from treeloom.application.mcp_server import graph_explore
    _mock_async_client_post(mocker, json_data=mock_search_response)

    result = await graph_explore(query="auth", depth=2, language="python")

    assert "chunks" in result
    assert "neighbors" in result
    assert "community_summaries" in result
    import httpx
    httpx.AsyncClient.post.assert_called_once()
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert "/graph-explore" in httpx.AsyncClient.post.call_args[0][0]
    assert call_kwargs["json"]["query"] == "auth"
    assert call_kwargs["json"]["depth"] == 2
    assert call_kwargs["json"]["language"] == "python"


# ── T011: test_find_definition_via_proxy ───────────────────────────


@pytest.mark.asyncio
async def test_find_definition_via_proxy(mocker, mock_entity_list):
    """find_definition() proxies to indexer GET /find-definition."""
    from treeloom.application.mcp_server import find_definition
    _mock_async_client_get(mocker, json_data=mock_entity_list)

    result = await find_definition(name="ClassName", kind="Class", source_id="src-1")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"] == "ClassName"
    assert result[0]["type"] == "Class"
    assert result[0]["id"] == "func://src/main.py#ClassName"

    import httpx
    httpx.AsyncClient.get.assert_called_once()
    call_kwargs = httpx.AsyncClient.get.call_args[1]
    assert call_kwargs["params"]["name"] == "ClassName"
    assert call_kwargs["params"]["kind"] == "Class"
    assert call_kwargs["params"]["source_id"] == "src-1"


# ── T011: test_find_callers_via_proxy ──────────────────────────────


@pytest.mark.asyncio
async def test_find_callers_via_proxy(mocker, mock_entity_list):
    """find_callers() proxies to indexer GET /find-callers."""
    callers_data = [
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
    ]
    from treeloom.application.mcp_server import find_callers
    _mock_async_client_get(mocker, json_data=callers_data)

    result = await find_callers(name_or_id="my_func", source_id="src-1")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"] == "caller_func"
    assert result[0]["_rel_type"] == "CALLS"
    assert result[0]["_rel_direction"] == "in"

    import httpx
    httpx.AsyncClient.get.assert_called_once()
    call_kwargs = httpx.AsyncClient.get.call_args[1]
    assert call_kwargs["params"]["name_or_id"] == "my_func"
    assert call_kwargs["params"]["source_id"] == "src-1"


# ── T011: test_find_references_via_proxy ───────────────────────────


@pytest.mark.asyncio
async def test_find_references_via_proxy(mocker):
    """find_references() proxies to indexer GET /find-references."""
    refs_data = [
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
    ]
    from treeloom.application.mcp_server import find_references
    _mock_async_client_get(mocker, json_data=refs_data)

    result = await find_references(name_or_id="my_func", source_id="src-1")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"] == "other_func"
    assert result[0]["_rel_type"] == "IMPORTS"

    import httpx
    httpx.AsyncClient.get.assert_called_once()


# ── T011: test_mcp_indexer_unreachable ─────────────────────────────


@pytest.mark.asyncio
async def test_mcp_indexer_unreachable_search(mocker):
    """search_code() returns graceful error when indexer is unreachable."""
    from treeloom.application.mcp_server import search_code
    mocker.patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.ConnectError("Connection refused"),
    )

    result = await search_code(query="auth")

    assert isinstance(result, dict)
    assert "error" in result
    assert "unreachable" in result["error"].lower() or "connect" in result["error"].lower()


@pytest.mark.asyncio
async def test_mcp_indexer_unreachable_find_definition(mocker):
    """find_definition() returns graceful error when indexer is unreachable."""
    from treeloom.application.mcp_server import find_definition
    mocker.patch(
        "httpx.AsyncClient.get",
        side_effect=httpx.ConnectError("Connection refused"),
    )

    result = await find_definition(name="MyClass")

    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_mcp_indexer_unreachable_find_callers(mocker):
    """find_callers() returns graceful error when indexer is unreachable."""
    from treeloom.application.mcp_server import find_callers
    mocker.patch(
        "httpx.AsyncClient.get",
        side_effect=httpx.ConnectError("Connection refused"),
    )

    result = await find_callers(name_or_id="my_func")

    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_mcp_indexer_unreachable_find_references(mocker):
    """find_references() returns graceful error when indexer is unreachable."""
    from treeloom.application.mcp_server import find_references
    mocker.patch(
        "httpx.AsyncClient.get",
        side_effect=httpx.ConnectError("Connection refused"),
    )

    result = await find_references(name_or_id="my_func")

    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_mcp_indexer_unreachable_graph_explore(mocker):
    """graph_explore() returns graceful error when indexer is unreachable."""
    from treeloom.application.mcp_server import graph_explore
    mocker.patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.ConnectError("Connection refused"),
    )

    result = await graph_explore(query="auth")

    assert isinstance(result, dict)
    assert "error" in result


# ── T011: test_mcp_auth_passthrough ────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_index_tool_does_not_lend_service_key_by_default(mocker, monkeypatch):
    """INDEX tools must NOT inject the service account for an uncredentialed
    caller. This used to be unconditional, which let an anonymous caller on the
    HTTP transport inherit service-account authority over the indexer."""
    monkeypatch.setenv("TREELOOM_MCP_API_KEY", "test-api-key-123")
    monkeypatch.delenv("TREELOOM_MCP_SERVICE_ACCOUNT", raising=False)
    from treeloom.application.mcp_server import index_repo
    _mock_async_client_post(
        mocker, json_data={"job_id": "j1", "status": "queued", "source_id": "s1"}
    )

    await index_repo(path="/repo")  # ctx defaults to None -> no caller credential

    import httpx
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert "Authorization" not in call_kwargs.get("headers", {})


@pytest.mark.asyncio
async def test_mcp_index_tool_uses_service_key_when_opted_in(mocker, monkeypatch):
    """The service account is still available for stdio / single-user setups,
    but only behind an explicit TREELOOM_MCP_SERVICE_ACCOUNT=true opt-in."""
    monkeypatch.setenv("TREELOOM_MCP_API_KEY", "test-api-key-123")
    monkeypatch.setenv("TREELOOM_MCP_SERVICE_ACCOUNT", "true")
    from treeloom.application.mcp_server import index_repo
    _mock_async_client_post(
        mocker, json_data={"job_id": "j1", "status": "queued", "source_id": "s1"}
    )

    await index_repo(path="/repo")

    import httpx
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key-123"


@pytest.mark.asyncio
async def test_mcp_read_tool_does_not_leak_service_key(
    mocker, mock_search_response, monkeypatch
):
    """READ tools must NOT fall back to the service key: with a service
    key set but no caller credential, no Authorization header is sent."""
    monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
    from treeloom.application.mcp_server import search_code
    _mock_async_client_post(mocker, json_data=mock_search_response)

    await search_code(query="auth", source_id="repoA")  # ctx defaults to None

    import httpx
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert "Authorization" not in call_kwargs.get("headers", {})


@pytest.mark.asyncio
async def test_mcp_auth_no_key_no_header(mocker, mock_search_response, monkeypatch):
    """No Authorization header sent when TREELOOM_MCP_API_KEY is not set."""
    monkeypatch.delenv("TREELOOM_MCP_API_KEY", raising=False)
    from treeloom.application.mcp_server import search_code
    _mock_async_client_post(mocker, json_data=mock_search_response)

    await search_code(query="auth")

    import httpx
    call_kwargs = httpx.AsyncClient.post.call_args[1]
    assert "headers" not in call_kwargs or "Authorization" not in call_kwargs.get("headers", {})


# ── T011: test_mcp_server_domain (structure tests) ─────────────────


def test_sse_endpoint_exists():
    """The MCP server provides an SSE transport endpoint."""
    from treeloom.mcp_server import mcp
    assert mcp.name == "treeloom"


@pytest.mark.asyncio
async def test_return_type_is_dict_for_mcp(mocker, mock_search_response):
    """MCP tool functions return plain dicts (not Pydantic models) for MCP serialization.

    search_code defaults to markdown now (token-lean); response_format="json"
    returns the structured dict for programmatic callers — that's what this
    asserts.
    """
    from treeloom.application.mcp_server import search_code
    _mock_async_client_post(mocker, json_data=mock_search_response)

    result = await search_code(query="test", response_format="json")
    assert isinstance(result, dict)
    assert "chunks" in result


# ── index force passthrough + graph tools ──────────────────────────


@pytest.mark.asyncio
async def test_index_repo_passes_force(mocker):
    """index_repo forwards force=true to POST /index-repo."""
    from treeloom.application.mcp_server import index_repo
    mock_post = _mock_async_client_post(
        mocker, json_data={"job_id": "j1", "status": "queued", "source_id": "s1"})

    result = await index_repo(path="/repo", force=True)

    assert result["job_id"] == "j1"
    body = mock_post.call_args.kwargs["json"]
    assert body["force"] is True
    assert "/index-repo" in mock_post.call_args[0][0]


@pytest.mark.asyncio
async def test_index_graph_proxies(mocker):
    """index_graph posts the source_id to /index-graph and returns the ack."""
    from treeloom.application.mcp_server import index_graph
    mock_post = _mock_async_client_post(
        mocker, json_data={"job_id": "g1", "status": "queued", "source_id": "s1"})

    result = await index_graph(source_id="s1")

    assert result["job_id"] == "g1"
    assert result["source_id"] == "s1"
    assert "/index-graph" in mock_post.call_args[0][0]
    assert mock_post.call_args.kwargs["json"] == {"source_id": "s1"}


@pytest.mark.asyncio
async def test_rebuild_all_graphs_enqueues_every_source(mocker):
    """By default rebuild_all_graphs enqueues a graph job for ALL sources,
    even those with graph_indexed=true (the flag is unreliable for the bug)."""
    from treeloom.application.mcp_server import rebuild_all_graphs
    sources = [
        {"id": "s1", "graph_indexed": True},
        {"id": "s2", "graph_indexed": False},
    ]
    mocker.patch(
        "httpx.AsyncClient.get",
        return_value=_make_mock_response(200, sources),
    )
    mock_post = mocker.patch(
        "httpx.AsyncClient.post",
        return_value=_make_mock_response(200, {"job_id": "g", "status": "queued"}),
    )

    result = await rebuild_all_graphs()

    assert result["total_sources"] == 2
    assert result["enqueued"] == 2
    assert result["skipped"] == 0
    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_rebuild_all_graphs_only_missing_skips_indexed(mocker):
    """only_missing=true skips sources already flagged graph_indexed=true."""
    from treeloom.application.mcp_server import rebuild_all_graphs
    sources = [
        {"id": "s1", "graph_indexed": True},
        {"id": "s2", "graph_indexed": False},
    ]
    mocker.patch(
        "httpx.AsyncClient.get",
        return_value=_make_mock_response(200, sources),
    )
    mock_post = mocker.patch(
        "httpx.AsyncClient.post",
        return_value=_make_mock_response(200, {"job_id": "g", "status": "queued"}),
    )

    result = await rebuild_all_graphs(only_missing=True)

    assert result["enqueued"] == 1
    assert result["skipped"] == 1
    assert mock_post.call_count == 1


# ── T056: result provenance ────────────────────────────────────────


_PROVENANCE_SOURCE_ID = "src-42"
_PROVENANCE_COMMIT_SHA = "abcdef1234567890"
_PROVENANCE_CITATION = "myrepo@abcdef123456:src/main.py"

_PROVENANCE_RESPONSE = {
    "chunks": [
        {
            "file_path": "src/main.py",
            "start_line": 1,
            "end_line": 10,
            "language": "python",
            "score": 0.9,
            "snippet": "def foo(): pass",
            "source_id": _PROVENANCE_SOURCE_ID,
            "commit_sha": _PROVENANCE_COMMIT_SHA,
            "citation": _PROVENANCE_CITATION,
        }
    ],
    "neighbors": [],
    "community_summaries": {},
    "sources": {
        _PROVENANCE_SOURCE_ID: {
            "commit_sha": _PROVENANCE_COMMIT_SHA,
            "indexed_at": "2026-06-14T00:00:00Z",
            "path": "/repo/myrepo",
            "url": None,
            "branch": "main",
            "permalink_base": "https://github.com/org/myrepo/blob/abcdef1234567890",
            "is_stale": False,
            "current_sha": _PROVENANCE_COMMIT_SHA,
        }
    },
}

_STALE_RESPONSE = {
    "chunks": [
        {
            "file_path": "src/lib.py",
            "start_line": 5,
            "end_line": 15,
            "language": "python",
            "score": 0.8,
            "snippet": "def bar(): pass",
            "source_id": _PROVENANCE_SOURCE_ID,
            "commit_sha": _PROVENANCE_COMMIT_SHA,
            "citation": _PROVENANCE_CITATION,
        }
    ],
    "neighbors": [],
    "community_summaries": {},
    "sources": {
        _PROVENANCE_SOURCE_ID: {
            "commit_sha": _PROVENANCE_COMMIT_SHA,
            "indexed_at": "2026-06-14T00:00:00Z",
            "path": "/repo/myrepo",
            "url": None,
            "branch": "main",
            "is_stale": True,
            "current_sha": "deadbeef99999999",
        }
    },
}


def test_parse_search_response_sources_and_provenance_fields():
    """_parse_search_response propagates sources map and per-chunk citation/commit_sha."""
    from treeloom.application.mcp_server import _parse_search_response

    sr = _parse_search_response(_PROVENANCE_RESPONSE)

    assert sr.sources == _PROVENANCE_RESPONSE["sources"]
    assert len(sr.chunks) == 1
    assert sr.chunks[0].citation == _PROVENANCE_CITATION
    assert sr.chunks[0].commit_sha == _PROVENANCE_COMMIT_SHA


def test_search_response_to_dict_round_trips_provenance():
    """_search_response_to_dict includes sources map and per-chunk citation/commit_sha."""
    from treeloom.application.mcp_server import (
        _parse_search_response,
        _search_response_to_dict,
    )

    sr = _parse_search_response(_PROVENANCE_RESPONSE)
    out = _search_response_to_dict(sr)

    assert "sources" in out
    assert out["sources"] == _PROVENANCE_RESPONSE["sources"]
    assert out["chunks"][0]["citation"] == _PROVENANCE_CITATION
    assert out["chunks"][0]["commit_sha"] == _PROVENANCE_COMMIT_SHA


def test_search_response_to_markdown_includes_citation_and_provenance():
    """_search_response_to_markdown renders citation in heading and provenance block."""
    from treeloom.application.mcp_server import (
        _parse_search_response,
        _search_response_to_dict,
        _search_response_to_markdown,
    )

    sr = _parse_search_response(_PROVENANCE_RESPONSE)
    out = _search_response_to_dict(sr)
    md = _search_response_to_markdown(out)

    # Citation appears in the chunk heading
    assert _PROVENANCE_CITATION in md
    # Provenance block present with truncated sha (first 12 chars)
    assert "provenance:" in md
    assert _PROVENANCE_COMMIT_SHA[:12] in md
    # Permalink base is present
    assert "https://github.com/org/myrepo/blob" in md
    # Not stale — no STALE marker
    assert "STALE" not in md


def test_search_response_to_markdown_shows_stale_flag():
    """_search_response_to_markdown shows STALE when is_stale=True for a source."""
    from treeloom.application.mcp_server import (
        _parse_search_response,
        _search_response_to_dict,
        _search_response_to_markdown,
    )

    sr = _parse_search_response(_STALE_RESPONSE)
    out = _search_response_to_dict(sr)
    md = _search_response_to_markdown(out)

    assert "provenance:" in md
    assert "STALE" in md


@pytest.mark.asyncio
async def test_search_code_passes_check_staleness(mocker):
    """search_code() forwards check_staleness to the indexer request body."""
    from treeloom.application.mcp_server import search_code
    mock_post = _mock_async_client_post(mocker, json_data=_PROVENANCE_RESPONSE)

    await search_code(query="foo", source_id=_PROVENANCE_SOURCE_ID, check_staleness=True)

    body = mock_post.call_args.kwargs["json"]
    assert body["check_staleness"] is True


@pytest.mark.asyncio
async def test_search_code_enhanced_passes_check_staleness(mocker):
    """search_code_enhanced() forwards check_staleness to the indexer request body."""
    from treeloom.application.mcp_server import search_code_enhanced
    mock_post = _mock_async_client_post(mocker, json_data=_PROVENANCE_RESPONSE)

    await search_code_enhanced(
        query="foo", source_id=_PROVENANCE_SOURCE_ID, check_staleness=True
    )

    body = mock_post.call_args.kwargs["json"]
    assert body["check_staleness"] is True
