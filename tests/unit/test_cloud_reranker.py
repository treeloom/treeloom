"""Cloud (Voyage) reranker adapter — request shape, index mapping, retry."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from treeloom.adapters.rerank import cloud_reranker as cr


@pytest.fixture(autouse=True)
def _voyage_env(monkeypatch):
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "voyage")
    monkeypatch.setattr(cr, "_client", None)
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    for v in ("VOYAGE_API_KEY", "COHERE_API_KEY", "JINA_API_KEY", "RERANKER_MODEL"):
        monkeypatch.delenv(v, raising=False)
    yield


def _resp(status: int, data: list[dict] | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = text
    if data is not None:
        resp.json.return_value = {"data": data}
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _install_client(monkeypatch, responses: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(side_effect=responses)
    monkeypatch.setattr(cr, "_get_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_empty_texts_no_http(monkeypatch):
    client = _install_client(monkeypatch, [])
    assert await cr.rerank_texts("q", []) == []
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_scores_pair_with_indices_and_request_shape(monkeypatch):
    client = _install_client(monkeypatch, [
        _resp(200, [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]),
    ])
    out = await cr.rerank_texts("find the auth handler", ["a", "b", "c"])
    assert out == [(0, 0.2), (1, 0.9), (2, 0.5)]
    body = client.post.await_args.kwargs["json"]
    assert body == {
        "model": "rerank-2.5",
        "query": "find the auth handler",
        "documents": ["a", "b", "c"],
        "truncation": True,
        "return_documents": False,
    }
    url = client.post.await_args.args[0]
    assert url == "https://api.voyageai.com/v1/rerank"


@pytest.mark.asyncio
async def test_lite_model_via_env(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "rerank-2.5-lite")
    client = _install_client(monkeypatch, [
        _resp(200, [{"index": 0, "relevance_score": 0.7}]),
    ])
    await cr.rerank_texts("q", ["a"])
    assert client.post.await_args.kwargs["json"]["model"] == "rerank-2.5-lite"


@pytest.mark.asyncio
async def test_chunking_maps_local_indices_to_global(monkeypatch):
    # cap the per-request size to force two requests over 3 docs
    monkeypatch.setitem(cr._PROVIDERS["voyage"], "max_docs", 2)
    client = _install_client(monkeypatch, [
        _resp(200, [{"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1}]),   # docs a,b
        _resp(200, [{"index": 0, "relevance_score": 0.5}]),   # doc c
    ])
    out = await cr.rerank_texts("q", ["a", "b", "c"])
    # second chunk's local index 0 maps to global 2
    assert sorted(out) == [(0, 0.1), (1, 0.9), (2, 0.5)]
    assert client.post.await_count == 2
    assert client.post.await_args_list[1].kwargs["json"]["documents"] == ["c"]


@pytest.mark.asyncio
async def test_429_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _install_client(monkeypatch, [
        _resp(429, text="rate limited"),
        _resp(200, [{"index": 0, "relevance_score": 0.8}]),
    ])
    out = await cr.rerank_texts("q", ["a"])
    assert out == [(0, 0.8)] and client.post.await_count == 2


@pytest.mark.asyncio
async def test_400_fails_immediately(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _install_client(monkeypatch, [_resp(400, text="bad model")])
    with pytest.raises(httpx.HTTPStatusError):
        await cr.rerank_texts("q", ["a"])
    assert client.post.await_count == 1


def test_missing_key_fails_loud(monkeypatch):
    monkeypatch.delenv("RERANKER_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setattr(cr, "_client", None)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        cr._get_client()


@pytest.mark.asyncio
async def test_zeroentropy_request_shape_and_results_key(monkeypatch):
    """ZeroEntropy: 'results' array, {model,query,documents}, optional latency."""
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "zeroentropy")
    monkeypatch.delenv("RERANKER_LATENCY", raising=False)
    client = MagicMock()
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"results": [
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.3},
    ]}
    resp.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(cr, "_get_client", lambda: client)

    out = await cr.rerank_texts("q", ["a", "b", "c"])
    assert sorted(out) == [(0, 0.3), (2, 0.95)]
    assert client.post.await_args.args[0] == "https://api.zeroentropy.dev/v1/models/rerank"
    body = client.post.await_args.kwargs["json"]
    assert body == {"model": "zerank-2", "query": "q", "documents": ["a", "b", "c"]}
    assert "latency" not in body


@pytest.mark.asyncio
async def test_zeroentropy_latency_mode_included_when_set(monkeypatch):
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "zeroentropy")
    monkeypatch.setenv("RERANKER_LATENCY", "fast")
    client = MagicMock()
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
    resp.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(cr, "_get_client", lambda: client)
    await cr.rerank_texts("q", ["a"])
    assert client.post.await_args.kwargs["json"]["latency"] == "fast"


def test_active_provider_key_wins_over_other_providers(monkeypatch):
    """Regression: with multiple provider keys set, the ACTIVE provider's key
    must be chosen — not a fixed precedence (the Cohere-using-Voyage-key bug)."""
    monkeypatch.delenv("RERANKER_API_KEY", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")
    monkeypatch.setenv("COHERE_API_KEY", "cohere-key")
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "cohere")
    assert cr._api_key() == "cohere-key"
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "voyage")
    assert cr._api_key() == "voyage-key"
    # explicit generic override beats both
    monkeypatch.setenv("RERANKER_API_KEY", "explicit")
    assert cr._api_key() == "explicit"


@pytest.mark.asyncio
async def test_unsupported_provider_fails_loud(monkeypatch):
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "jina")  # stubbed, not impl'd
    with pytest.raises(RuntimeError, match="no cloud-rerank implementation"):
        await cr.rerank_texts("q", ["a"])


@pytest.mark.asyncio
async def test_cohere_request_shape_and_results_key(monkeypatch):
    """Cohere keys its array 'results' and omits Voyage's truncation field."""
    monkeypatch.setattr(cr, "RERANKER_PROVIDER", "cohere")
    client = MagicMock()
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"results": [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]}
    resp.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(cr, "_get_client", lambda: client)

    out = await cr.rerank_texts("find auth", ["a", "b"])
    assert sorted(out) == [(0, 0.2), (1, 0.9)]
    assert client.post.await_args.args[0] == "https://api.cohere.com/v2/rerank"
    body = client.post.await_args.kwargs["json"]
    assert body == {"model": "rerank-v3.5", "query": "find auth",
                    "documents": ["a", "b"]}
    assert "truncation" not in body and "return_documents" not in body
