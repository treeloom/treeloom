"""OpenAI embedding adapter — batching, ordering, truncation, retry."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from treeloom.adapters.openai_api import embedding_adapter as oa


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(oa, "_client", None)
    monkeypatch.setattr(oa, "_semaphore", None)
    monkeypatch.setattr(oa, "EMBEDDING_API_KEY", "test-key")
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
    monkeypatch.setattr(oa, "_get_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_empty_input_no_http(monkeypatch):
    client = _install_client(monkeypatch, [])
    assert await oa.embed([]) == []
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_order_preserved_via_index_field(monkeypatch):
    # API returns out-of-order data entries — adapter must sort by index.
    client = _install_client(monkeypatch, [
        _resp(200, [
            {"index": 1, "embedding": [1.0]},
            {"index": 0, "embedding": [0.0]},
        ]),
    ])
    out = await oa.embed(["a", "b"])
    assert out == [[0.0], [1.0]]
    body = client.post.await_args.kwargs["json"]
    assert body["input"] == ["a", "b"] and body["model"] == oa.EMBEDDING_MODEL


@pytest.mark.asyncio
async def test_batching_by_count(monkeypatch):
    monkeypatch.setattr(oa, "MAX_BATCH_SIZE", 2)
    client = _install_client(monkeypatch, [
        _resp(200, [{"index": 0, "embedding": [i]}, {"index": 1, "embedding": [i + 1]}])
        for i in (0, 2)
    ])
    out = await oa.embed(["a", "b", "c", "d"])
    assert client.post.await_count == 2
    assert len(out) == 4


@pytest.mark.asyncio
async def test_oversized_input_truncated(monkeypatch):
    monkeypatch.setattr(oa, "EMBED_MAX_TOKENS_PER_INPUT", 4)
    client = _install_client(
        monkeypatch, [_resp(200, [{"index": 0, "embedding": [1.0]}])]
    )
    await oa.embed(["one two three four five six seven eight nine ten"])
    sent = client.post.await_args.kwargs["json"]["input"][0]
    assert len(oa._get_encoder().encode(sent)) <= 4


@pytest.mark.asyncio
async def test_empty_string_becomes_space(monkeypatch):
    client = _install_client(
        monkeypatch, [_resp(200, [{"index": 0, "embedding": [1.0]}])]
    )
    await oa.embed([""])
    assert client.post.await_args.kwargs["json"]["input"] == [" "]


@pytest.mark.asyncio
async def test_429_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _install_client(monkeypatch, [
        _resp(429, text="rate limited"),
        _resp(200, [{"index": 0, "embedding": [1.0]}]),
    ])
    out = await oa.embed(["a"])
    assert out == [[1.0]] and client.post.await_count == 2


@pytest.mark.asyncio
async def test_400_fails_immediately(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _install_client(monkeypatch, [_resp(400, text="bad request")])
    with pytest.raises(httpx.HTTPStatusError):
        await oa.embed(["a"])
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_embed_query_applies_prefix(monkeypatch):
    monkeypatch.setattr(oa, "EMBEDDING_QUERY_PREFIX", "query: ")
    client = _install_client(
        monkeypatch, [_resp(200, [{"index": 0, "embedding": [1.0]}])]
    )
    await oa.embed_query(["find foo"])
    assert client.post.await_args.kwargs["json"]["input"] == ["query: find foo"]


def test_missing_key_fails_loud(monkeypatch):
    monkeypatch.setattr(oa, "EMBEDDING_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        oa._get_client()
