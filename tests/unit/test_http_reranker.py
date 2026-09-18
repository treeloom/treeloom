"""HTTP reranker adapter — TEI /rerank protocol, batching, index mapping."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from treeloom.adapters.rerank import http_reranker


def _mock_client(responses: list[list[dict]]):
    """A reused-AsyncClient mock returning canned /rerank JSON per POST."""
    client = MagicMock()
    posts = []
    for payload in responses:
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        posts.append(resp)
    client.post = AsyncMock(side_effect=posts)
    return client


@pytest.mark.asyncio
async def test_empty_texts_no_http():
    assert await http_reranker.rerank_texts("q", []) == []


@pytest.mark.asyncio
async def test_scores_map_to_global_indexes(monkeypatch):
    monkeypatch.setattr(http_reranker, "RERANKER_URL", "http://r")
    client = _mock_client([
        [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}],
        [{"index": 0, "score": 0.5}],
    ])
    with patch.object(http_reranker, "_get_client", return_value=client):
        out = await http_reranker.rerank_texts("q", ["a", "b", "c"], batch_size=2)
    # batch 0 covers texts[0:2]; batch 1 covers texts[2:] at offset 2. Batches
    # run concurrently, so assert on the order-independent result set.
    assert sorted(out) == [(0, 0.1), (1, 0.9), (2, 0.5)]
    assert client.post.await_count == 2
    bodies = [c.kwargs["json"] for c in client.post.await_args_list]
    assert {"query": "q", "texts": ["a", "b"], "raw_scores": True} in bodies
    assert {"query": "q", "texts": ["c"], "raw_scores": True} in bodies


@pytest.mark.asyncio
async def test_missing_url_fails_loud(monkeypatch):
    monkeypatch.setattr(http_reranker, "RERANKER_URL", "")
    with pytest.raises(RuntimeError, match="RERANKER_URL"):
        await http_reranker.rerank_texts("q", ["a"])
