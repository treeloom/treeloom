"""Embedding proxy batch-size handling.

A TEI backend rejects an oversized /embed with HTTP 422
("batch size N > maximum allowed batch size M"). The proxy must split the batch
to the backend's cap and retry on the SAME backend — NOT trip the circuit
breaker (the backend is healthy, the batch is too big) — and learn the cap so
later oversized batches are pre-split.
"""
from __future__ import annotations

import httpx
import pytest

from treeloom.adapters.tei.embedding_proxy import (
    Backend,
    BackendClass,
    EmbeddingProxy,
    _parse_batch_limit,
)


class _FakeResp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


class _FakeClient:
    """A TEI backend that 422s any /embed with more than `max_batch` inputs."""

    def __init__(self, max_batch: int = 32):
        self.max_batch = max_batch
        self.calls: list[int] = []  # input-count of each POST

    async def post(self, url, json=None):
        n = len(json["inputs"])
        self.calls.append(n)
        if n > self.max_batch:
            return _FakeResp(
                422,
                text=f'{{"error":"batch size {n} > maximum allowed batch size {self.max_batch}"}}',
            )
        return _FakeResp(200, payload=[[0.0, 0.1, 0.2, 0.3] for _ in range(n)])


def _proxy(client, max_batch_size: int = 128) -> EmbeddingProxy:
    return EmbeddingProxy(
        [Backend(url="http://fake", klass=BackendClass.GPU)],
        client=client,
        max_batch_size=max_batch_size,
    )


def test_parse_batch_limit():
    assert _parse_batch_limit('{"error":"batch size 40 > maximum allowed batch size 32"}') == 32
    assert _parse_batch_limit("connection refused") is None
    assert _parse_batch_limit("") is None


@pytest.mark.asyncio
async def test_oversized_batch_split_returns_all_in_order():
    client = _FakeClient(max_batch=32)
    proxy = _proxy(client, max_batch_size=128)
    out = await proxy.embed([f"t{i}" for i in range(40)])
    assert len(out) == 40  # nothing dropped
    # the 40 was rejected once, then re-sent as 32 + 8
    assert client.calls[0] == 40
    assert sorted(client.calls[1:]) == [8, 32]


@pytest.mark.asyncio
async def test_422_does_not_trip_circuit_breaker():
    client = _FakeClient(max_batch=32)
    proxy = _proxy(client, max_batch_size=128)
    await proxy.embed([f"t{i}" for i in range(40)])
    backend = proxy._backends[0]
    assert backend.cb_state.name == "CLOSED"
    assert backend.failures == 0


@pytest.mark.asyncio
async def test_learns_cap_and_pre_splits_next_time():
    client = _FakeClient(max_batch=32)
    proxy = _proxy(client, max_batch_size=128)
    await proxy.embed([f"a{i}" for i in range(40)])
    assert proxy._backends[0].max_batch_size == 32  # learned
    client.calls.clear()
    out = await proxy.embed([f"b{i}" for i in range(40)])
    assert len(out) == 40
    # pre-split using the learned cap — no oversized request, no repeat 422
    assert 40 not in client.calls
    assert all(c <= 32 for c in client.calls)
