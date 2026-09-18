"""Domain tests for embedder — routing through the cost-aware TEI proxy.

`treeloom.embedder.embed` delegates to the module-level EmbeddingProxy;
`set_proxy()` is the documented test seam (no HTTP).
"""
import pytest

from treeloom.adapters.tei import embedding_proxy


class _StubProxy:
    """Duck-typed proxy capturing calls and returning fixed-dim embeddings."""

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
def stub_proxy():
    old = embedding_proxy._proxy
    stub = _StubProxy()
    embedding_proxy.set_proxy(stub)
    yield stub
    embedding_proxy._proxy = old


@pytest.mark.asyncio
async def test_embed_single_text(stub_proxy):
    from treeloom.embedder import embed
    result = await embed(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1536
    assert stub_proxy.calls == [["hello world"]]


@pytest.mark.asyncio
async def test_embed_batch_of_32(stub_proxy):
    from treeloom.embedder import embed
    texts = [f"text {i}" for i in range(32)]
    result = await embed(texts)
    assert len(result) == 32


@pytest.mark.asyncio
async def test_embed_empty_list(stub_proxy):
    from treeloom.embedder import embed
    result = await embed([])
    assert result == []
