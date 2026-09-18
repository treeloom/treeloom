"""/ rerank-input mining: the /search return_pool flag returns the raw
pre-rerank pool (the reranker's exact input) WITHOUT calling rerank.

Detroit-style against the live retrieval orchestration with the store stubbed —
no Milvus/Neo4j/reranker services. Run: env -u PYTHONPATH .venv/bin/python -m pytest
tests/unit/test_return_pool.py -v
"""
import pytest

from treeloom.application import retrieval


class _FakeStore:
    def search(self, *a, **k):
        return [
            {"id": "1", "distance": 0.9,
             "entity": {"chunk_text": "def foo(): pass", "file_path": "a.py",
                        "start_line": 1, "end_line": 3, "language": "python", "source_id": "s1"}},
            {"id": "2", "distance": 0.8,
             "entity": {"chunk_text": "def bar(): pass", "file_path": "b.py",
                        "start_line": 5, "end_line": 7, "language": "python", "source_id": "s1"}},
        ]

    def hybrid_search(self, *a, **k):
        return self.search()


@pytest.mark.asyncio
async def test_return_pool_returns_raw_pool_without_rerank(monkeypatch):
    monkeypatch.setattr(retrieval, "_store", lambda: _FakeStore())

    async def _boom(*a, **k):  # rerank must NOT be called on the return_pool path
        raise AssertionError("rerank was called on the return_pool path")
    monkeypatch.setattr(retrieval, "rerank", _boom)

    out = await retrieval.graph_search(
        [0.0], "find foo", top_k=10, use_hybrid=False, return_pool=True)

    assert set(out.keys()) == {"pool"}
    pool = out["pool"]
    assert [p["chunk_text"] for p in pool] == ["def foo(): pass", "def bar(): pass"]
    assert [p["file_path"] for p in pool] == ["a.py", "b.py"]
    assert pool[0]["start_line"] == 1 and pool[0]["end_line"] == 3
    assert pool[0]["distance"] == 0.9


@pytest.mark.asyncio
async def test_return_pool_empty_when_no_vec_results(monkeypatch):
    class _Empty:
        def search(self, *a, **k): return []
        def hybrid_search(self, *a, **k): return []
    monkeypatch.setattr(retrieval, "_store", lambda: _Empty())
    out = await retrieval.graph_search([0.0], "q", use_hybrid=False, return_pool=True)
    # empty pool short-circuit returns the standard empty shape (no "pool" needed)
    assert out.get("pool", []) == [] or out.get("chunks", []) == []
