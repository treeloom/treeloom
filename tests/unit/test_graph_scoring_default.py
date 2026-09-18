"""graph RESCORING on by default; payload-without-rescoring opt-in.

These are Detroit-style tests against the live retrieval orchestration
(`treeloom.application.retrieval.graph_search`) with its store/community
collaborators stubbed. They pin the invariants:

1. The `USE_GRAPH_SCORING` env fallback defaults to ON — kept it on
   because it lifts rank-1 under the production reranker (guava symbol-free
   recall@1 0.16→0.49); payload-only is a Cohere-class opt-in.
2. With graph scoring OFF (the opt-in payload-only mode) `graph_search` still
   returns the neighbor + community-summary PAYLOAD, and does NOT call
   `graph_rescore`.
3. With graph scoring ON `graph_rescore` runs (payload still present too).
4. `use_graph_scoring=None` falls back to the env default (ON) → rescore runs.
"""
import ast
import asyncio
import inspect
import os

from treeloom.application import retrieval


def test_env_default_graph_scoring_on():
    """The module's USE_GRAPH_SCORING env fallback defaults to ON.

    Asserted against the *source default* (the literal in the
    `os.environ.get("USE_GRAPH_SCORING", "<default>")` call) so a developer's
    local .env setting USE_GRAPH_SCORING=0 doesn't mask a code regression.
    """
    src = inspect.getsource(retrieval)
    tree = ast.parse(src)
    default = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "USE_GRAPH_SCORING"):
            default = node.args[1].value
            break
    assert default == "1", f"expected env default '1' (on), got {default!r}"
    # And the runtime value honors an explicitly-cleared env.
    if "USE_GRAPH_SCORING" not in os.environ:
        assert retrieval.USE_GRAPH_SCORING is True


class _FakeStore:
    """Minimal vector-store stub returning two milvus-shaped hits."""

    def search(self, *a, **k):
        return [
            {"id": "1", "distance": 0.9,
             "entity": {"chunk_text": "def foo(): pass", "file_path": "a.py",
                        "start_line": 1, "end_line": 1, "language": "python",
                        "source_id": "s1"}},
            {"id": "2", "distance": 0.8,
             "entity": {"chunk_text": "def bar(): pass", "file_path": "b.py",
                        "start_line": 1, "end_line": 1, "language": "python",
                        "source_id": "s1"}},
        ]

    def hybrid_search(self, *a, **k):
        return self.search()


def _patch_common(monkeypatch, rescore_calls):
    """Stub every collaborator graph_search touches except the rescore branch."""
    monkeypatch.setattr(retrieval, "_store", lambda: _FakeStore())
    monkeypatch.setattr(retrieval, "SYMBOL_PROMOTE", False)
    monkeypatch.setattr(retrieval, "ADAPTIVE_TOPK", False)

    async def _rerank(query, results, top_k=5, batch_size=None):
        for r in results:
            r["relevance_score"] = float(r["distance"])
        return results[:top_k]

    monkeypatch.setattr(retrieval, "rerank", _rerank)

    # Entities for both files: gives the response a graph identity and lets the
    # neighbor batch find matched ids.
    async def _entities(paths, **k):
        return {
            "a.py": [{"id": "e_a", "name": "foo", "type": "Function",
                      "file_path": "a.py", "community_id": 7}],
            "b.py": [{"id": "e_b", "name": "bar", "type": "Function",
                      "file_path": "b.py", "community_id": 7}],
        }

    monkeypatch.setattr(retrieval.graph_store, "get_entities_by_files_batch", _entities)

    async def _neighbors(ids, **k):
        # e_a's neighbor is an internal entity the agent can open.
        return {"e_a": [{"id": "nb1", "type": "Function", "name": "helper",
                         "file_path": "c.py", "start_line": 5, "end_line": 9,
                         "language": "python"}]}

    monkeypatch.setattr(retrieval.graph_store, "get_neighbors_batch", _neighbors)

    async def _community_ids(ids):
        return {7}

    monkeypatch.setattr(retrieval.graph_store, "get_community_ids", _community_ids)

    async def _summaries(cids):
        return {"7": "community seven summary"}

    monkeypatch.setattr(retrieval.community, "get_community_summaries", _summaries)

    async def _embs():
        return {7: [0.1, 0.2]}

    monkeypatch.setattr(retrieval, "_get_community_embeddings", _embs)

    async def _rescore(reranked, *a, **k):
        rescore_calls.append(True)
        for r in reranked:
            r["final_score"] = 1.0
        return reranked

    monkeypatch.setattr(retrieval, "graph_rescore", _rescore)


def test_graph_scoring_off_keeps_payload_skips_rescore(monkeypatch):
    rescore_calls: list = []
    _patch_common(monkeypatch, rescore_calls)

    result = asyncio.run(retrieval.graph_search(
        query_embedding=[0.1, 0.2], query_text="natural language query",
        top_k=5, use_graph_scoring=False, use_hybrid=False, use_hyde=False,
    ))

    # PAYLOAD is retained even with rescoring off.
    assert result["chunks"], "chunks should be present"
    assert result["neighbors"], "neighbor payload must survive rescore-off"
    assert result["community_summaries"] == {"7": "community seven summary"}
    # Rescore branch was NOT taken.
    assert rescore_calls == []


def test_graph_scoring_on_runs_rescore(monkeypatch):
    rescore_calls: list = []
    _patch_common(monkeypatch, rescore_calls)

    result = asyncio.run(retrieval.graph_search(
        query_embedding=[0.1, 0.2], query_text="natural language query",
        top_k=5, use_graph_scoring=True, use_hybrid=False, use_hyde=False,
    ))

    assert rescore_calls == [True], "rescore must run when explicitly enabled"
    # Payload still present in the rescore-on path too.
    assert result["neighbors"]
    assert result["community_summaries"]


def test_graph_scoring_default_none_uses_env_on(monkeypatch):
    """Passing use_graph_scoring=None falls back to USE_GRAPH_SCORING (now ON)."""
    rescore_calls: list = []
    _patch_common(monkeypatch, rescore_calls)
    monkeypatch.setattr(retrieval, "USE_GRAPH_SCORING", True)

    asyncio.run(retrieval.graph_search(
        query_embedding=[0.1, 0.2], query_text="q", top_k=5,
        use_graph_scoring=None, use_hybrid=False, use_hyde=False,
    ))
    assert rescore_calls == [True]
