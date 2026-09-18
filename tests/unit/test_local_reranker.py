"""Local CPU cross-encoder reranker — contract, lazy load, score semantics."""
from unittest.mock import MagicMock

import pytest

from treeloom.adapters.rerank import local_cross_encoder as lc


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(lc, "_encoder", None)
    yield


def _install_encoder(monkeypatch, scores: list[float]) -> MagicMock:
    enc = MagicMock()
    enc.rerank.return_value = iter(scores)
    monkeypatch.setattr(lc, "_get_encoder", lambda: enc)
    return enc


@pytest.mark.asyncio
async def test_empty_texts_no_model_load(monkeypatch):
    enc = _install_encoder(monkeypatch, [])
    assert await lc.rerank_texts("q", []) == []
    enc.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_scores_pair_with_input_indexes(monkeypatch):
    enc = _install_encoder(monkeypatch, [0.2, 4.7, -1.3])
    out = await lc.rerank_texts("q", ["a", "b", "c"])
    assert out == [(0, 0.2), (1, 4.7), (2, -1.3)]
    enc.rerank.assert_called_once_with(
        "q", ["a", "b", "c"], batch_size=lc.RERANK_LOCAL_BATCH_SIZE
    )


@pytest.mark.asyncio
async def test_raw_logits_passed_through_unmodified(monkeypatch):
    """Raw (unbounded, possibly negative) logits must not be normalized here —
    cross-batch comparability and the RERANK_NORMALIZE fusion depend on it."""
    _install_encoder(monkeypatch, [-8.25, 11.5])
    out = await lc.rerank_texts("q", ["a", "b"])
    assert out == [(0, -8.25), (1, 11.5)]


@pytest.mark.asyncio
async def test_downstream_ordering_invariant_under_affine_transform(monkeypatch):
    """graph_rescore min-max normalizes within the candidate set, so an
    affine change of score scale (a different cross-encoder's logit range)
    must produce the identical final ordering."""
    from treeloom.application import retrieval

    def hits(scores):
        return [
            {"entity": {"file_path": f"/f{i}.py"}, "relevance_score": s}
            for i, s in enumerate(scores)
        ]

    base = [2.0, -1.0, 0.5]
    affine = [s * 7.3 + 42.0 for s in base]
    out_a = await retrieval.graph_rescore(hits(base), [], "q", {}, {})
    out_b = await retrieval.graph_rescore(hits(affine), [], "q", {}, {})
    order_a = [r["entity"]["file_path"] for r in out_a]
    order_b = [r["entity"]["file_path"] for r in out_b]
    assert order_a == order_b == ["/f0.py", "/f2.py", "/f1.py"]
    # normalized rel signals are identical (to float precision), not just
    # same-ordered
    assert [r["graph_signals"]["rel_norm"] for r in out_a] == pytest.approx(
        [r["graph_signals"]["rel_norm"] for r in out_b]
    )


def test_default_model_is_supported_by_fastembed():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    assert lc.RERANKER_LOCAL_MODEL in supported
    # the documented quality option (GPU-parity on featbit-clean-100)
    assert "jinaai/jina-reranker-v2-base-multilingual" in supported
