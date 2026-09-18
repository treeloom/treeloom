"""Local sentence-transformers CrossEncoder reranker adapter."""
from unittest.mock import MagicMock

import pytest

from treeloom.adapters.rerank import st_cross_encoder as st


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(st, "_encoder", None)
    yield


def _install_encoder(monkeypatch, scores):
    enc = MagicMock()
    enc.predict.return_value = scores
    monkeypatch.setattr(st, "_get_encoder", lambda: enc)
    return enc


@pytest.mark.asyncio
async def test_empty_texts_no_model_load(monkeypatch):
    enc = _install_encoder(monkeypatch, [])
    assert await st.rerank_texts("q", []) == []
    enc.predict.assert_not_called()


@pytest.mark.asyncio
async def test_scores_pair_with_indices_and_pairs_built(monkeypatch):
    enc = _install_encoder(monkeypatch, [0.2, 4.7, -1.3])
    out = await st.rerank_texts("find auth", ["a", "b", "c"])
    assert out == [(0, 0.2), (1, 4.7), (2, -1.3)]
    # CrossEncoder.predict gets (query, doc) pairs
    pairs = enc.predict.call_args.args[0]
    assert pairs == [("find auth", "a"), ("find auth", "b"), ("find auth", "c")]


@pytest.mark.asyncio
async def test_raw_scores_passed_through(monkeypatch):
    # numpy-like floats must come back as plain floats, unnormalized
    _install_encoder(monkeypatch, [12.5, -8.0])
    out = await st.rerank_texts("q", ["a", "b"])
    assert out == [(0, 12.5), (1, -8.0)]
    assert all(isinstance(s, float) for _, s in out)


def test_device_auto_and_override(monkeypatch):
    monkeypatch.setattr(st, "RERANKER_ST_DEVICE", "cpu")
    assert st._device() == "cpu"
    monkeypatch.setattr(st, "RERANKER_ST_DEVICE", "")
    # auto: returns cuda or cpu depending on torch; just assert it's one of them
    assert st._device() in ("cuda", "cpu")


def test_default_model_is_apache_licensed():
    # the out-of-box default must be the commercially-free small model
    assert st.RERANKER_LOCAL_MODEL == "zeroentropy/zerank-1-small"
