"""Query-class-dependent payload (sub-change 4).

`_query_class_payload` drops the neighbor/community aux payload for
SYMBOL-BEARING queries (the answer file is already promoted to rank 1, so the
aux is low-value there) while keeping it for SYMBOL-FREE queries (where it earns
its tokens). Off by default; only acts when enabled.
"""
from __future__ import annotations

from treeloom.application.retrieval import _query_class_payload, _query_symbols

_NEIGHBORS = [{"id": "a"}, {"id": "b"}]
_SUMMARIES = {"src#1": "a community summary"}


class TestClassifierSanity:
    def test_symbol_bearing_detected(self):
        assert _query_symbols("how does AuthTokenService refresh tokens")
        assert _query_symbols("what calls set_physics_ticks_per_second")

    def test_symbol_free_has_no_identifiers(self):
        # plain English words (>=5 chars) must NOT count as code identifiers,
        # else every query would be misclassified symbol-bearing.
        assert _query_symbols("how are feature flag evaluation results cached") == []
        assert _query_symbols("where is the retry logic handled") == []


class TestDropsAuxForSymbolBearingWhenEnabled:
    def test_symbol_bearing_enabled_drops_aux(self):
        n, s = _query_class_payload(
            "explain AuthTokenService", list(_NEIGHBORS), dict(_SUMMARIES), True
        )
        assert n == [] and s == {}

    def test_symbol_free_enabled_keeps_aux(self):
        n, s = _query_class_payload(
            "how are results cached", list(_NEIGHBORS), dict(_SUMMARIES), True
        )
        assert n == _NEIGHBORS and s == _SUMMARIES


class TestDisabledIsNoOp:
    def test_symbol_bearing_disabled_keeps_aux(self):
        n, s = _query_class_payload(
            "explain AuthTokenService", list(_NEIGHBORS), dict(_SUMMARIES), False
        )
        assert n == _NEIGHBORS and s == _SUMMARIES

    def test_symbol_free_disabled_keeps_aux(self):
        n, s = _query_class_payload(
            "how are results cached", list(_NEIGHBORS), dict(_SUMMARIES), False
        )
        assert n == _NEIGHBORS and s == _SUMMARIES
