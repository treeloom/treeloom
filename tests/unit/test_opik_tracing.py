"""Hermetic tests for the opt-in Opik tracking adapter.

No `opik` install or server needed — covers the pure result-shaping helpers and
the load-bearing guarantee that when tracking is OFF the module is a true no-op
that never imports `opik` (default benchmark runs must be byte-identical).
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def ot(monkeypatch):
    # Ensure a clean, disabled state regardless of ambient env, and a fresh
    # module so the cached enabled() flag is recomputed.
    for var in ("OPIK_TRACK", "OPIK_URL", "OPIK_URL_OVERRIDE"):
        monkeypatch.delenv(var, raising=False)
    sys.modules.pop("treeloom.adapters.benchmark.opik_tracing", None)
    return importlib.import_module("treeloom.adapters.benchmark.opik_tracing")


def test_disabled_without_env(ot):
    assert ot.enabled() is False


def test_log_run_noop_and_no_opik_import_when_disabled(ot):
    sys.modules.pop("opik", None)
    # Must not raise and must not import opik.
    ot.log_run(run_name="r", repo_name="repo", queries=[], gold={}, rows=[],
               arms=["treeloom"], config={})
    assert "opik" not in sys.modules


def test_enabled_when_opik_track_set(ot, monkeypatch):
    monkeypatch.setenv("OPIK_TRACK", "1")
    sys.modules.pop("treeloom.adapters.benchmark.opik_tracing", None)
    mod = importlib.import_module("treeloom.adapters.benchmark.opik_tracing")
    # enabled() also requires the opik SDK to import; gate only on that when present.
    try:
        import opik  # noqa: F401
        assert mod.enabled() is True
    except Exception:
        assert mod.enabled() is False  # requested but SDK absent -> disabled


def test_usage_passes_through_cached_and_omits_zero(ot):
    arm = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
           "cached_input_tokens": 300, "cache_write_tokens": 0, "reasoning_tokens": 0}
    u = ot._usage(arm)
    assert u["prompt_tokens"] == 1200 and u["total_tokens"] == 1280
    assert u["cached_input_tokens"] == 300        # non-zero passes through
    assert "reasoning_tokens" not in u            # zero is omitted
    assert "cache_write_tokens" not in u


def test_scores_from_judge_and_retrieval(ot):
    arm = {"recall@1": 1.0, "recall@5": 0.8, "mrr": 0.75,
           "judge": {"correctness": 4.3, "completeness": 4.0}}
    scores = {s["name"]: s["value"] for s in ot._scores(arm)}
    assert scores == {"correctness": 4.3, "completeness": 4.0,
                      "recall@1": 1.0, "recall@5": 0.8, "mrr": 0.75}


def test_scores_skip_missing(ot):
    # No judge, partial retrieval — only present values are emitted.
    scores = {s["name"]: s["value"] for s in ot._scores({"recall@1": 0.0})}
    assert scores == {"recall@1": 0.0}


def test_tags_include_arm_repo_and_search_url(ot):
    tags = ot._tags("treeloom", "featbit", "run1",
                    {"search_url": "http://localhost:8002"})
    assert "arm:treeloom" in tags
    assert "repo:featbit" in tags
    assert "run:run1" in tags
    # search_url is the comparison axis that catches a wrong-target run.
    assert "search_url:http://localhost:8002" in tags
