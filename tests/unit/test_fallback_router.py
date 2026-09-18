"""Tests for FallbackRouter domain model."""

import pytest

from treeloom.domain.audit import FailureMode
from treeloom.domain.fallback_router import (
    FallbackRouter,
    make_cached_hyde_strategy,
    make_vector_only_strategy,
)


class TestFallbackRouter:
    """Prioritized fallback strategies."""

    def test_first_strategy_wins(self):
        router = FallbackRouter()
        router.register("s1", lambda q, fm, att: "result_from_s1")
        router.register("s2", lambda q, fm, att: "result_from_s2")

        result, name, error = router.execute(
            "query", FailureMode.SCHEMA_VIOLATION, 3
        )
        assert result == "result_from_s1"
        assert name == "s1"
        assert error is None

    def test_falls_through_to_second(self):
        router = FallbackRouter()
        router.register("s1", lambda q, fm, att: None)  # skips
        router.register("s2", lambda q, fm, att: "result_from_s2")

        result, name, error = router.execute(
            "query", FailureMode.TIMEOUT, 4
        )
        assert result == "result_from_s2"
        assert name == "s2"

    def test_all_exhausted_returns_none(self):
        router = FallbackRouter()
        router.register("s1", lambda q, fm, att: None)
        router.register("s2", lambda q, fm, att: None)

        result, name, error = router.execute(
            "query", FailureMode.LLM_ERROR, 3
        )
        assert result is None
        assert name is None
        assert error is not None

    def test_no_strategies_returns_error(self):
        router = FallbackRouter()
        result, name, error = router.execute(
            "query", FailureMode.CIRCUIT_OPEN, 1
        )
        assert result is None
        assert error is not None
        assert "no fallback" in error.lower()

    def test_crashed_strategy_is_skipped(self):
        router = FallbackRouter()
        router.register("crashy", lambda q, fm, att: 1 / 0)  # ZeroDivisionError
        router.register("safe", lambda q, fm, att: "recovered")

        result, name, error = router.execute(
            "query", FailureMode.CIRCUIT_OPEN, 1
        )
        assert result == "recovered"
        assert name == "safe"
        assert error is None

    def test_crash_error_is_recorded_when_all_fail(self):
        router = FallbackRouter()
        router.register("crashy1", lambda q, fm, att: 1 / 0)
        router.register("crashy2", lambda q, fm, att: {}["missing"])

        result, name, error = router.execute(
            "query", FailureMode.CIRCUIT_OPEN, 1
        )
        assert result is None
        assert error is not None
        assert "crashy2" in error  # Last error preserved

    def test_strategy_names_in_order(self):
        router = FallbackRouter()
        router.register("primary", lambda q, fm, att: None)
        router.register("secondary", lambda q, fm, att: None)
        assert router.strategy_names == ["primary", "secondary"]

    def test_cached_hyde_strategy_returns_cached_value(self):
        cache = {"hello": "hyde_vector_123"}
        strategy = make_cached_hyde_strategy(cache)
        result = strategy("hello", FailureMode.TIMEOUT, 2)
        assert result == "hyde_vector_123"

    def test_cached_hyde_strategy_returns_none_for_miss(self):
        cache = {"hello": "hyde_vector_123"}
        strategy = make_cached_hyde_strategy(cache)
        result = strategy("world", FailureMode.TIMEOUT, 2)
        assert result is None

    def test_vector_only_strategy_calls_search_fn(self):
        called_with = []

        def fake_search(query: str):
            called_with.append(query)
            return {"results": [query * 2]}

        strategy = make_vector_only_strategy(fake_search)
        result = strategy("test", FailureMode.CIRCUIT_OPEN, 1)

        assert called_with == ["test"]
        assert result == {"results": ["testtest"]}
