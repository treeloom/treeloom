"""Fallback router — prioritized strategies when retries are exhausted.

When the retry engine gives up, the fallback router provides a soft
landing instead of crashing. Strategies are tried in priority order;
the first one that returns a valid result wins.

Strategies catch their own exceptions — a failed fallback never
propagates to the caller.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from treeloom.domain.audit import FailureMode

# ── strategy type ───────────────────────────────────────────────────

# A fallback strategy: takes the original input and failure context,
# returns a result or None if it couldn't handle it.
FallbackStrategy = Callable[
    [Any, FailureMode, int],
    Optional[Any],
]


# ── the router ─────────────────────────────────────────────────────


class FallbackRouter:
    """Prioritized fallback strategies — first success wins.

    Usage:
        router = FallbackRouter()
        router.register("cached", lambda query, fm, att: cache.get(query))
        router.register("vector_only", lambda query, fm, att: search_no_hyde(query))

        result = router.execute(query, FailureMode.SCHEMA_VIOLATION, 3)
        if result is not None:
            return result
        # All fallbacks exhausted — return error
    """

    def __init__(self):
        self._strategies: list[tuple[str, FallbackStrategy]] = []

    def register(self, name: str, strategy: FallbackStrategy) -> None:
        """Register a fallback strategy. Earlier registrations have higher priority."""
        self._strategies.append((name, strategy))

    def execute(
        self,
        input_data: Any,
        failure_mode: FailureMode,
        attempts: int,
    ) -> tuple[Optional[Any], Optional[str], Optional[str]]:
        """Try each fallback in priority order. Returns (result, strategy_name, error).

        If no strategy returns a result, returns (None, None, error_message).
        """
        last_error: Optional[str] = None

        for name, strategy in self._strategies:
            try:
                result = strategy(input_data, failure_mode, attempts)
                if result is not None:
                    return (result, name, None)
            except Exception as exc:
                # Strategy crashed — log and try next
                last_error = f"{name}: {exc}"
                continue

        # All strategies exhausted
        return (
            None,
            None,
            last_error or "no fallback strategies configured",
        )

    @property
    def strategy_names(self) -> list[str]:
        """Names of registered strategies in priority order."""
        return [name for name, _ in self._strategies]


# ── built-in strategies for Treeloom ───────────────────────────────
# These are registered in the application wiring layer, not here.
# Domain module defines the router; adapters define concrete strategies.


def make_cached_hyde_strategy(cache: dict) -> FallbackStrategy:
    """Strategy: return a cached HyDE expansion vector if available.

    Args:
        cache: Dict mapping query → cached HyDE embedding vector.
    """
    def _cached_hyde(query: str, fm: FailureMode, att: int) -> Optional[Any]:
        return cache.get(query)
    return _cached_hyde


def make_vector_only_strategy(search_fn: Callable) -> FallbackStrategy:
    """Strategy: degrade to vector-only search (no HyDE, no graph).

    Args:
        search_fn: Function that takes (query) and returns search results.
    """
    def _vector_only(query: str, fm: FailureMode, att: int) -> Optional[Any]:
        return search_fn(query)
    return _vector_only
