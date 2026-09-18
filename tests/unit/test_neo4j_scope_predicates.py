"""Regression test for the search-500 bug.

`_scope_predicates` is a pure synchronous helper. It was once decorated with the
async `@_retry_on_disconnect` wrapper, which made it return a coroutine and 500'd
every /search and /graph-explore on a freshly-started indexer
("cannot unpack non-iterable coroutine object"). Guard that it stays sync and
returns a (clauses, params) tuple — catchable without a live Neo4j.
"""
import inspect

import pytest


def _load():
    try:
        from treeloom.adapters.neo4j.graph_store import _scope_predicates
    except Exception as exc:  # require_env(NEO4J_*) at import in some envs
        pytest.skip(f"neo4j graph_store not importable here: {exc}")
    return _scope_predicates


def test_scope_predicates_is_not_a_coroutine():
    fn = _load()
    assert not inspect.iscoroutinefunction(fn)


def test_scope_predicates_returns_clauses_and_params():
    fn = _load()
    clauses, params = fn("e", "src-1", ["src-2"])
    assert isinstance(clauses, list)
    assert isinstance(params, dict)
    assert params.get("scope_source_id") == "src-1"
    assert params.get("scope_exclude") == ["src-2"]
