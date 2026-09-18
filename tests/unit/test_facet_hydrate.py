"""Unit tests for facet hit_id + the hydrate_chunks companion.

No Milvus/Neo4j/LLM services: the hit_id helpers are pure, the facet emission is
the pure `_apply_response_mode` transform, and hydrate_chunks is exercised with a
fake store module monkeypatched in.
"""
import asyncio
import types

import pytest

from treeloom.domain.shared import make_hit_id, parse_hit_id
from treeloom.application import retrieval
from treeloom.application.retrieval import _apply_response_mode, hydrate_chunks


# ── hit_id helpers ─────────────────────────────────────────────────────────

def test_make_parse_roundtrip():
    assert make_hit_id("src/a/b.py", 10, 42) == "src/a/b.py:10-42"
    assert parse_hit_id("src/a/b.py:10-42") == ("src/a/b.py", 10, 42)


def test_parse_tolerates_colon_in_path():
    hid = make_hit_id("C:/x/y.cs", 1, 5)
    assert parse_hit_id(hid) == ("C:/x/y.cs", 1, 5)


@pytest.mark.parametrize("bad", ["garbage", "", "a:b", "a:1_2", None])
def test_parse_bad_returns_none(bad):
    assert parse_hit_id(bad) is None


# ── facet emits hit_id; full / summary_tail do not ─────────────────────────

def _chunk(i, snippet="body"):
    return {
        "file_path": f"src/f{i}.py", "start_line": i * 10,
        "end_line": i * 10 + 5, "language": "python", "score": 0.9,
        "snippet": snippet, "header": f"class C{i}",
    }


def test_facet_sets_hit_id_and_drops_body():
    out = _apply_response_mode([_chunk(1)], ["raw"], "facet")
    assert "snippet" not in out[0]
    assert out[0]["hit_id"] == make_hit_id("src/f1.py", 10, 15)


def test_full_and_summary_tail_have_no_hit_id():
    full = _apply_response_mode([_chunk(1)], ["raw"], "full")
    assert "hit_id" not in full[0]
    tail = _apply_response_mode([_chunk(0), _chunk(1), _chunk(2)],
                                ["a", "b", "c"], "summary_tail")
    assert all("hit_id" not in c for c in tail)


# ── hydrate_chunks orchestration ───────────────────────────────────────────

def _fake_store(bodies, *, has_method=True):
    mod = types.SimpleNamespace()
    if has_method:
        async def get_chunk_bodies(locations, source_id=None):
            mod.seen = (locations, source_id)
            return bodies
        mod.get_chunk_bodies = get_chunk_bodies
    return mod


def test_hydrate_parses_ids_shapes_bodies_and_strips_blanks(monkeypatch):
    bodies = [{
        "file_path": "src/f1.py", "start_line": 10, "end_line": 15,
        "language": "python", "chunk_text": "def f():\n\n    return 1\n   ",
        "source_id": "s1",
    }]
    store = _fake_store(bodies)
    monkeypatch.setattr(retrieval, "_store", lambda: store)

    out = asyncio.run(hydrate_chunks(
        ["src/f1.py:10-15", "garbage"], source_id="s1"))

    # bad id skipped -> only the one valid location forwarded
    assert store.seen == ([("src/f1.py", 10, 15)], "s1")
    assert len(out) == 1
    assert out[0]["file_path"] == "src/f1.py"
    assert out[0]["hit_id"] == "src/f1.py:10-15"
    # blank line dropped + trailing whitespace stripped
    assert out[0]["snippet"] == "def f():\n    return 1"


def test_hydrate_empty_when_no_valid_ids(monkeypatch):
    monkeypatch.setattr(retrieval, "_store", lambda: _fake_store([]))
    assert asyncio.run(hydrate_chunks(["garbage", ""])) == []


def test_hydrate_empty_when_backend_lacks_support(monkeypatch):
    monkeypatch.setattr(retrieval, "_store",
                        lambda: _fake_store([], has_method=False))
    assert asyncio.run(hydrate_chunks(["src/f1.py:1-5"])) == []
