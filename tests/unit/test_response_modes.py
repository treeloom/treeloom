"""Unit tests for the opt-in search response modes.

Covers the pure `_apply_response_mode` body-shaping transform:
- full  -> chunks returned unchanged (byte-identical to earlier).
- facet -> snippet body dropped, header/citation/score/metadata kept.
- summary_tail -> top-2 snippets kept, tail replaced by cached summary,
  falling back to the full snippet when no summary is cached (NEVER empty).

These exercise the seam directly with plain dicts + a mocked summary lookup,
so they need no Milvus/Neo4j/LLM services.
"""
from treeloom.application.retrieval import (
    RESPONSE_MODES,
    SUMMARY_TAIL_HEAD,
    _apply_response_mode,
)
from treeloom.llm import chunk_cache_key


def _chunk(i, snippet="body", header="", citation="", score=0.9):
    d = {
        "file_path": f"src/f{i}.py",
        "start_line": i * 10,
        "end_line": i * 10 + 5,
        "language": "python",
        "score": score,
        "snippet": snippet,
    }
    if header:
        d["header"] = header
    if citation:
        d["citation"] = citation
    return d


# ── full mode ────────────────────────────────────────────────────────────

def test_full_returns_input_unchanged_identity():
    chunks = [_chunk(0), _chunk(1)]
    out = _apply_response_mode(chunks, ["a", "b"], "full")
    assert out is chunks  # full mode is a true no-op
    assert out == [_chunk(0), _chunk(1)]


# ── facet mode ─────────────────────────────────────────────────────────────

def test_facet_drops_snippet_keeps_metadata_header_citation_score():
    chunks = [
        _chunk(0, snippet="def foo(): ...", header="# in Class A · defines foo",
               citation="repo@abc123:src/f0.py:0-5", score=0.81),
    ]
    out = _apply_response_mode(chunks, ["def foo(): ..."], "facet")
    c = out[0]
    assert "snippet" not in c
    assert c["header"] == "# in Class A · defines foo"
    assert c["citation"] == "repo@abc123:src/f0.py:0-5"
    assert c["score"] == 0.81
    assert c["file_path"] == "src/f0.py"
    assert c["start_line"] == 0 and c["end_line"] == 5
    assert c["language"] == "python"


def test_facet_does_not_mutate_input():
    chunks = [_chunk(0, snippet="x")]
    original = dict(chunks[0])
    _apply_response_mode(chunks, ["x"], "facet")
    assert chunks[0] == original  # transform copies, never mutates in place


# ── summary_tail mode ───────────────────────────────────────────────────────

def test_summary_tail_keeps_top_two_snippets():
    chunks = [_chunk(i, snippet=f"full body {i}") for i in range(4)]
    raws = [f"raw{i}" for i in range(4)]
    summaries = {
        chunk_cache_key("raw2"): "summary of 2",
        chunk_cache_key("raw3"): "summary of 3",
    }
    out = _apply_response_mode(chunks, raws, "summary_tail", summaries)
    # ranks 0,1 (the SUMMARY_TAIL_HEAD head) keep full snippets
    assert SUMMARY_TAIL_HEAD == 2
    assert out[0]["snippet"] == "full body 0"
    assert out[1]["snippet"] == "full body 1"
    # ranks 2,3 swapped to the cached summary
    assert out[2]["snippet"] == "summary of 2"
    assert out[3]["snippet"] == "summary of 3"


def test_summary_tail_falls_back_to_snippet_when_summary_absent():
    chunks = [_chunk(i, snippet=f"full body {i}") for i in range(3)]
    raws = ["raw0", "raw1", "raw2"]
    # No summary cached for raw2 -> must fall back to its full snippet.
    out = _apply_response_mode(chunks, raws, "summary_tail", {})
    assert out[2]["snippet"] == "full body 2"
    # NEVER empty
    assert out[2]["snippet"]


def test_summary_tail_unsummarizable_merged_hit_falls_back():
    # A merged adjacent run has raw_text None -> un-summarizable -> snippet kept.
    chunks = [_chunk(i, snippet=f"full body {i}") for i in range(3)]
    raws = ["raw0", "raw1", None]
    summaries = {chunk_cache_key("raw2"): "should not be used"}
    out = _apply_response_mode(chunks, raws, "summary_tail", summaries)
    assert out[2]["snippet"] == "full body 2"


def test_summary_tail_never_emits_empty_body():
    chunks = [_chunk(i, snippet=f"body{i}") for i in range(5)]
    raws = [f"raw{i}" for i in range(5)]
    # Only one tail summary present; the rest must fall back, never empty.
    summaries = {chunk_cache_key("raw3"): "S3"}
    out = _apply_response_mode(chunks, raws, "summary_tail", summaries)
    for c in out:
        assert c["snippet"], f"empty body emitted: {c}"
    assert out[3]["snippet"] == "S3"
    assert out[2]["snippet"] == "body2"
    assert out[4]["snippet"] == "body4"


def test_summary_tail_drops_internal_header_field():
    # header is a full-mode internal field surfaced only by facet; summary_tail
    # must not leak it onto the wire.
    chunks = [_chunk(i, snippet=f"b{i}", header="# internal") for i in range(3)]
    out = _apply_response_mode(chunks, ["r0", "r1", "r2"], "summary_tail", {})
    for c in out:
        assert "header" not in c


# ── validation ──────────────────────────────────────────────────────────────

def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        _apply_response_mode([_chunk(0)], ["x"], "bogus")


def test_response_modes_constant():
    assert RESPONSE_MODES == ("full", "facet", "summary_tail")
