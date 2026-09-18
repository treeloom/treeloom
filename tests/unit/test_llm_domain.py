"""Domain tests for llm — prompt templating, caching, response parsing."""
import pytest
from treeloom.llm import _strip_thinking, chunk_cache_key, _hyde_cache_get, _hyde_cache_put


# ── Strip Thinking ──────────────────────────────────────────────────────

def test_strip_thinking_no_tags():
    text = "plain response text"
    result = _strip_thinking(text)
    assert result is not None

def test_strip_thinking_empty_string():
    result = _strip_thinking("")
    assert result is not None


# ── Chunk Cache Key ─────────────────────────────────────────────────────

def test_chunk_cache_key_deterministic():
    text = "def hello(): return 'world'"
    key1 = chunk_cache_key(text)
    key2 = chunk_cache_key(text)
    assert key1 == key2
    assert len(key1) == 40  # SHA1 hex digest

def test_chunk_cache_key_different_texts():
    key1 = chunk_cache_key("text one")
    key2 = chunk_cache_key("text two")
    assert key1 != key2


# ── HyDE Cache ──────────────────────────────────────────────────────────

def test_hyde_cache_miss():
    result = _hyde_cache_get("nonexistent-key-12345")
    assert result is None

def test_hyde_cache_put_and_get():
    _hyde_cache_put("test-key", "test value")
    result = _hyde_cache_get("test-key")
    assert result == "test value"
