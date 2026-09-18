"""Backward-compatible re-export of llm module."""
from treeloom.adapters.llm_api.llm_adapter import (
    _strip_thinking,
    _get_client,
    _get_semaphore,
    _hyde_cache_get,
    _hyde_cache_put,
    _chat,
    chunk_cache_key,
    cache_get_many,
    cache_put,
    generate_hyde,
    generate_summary,
    summarize_with_cache,
)  # noqa: F401
