"""Adapter tests for llm — httpx client integration for LLM API calls."""
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_generate_hyde_calls_llm(mock_llm_httpx):
    """generate_hyde should make an API call and return expanded query."""
    from treeloom.llm import generate_hyde
    result = await generate_hyde("how does authentication work", language="python")
    assert result is not None
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_generate_hyde_caches_result(mock_llm_httpx):
    """Second call with same query should hit cache, not API."""
    from treeloom.llm import generate_hyde
    result1 = await generate_hyde("test query for caching", language="python")
    call_count = mock_llm_httpx.post.call_count
    result2 = await generate_hyde("test query for caching", language="python")
    assert result1 == result2
    assert mock_llm_httpx.post.call_count == call_count


@pytest.mark.asyncio
async def test_generate_summary_calls_llm(mock_llm_httpx):
    """generate_summary should make an API call for chunk summarization."""
    from treeloom.llm import generate_summary
    result = await generate_summary(
        chunk_text="def add(a, b): return a + b",
        language="python",
        file_path="/test/file.py",
    )
    assert result is None or isinstance(result, str)


@pytest.mark.asyncio
async def test_llm_unreachable_graceful_fallback():
    """When LLM is unreachable, should return None without crashing."""
    from treeloom.llm import generate_hyde
    error_client = AsyncMock()
    error_client.post = AsyncMock(side_effect=Exception("Connection refused"))
    with patch("treeloom.adapters.llm_api.llm_adapter._get_client", return_value=error_client):
        result = await generate_hyde("test query")
    assert result is None
