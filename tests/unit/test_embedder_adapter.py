"""Adapter tests for embedder — OpenRouter HTTP client integration."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx


@pytest.mark.asyncio
async def test_embed_calls_openrouter_with_correct_payload():
    """embed() should POST to OpenRouter /embeddings with correct body."""
    from treeloom.adapters.openrouter.embedding_adapter import embed
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "data": [{"embedding": [0.1] * 1536}]
    })
    mock_client.post = AsyncMock(return_value=mock_resp)
    with patch("treeloom.adapters.openrouter.embedding_adapter._get_client", return_value=mock_client):
        await embed(["sample text"])
    mock_client.post.assert_called()


@pytest.mark.asyncio
async def test_embed_handles_openrouter_error():
    """embed() should handle HTTP errors gracefully."""
    from treeloom.adapters.openrouter.embedding_adapter import embed
    error_client = AsyncMock(spec=httpx.AsyncClient)
    error_resp = MagicMock()
    error_resp.status_code = 503
    error_resp.text = "Service Unavailable"
    error_resp.raise_for_status = MagicMock(side_effect=Exception("503"))
    error_client.post = AsyncMock(return_value=error_resp)
    with patch("treeloom.adapters.openrouter.embedding_adapter._get_client", return_value=error_client):
        with pytest.raises(Exception):
            await embed(["test"])
