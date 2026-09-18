"""Adapter tests for retriever — Milvus-dependent operations."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_init_collection_idempotent():
    """init_collection consults has_collection and skips create when present.

    Patches the MilvusClient class through vector_store's own module
    reference — init_collection invokes operations as unbound
    `MilvusClient.<method>` via `_execute_with_reconnect`.
    """
    from treeloom.adapters.milvus import vector_store as vs

    adapter = vs.MilvusAdapter(host="localhost", port="19530")
    adapter._client = MagicMock()
    with patch.object(vs.MilvusClient, "has_collection", MagicMock(return_value=True)) as has_c, \
         patch.object(vs.MilvusClient, "create_collection", MagicMock()) as create_c:
        await adapter.init_collection()
    has_c.assert_called_once()
    create_c.assert_not_called()
