"""Dependency injection wiring: create adapter instances and inject into domain services.

This module creates all adapter instances and injects them into domain services.
Used by the application layer to bootstrap the system.
"""
from treeloom.domain.indexing import EmbeddingPort, IndexingService
from treeloom.domain.search import SearchService, CommunityPort, GraphSearchPort
from treeloom.adapters.milvus.vector_store import MilvusVectorStoreAdapter
from treeloom.adapters.tei.embedding_proxy import TEIEmbeddingAdapter
from treeloom.adapters.tei.reranker import TEIRerankerAdapter
from treeloom.adapters.neo4j.graph_store import Neo4jGraphStoreAdapter
from treeloom.adapters.llm_api.llm_adapter import LLMAdapter
from treeloom.adapters.graph.community_adapter import CommunityAdapter
from treeloom.application.search import SearchService as SearchServiceImpl
from treeloom.application.indexer_service import IndexerService as IndexerServiceImpl


def create_dependencies() -> dict:
    """Create all adapter instances and return as DI container."""
    return {
        "embedding_port": TEIEmbeddingAdapter(),
        "rerank_port": TEIRerankerAdapter(),
        "vector_store": MilvusVectorStoreAdapter(),
        "graph_store": Neo4jGraphStoreAdapter(),
        "llm": LLMAdapter(),
        "community": CommunityAdapter(),
    }


def create_search_service(deps: dict | None = None) -> SearchService:
    """Create SearchService with dependency injection."""
    if deps is None:
        deps = create_dependencies()
    return SearchServiceImpl(
        vector_store=deps["vector_store"],
        rerank_port=deps["rerank_port"],
        graph_search_port=deps["graph_store"],
        community_port=deps["community"],
        llm_port=deps["llm"],
        embedding_port=deps["embedding_port"],
    )


def create_indexer_service(deps: dict | None = None) -> IndexingService:
    """Create IndexerService with dependency injection."""
    if deps is None:
        deps = create_dependencies()
    return IndexerServiceImpl(
        embedding_port=deps["embedding_port"],
        graph_store=deps["graph_store"],
    )
