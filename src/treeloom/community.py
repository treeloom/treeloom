"""Backward-compatible re-export of community module."""
from treeloom.adapters.graph.community_adapter import (
    _build_nx_graph,
    build_all_sources,
    detect_communities,
    get_all_community_summaries,
    get_community_summaries,
    load_community_embeddings,
    run_post_index_signals,
    summarize_community,
)  # noqa: F401
