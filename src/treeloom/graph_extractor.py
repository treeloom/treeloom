"""Backward-compatible re-export of graph extractor module."""
from treeloom.adapters.tree_sitter.graph_extractor import (
    _detect_language,
    _fallback_module_entity,
    _find_enclosing_func,
    _resolve_import_path,
    _run_query,
    Entity,
    Relationship,
    extract_graph,
    LANGUAGE_QUERIES,
)  # noqa: F401
