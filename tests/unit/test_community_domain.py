"""Domain tests for treeloom.community module.

Tests pure domain logic: _build_nx_graph, summarize_community.
Follows Principle VIII — Detroit-style, mock only external I/O (graph_store).
"""

from __future__ import annotations

import networkx as nx

from treeloom.community import _build_nx_graph, summarize_community


# ---------------------------------------------------------------------------
# _build_nx_graph tests
# ---------------------------------------------------------------------------

def test_build_nx_graph_empty_nodes():
    """_build_nx_graph with an empty list returns a graph with no nodes or edges."""
    G = _build_nx_graph([])
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_build_nx_graph_single_node():
    """_build_nx_graph with a single node dict creates a graph with 1 node, 0 edges."""
    node = {
        "id": "func://test/file.py::my_func",
        "name": "my_func",
        "type": "Function",
        "file_path": "file.py",
    }
    G = _build_nx_graph([node])
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0
    assert G.has_node("func://test/file.py::my_func")
    # Node attributes are stored
    assert G.nodes["func://test/file.py::my_func"]["name"] == "my_func"


def test_build_nx_graph_with_relationships():
    """_build_nx_graph adds edges for _rels entries with target_id."""
    nodes = [
        {
            "id": "func://test/a.py::foo",
            "name": "foo",
            "type": "Function",
            "file_path": "a.py",
            "_rels": [
                {"target_id": "func://test/b.py::bar", "type": "CALLS"},
                {"target_id": "func://test/c.py::baz", "type": "CALLS"},
            ],
        },
        {
            "id": "func://test/b.py::bar",
            "name": "bar",
            "type": "Function",
            "file_path": "b.py",
        },
        {
            "id": "func://test/c.py::baz",
            "name": "baz",
            "type": "Function",
            "file_path": "c.py",
        },
    ]
    G = _build_nx_graph(nodes)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.has_edge("func://test/a.py::foo", "func://test/b.py::bar")
    assert G.has_edge("func://test/a.py::foo", "func://test/c.py::baz")
    # Edge has the relationship type
    edge_data = G.get_edge_data("func://test/a.py::foo", "func://test/b.py::bar")
    assert edge_data["type"] == "CALLS"


# ---------------------------------------------------------------------------
# summarize_community tests
# ---------------------------------------------------------------------------

def test_summarize_empty_community():
    """Empty entity list produces a summary with 0 entities, 0 files."""
    result = summarize_community([])
    assert result == "0 entities across 0 files: "


def test_summarize_community_single_entity():
    """Single entity produces a summary showing type:name and correct counts."""
    entities = [
        {
            "id": "func://test/file.py::my_func",
            "name": "my_func",
            "type": "Function",
            "file_path": "file.py",
        },
    ]
    result = summarize_community(entities)
    assert result == "1 entities across 1 files: Function:my_func"


def test_summarize_community_multiple_entities():
    """Multiple entities produce a comma-delimited list, capped at 10 names."""
    entities = [
        {
            "id": f"func://test/f.py::func{i}",
            "name": f"func{i}",
            "type": "Function",
            "file_path": "f.py",
        }
        for i in range(3)
    ]
    entities.append(
        {
            "id": "class://test/c.py::MyClass",
            "name": "MyClass",
            "type": "Class",
            "file_path": "c.py",
        }
    )
    result = summarize_community(entities)
    assert result.startswith("4 entities across 2 files: ")
    assert "Function:func0" in result
    assert "Function:func1" in result
    assert "Function:func2" in result
    assert "Class:MyClass" in result
    # Names are comma-separated
    assert ", " in result


def test_summarize_community_no_file_paths():
    """Raw file_path values must NOT appear in the summary output.

    The summary only shows entity type:name pairs, never the raw file paths.
    """
    entities = [
        {
            "id": "func://test/src/main.py::helper",
            "name": "helper",
            "type": "Function",
            "file_path": "src/main.py",
        },
        {
            "id": "func://test/src/utils.py::util",
            "name": "util",
            "type": "Function",
            "file_path": "src/utils.py",
        },
        {
            "id": "mod://test/pkg/__init__.py",
            "name": "pkg/__init__.py",
            "type": "Module",
            "file_path": "src/pkg/__init__.py",
        },
    ]
    result = summarize_community(entities)

    # Raw file_path values must not leak through
    assert "src/main.py" not in result
    assert "src/utils.py" not in result
    assert "src/pkg/__init__.py" not in result

    # The Module entity uses Path(name).name, so only __init__.py appears
    assert "Module:__init__.py" in result
    assert "3 entities across 3 files" in result
