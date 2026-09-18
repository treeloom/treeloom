"""Domain tests for graph_extractor module.

Tests _detect_language, _resolve_import_path, _fallback_module_entity,
and the main extract_graph function across various languages, file types,
and edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from treeloom.graph_extractor import (
    _detect_language,
    _resolve_import_path,
    _fallback_module_entity,
    extract_graph,
)


def _extract_from_string(source: str, suffix: str = ".py") -> tuple[list, list]:
    """Helper: write source to temp file with given suffix and call extract_graph."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(source)
        file_path = f.name
    # extract_graph uses file_path for language detection and source for parsing
    return extract_graph(file_path, source)


# ---------------------------------------------------------------------------
# _detect_language tests
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    """Language detection from file extensions via LANGUAGE_MAP."""

    def test_detect_language_python(self):
        assert _detect_language("test.py") == "python"

    def test_detect_language_csharp(self):
        assert _detect_language("test.cs") == "csharp"

    def test_detect_language_javascript(self):
        assert _detect_language("test.js") == "javascript"

    def test_detect_language_typescript(self):
        assert _detect_language("test.ts") == "typescript"

    def test_detect_language_go(self):
        assert _detect_language("test.go") == "go"

    def test_detect_language_rust(self):
        assert _detect_language("test.rs") == "rust"

    def test_detect_language_unsupported_returns_none(self):
        assert _detect_language("test.xyz") is None


# ---------------------------------------------------------------------------
# _resolve_import_path tests
# ---------------------------------------------------------------------------

class TestResolveImportPath:
    """Import path resolution for dot-prefix, relative, and absolute modules."""

    def test_resolve_import_path_dot_prefix(self):
        result = _resolve_import_path("./utils", "src/module.py")
        expected = Path.cwd() / "src" / "utils.py"
        assert Path(result) == expected

    def test_resolve_import_path_relative(self):
        result = _resolve_import_path("../shared/helpers", "src/sub/module.py")
        expected = Path.cwd() / "src" / "shared" / "helpers.py"
        assert Path(result) == expected

    def test_resolve_import_path_absolute_module(self):
        result = _resolve_import_path("os.path", "test.py")
        assert result == "os/path.py"


# ---------------------------------------------------------------------------
# extract_graph integration-style tests (via _extract_from_string helper)
# ---------------------------------------------------------------------------

class TestExtractPython:
    """Extraction of Python-specific constructs (class, function, calls)."""

    def test_extract_python_class_with_method(self):
        source = """\
class MyClass:
    def my_method(self):
        pass
"""
        entities, relationships = _extract_from_string(source)

        class_ents = [e for e in entities if e["type"] == "Class"]
        func_ents = [e for e in entities if e["type"] == "Function"]

        assert len(class_ents) == 1
        assert class_ents[0]["name"] == "MyClass"
        assert len(func_ents) == 1
        assert func_ents[0]["name"] == "my_method"

        # Should have DEFINES relationships: module -> class, module -> method
        defines_rels = [r for r in relationships if r["type"] == "DEFINES"]
        assert len(defines_rels) == 2

    def test_extract_python_function_with_calls(self):
        source = """\
def foo():
    pass

def bar():
    foo()
"""
        entities, relationships = _extract_from_string(source)

        func_names = [e["name"] for e in entities if e["type"] == "Function"]
        assert "foo" in func_names
        assert "bar" in func_names

        calls_rels = [r for r in relationships if r["type"] == "CALLS"]
        assert len(calls_rels) == 1
        assert "bar" in calls_rels[0]["source_id"]
        assert "foo" in calls_rels[0]["target_id"]

    def test_extract_empty_file(self):
        entities, relationships = _extract_from_string("")
        module_ents = [e for e in entities if e["type"] == "Module"]
        assert len(module_ents) == 1
        assert len(relationships) == 0

    def test_extract_file_with_only_comments(self):
        entities, relationships = _extract_from_string(
            "# just a comment\n# another comment"
        )
        module_ents = [e for e in entities if e["type"] == "Module"]
        assert len(module_ents) == 1
        # No functions or classes extracted — only the Module entity
        non_module = [e for e in entities if e["type"] != "Module"]
        assert len(non_module) == 0


# ---------------------------------------------------------------------------
# Entity / return-value structural tests
# ---------------------------------------------------------------------------

class TestEntityStructure:
    """Structural assertions on entities and the extract_graph return type."""

    def test_entity_has_required_fields(self):
        entities, _ = _extract_from_string("def foo(): pass")
        required_fields = {
            "id", "type", "name", "file_path",
            "start_line", "end_line", "language",
        }
        for ent in entities:
            missing = required_fields - set(ent.keys())
            assert not missing, f"Entity missing fields: {missing}"

    def test_extract_graph_returns_tuple_of_lists(self):
        result = extract_graph("test.py", "def foo(): pass")
        assert isinstance(result, tuple)
        assert len(result) == 2
        entities, relationships = result
        assert isinstance(entities, list)
        assert isinstance(relationships, list)
