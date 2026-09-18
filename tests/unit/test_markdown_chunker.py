"""Markdown ingestion.

Markdown docs were silently dropped: `.md` is absent from LANGUAGE_MAP (because
tree-sitter-markdown's C++ grammar crashes the indexer), so detect_language
returned None and parse_and_chunk returned []. These tests verify markdown is
now detected and chunked via the chonkie RecursiveChunker markdown recipe — a
pure-Python, header-aware splitter that NEVER touches the tree-sitter grammar —
and that graph extraction degrades to a safe Module-only entity.
"""

from __future__ import annotations

from treeloom.indexer import CodeIndexer, detect_language, SUPPORTED_EXTENSIONS
from treeloom.adapters.tree_sitter.indexer import _make_chunker


def _doc(sections: int) -> str:
    """A markdown doc with `sections` heading sections, each big enough that the
    whole thing exceeds CHUNK_SIZE (512 tokens) and must split into >1 chunk."""
    parts = ["# Title\n\nIntro paragraph describing the project.\n"]
    for n in range(sections):
        parts.append(f"## Section {n}\n\n" + ("word " * 90) + "\n")
    return "\n".join(parts)


class TestDetectLanguageMarkdown:
    def test_md_markdown_mdx_detected_as_markdown(self):
        assert detect_language("README.md") == "markdown"
        assert detect_language("docs/guide.markdown") == "markdown"
        assert detect_language("page.mdx") == "markdown"

    def test_code_extensions_unaffected(self):
        assert detect_language("x.py") == "python"
        assert detect_language("x.cs") == "csharp"

    def test_unknown_extension_still_none(self):
        assert detect_language("notes.txt") is None
        assert detect_language("data.bin") is None

    def test_supported_extensions_includes_markdown_and_code(self):
        assert {".md", ".markdown", ".mdx"} <= SUPPORTED_EXTENSIONS
        assert ".py" in SUPPORTED_EXTENSIONS


class TestMarkdownChunking:
    def test_parse_and_chunk_returns_chunks_not_empty(self):
        # Before this returned [] (file skipped entirely).
        chunks = CodeIndexer().parse_and_chunk("docs/design.md", source=_doc(8))
        assert len(chunks) >= 1

    def test_large_doc_splits_into_multiple_chunks(self):
        chunks = CodeIndexer().parse_and_chunk("docs/design.md", source=_doc(8))
        assert len(chunks) > 1

    def test_chunks_tagged_markdown_with_valid_line_ranges(self):
        src = _doc(8)
        n_lines = len(src.splitlines())
        chunks = CodeIndexer().parse_and_chunk("docs/design.md", source=src)
        for c in chunks:
            assert c["language"] == "markdown"
            assert c["file_path"] == "docs/design.md"
            # end_line may be n_lines+1 when the file ends in a newline — a
            # pre-existing property of the shared char-offset→line mapping, not
            # markdown-specific. Assert validity without re-asserting that quirk.
            assert 1 <= c["start_line"] <= c["end_line"] <= n_lines + 1
            assert c["text"].strip()

    def test_first_chunk_starts_at_a_heading_boundary(self):
        chunks = CodeIndexer().parse_and_chunk("docs/design.md", source=_doc(8))
        assert chunks[0]["text"].lstrip().startswith("#")

    def test_empty_or_whitespace_markdown_yields_nothing(self):
        assert CodeIndexer().parse_and_chunk("e.md", source="   \n\n  ") == []

    def test_adversarial_markdown_does_not_crash(self):
        # Input shaped to trip the tree-sitter-markdown grammar (deep heading
        # runs, an unterminated code fence, blockquote/table fragments). The
        # recursive splitter must handle it without crashing the indexer.
        nasty = (
            "#" * 4000 + "\n```\nunterminated fence\n"
            + "> " * 800 + "\n| a | b |\n|---|\n" + "text " * 50
        )
        chunks = CodeIndexer().parse_and_chunk("weird.md", source=nasty)
        assert isinstance(chunks, list)
        assert all(c["language"] == "markdown" for c in chunks)


class TestMarkdownChunkerNeverTreeSitter:
    def test_markdown_routes_to_recursive_chunker(self):
        # The whole point of markdown must NOT build a CodeChunker
        # (which would invoke the crashing tree-sitter markdown grammar).
        chunker = _make_chunker("markdown")
        assert type(chunker).__name__ == "RecursiveChunker"

    def test_code_language_still_uses_codechunker(self):
        chunker = _make_chunker("python")
        assert type(chunker).__name__ == "CodeChunker"


class TestMarkdownGraphFallback:
    def test_extract_graph_returns_module_only_no_crash(self):
        from treeloom.adapters.tree_sitter.graph_extractor import extract_graph

        entities, rels = extract_graph("docs/design.md", _doc(3))
        assert [e["type"] for e in entities] == ["Module"]
        assert rels == []
