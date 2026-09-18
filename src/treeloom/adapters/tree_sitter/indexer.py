import bisect
import logging
import os
from pathlib import Path

import tiktoken
from chonkie import CodeChunker, RecursiveChunker, TokenChunker
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import get_language

from treeloom.infrastructure.fileio import read_source

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".cs": "csharp",
    ".jsx": "javascript",
    ".html": "html",
    ".css": "css",
}

# Markdown is deliberately kept OUT of LANGUAGE_MAP — tree-sitter-markdown's C++
# grammar asserts and crashes the indexer, and graph_extractor._detect_language
# reads LANGUAGE_MAP, so keeping .md out routes every markdown file to the safe
# Module-only graph fallback (no grammar). Detection + chunking for these
# extensions is handled separately via the chonkie RecursiveChunker markdown
# recipe (a pure-Python, header-aware splitter), never CodeChunker/tree-sitter.
MARKDOWN_MAP = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
}

# Single source of truth for "is this file indexable?" — used by the repo/
# directory walk filter (indexer_service._collect_files). Kept as a union so it
# can never drift from the two maps above.
SUPPORTED_EXTENSIONS = set(LANGUAGE_MAP) | set(MARKDOWN_MAP)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
FALLBACK_CHUNK_CHARS = int(os.environ.get("FALLBACK_CHUNK_CHARS", "2048"))

# Pre-download once — cached offline after first call
_tiktoken_enc = tiktoken.get_encoding("cl100k_base")


def _character_chunks(source: str, file_path: str, lang: str) -> list[dict]:
    """Split source into fixed-size character chunks when chunking fails.

    Last-resort fallback for files neither the structural nor the token
    chunker can handle. Each chunk is FALLBACK_CHUNK_CHARS characters,
    split at newline boundaries.
    """
    lines = source.split("\n")
    results: list[dict] = []
    buf = ""
    start_line = 1
    for i, line in enumerate(lines, start=1):
        if len(buf) + len(line) > FALLBACK_CHUNK_CHARS and buf:
            results.append({
                "text": buf.rstrip(),
                "file_path": file_path,
                "language": lang,
                "start_line": start_line,
                "end_line": i - 1,
            })
            buf = line + "\n"
            start_line = i
        else:
            buf += line + "\n"
    if buf.strip():
        results.append({
            "text": buf.rstrip(),
            "file_path": file_path,
            "language": lang,
            "start_line": start_line,
            "end_line": len(lines),
        })
    return results


def detect_language(file_path: str) -> str | None:
    suffix = Path(file_path).suffix
    return LANGUAGE_MAP.get(suffix) or MARKDOWN_MAP.get(suffix)


def _make_markdown_chunker():
    """Header-aware chunker for Markdown docs — NEVER the tree-sitter path.

    Markdown is excluded from LANGUAGE_MAP (tree-sitter-markdown crashes the
    indexer), so it must not go through CodeChunker. chonkie's RecursiveChunker
    `markdown` recipe is a pure-Python recursive splitter whose top level breaks
    on heading markers (######..#), then paragraphs, then lines — so chunks
    start at section boundaries. Its chunks expose the same
    .text/.start_index/.end_index surface as CodeChunker, so the line-mapping in
    parse_and_chunk works unchanged. Falls back to token windows if the recipe
    is unavailable.
    """
    try:
        return RecursiveChunker.from_recipe(
            name="markdown", tokenizer=_tiktoken_enc, chunk_size=CHUNK_SIZE
        )
    except Exception:
        logger.warning(
            "markdown recipe unavailable — using token windows for markdown"
        )
        return TokenChunker(tokenizer=_tiktoken_enc, chunk_size=CHUNK_SIZE)


def _make_chunker(lang: str):
    """Build a structural chunker for `lang`, falling back to token windows.

    Markdown routes to the dedicated recursive splitter (never tree-sitter).
    For code, CodeChunker aligns chunks to tree-sitter node boundaries
    (functions/classes stay whole), which embeds better and returns snippets
    that start at definition boundaries. chonkie's own parser setup calls
    tree_sitter_language_pack.get_parser(), which since 1.8.x returns a Rust
    pyo3 parser whose API chonkie can't drive (parse() wants str, Tree.root_node
    is a method) — so we overwrite `.parser` with the Python-binding parser via
    get_language(), the same path graph_extractor uses. Built per call: parsers
    are not safe for concurrent parse() and parse_and_chunk runs in threads.
    """
    if lang == "markdown":
        return _make_markdown_chunker()
    try:
        chunker = CodeChunker(
            tokenizer=_tiktoken_enc, chunk_size=CHUNK_SIZE, language=lang
        )
        chunker.parser = TSParser(get_language(lang))
        return chunker
    except Exception:
        logger.warning(
            "no structural chunker for language %r — using token windows", lang
        )
        return TokenChunker(tokenizer=_tiktoken_enc, chunk_size=CHUNK_SIZE)


class CodeIndexer:
    def parse_and_chunk(self, file_path: str, source: str | None = None) -> list[dict]:
        lang = detect_language(file_path)
        if not lang:
            return []

        if source is None:
            source = read_source(file_path)

        if not source.strip():
            return []

        chunker = _make_chunker(lang)

        try:
            raw_chunks = chunker.chunk(source)
        except Exception:
            # Chunking failed outright (malformed input the grammar can't
            # recover from, chonkie internals). Fall back to character-based
            # chunking instead of returning one massive chunk that may OOM
            # embed servers.
            logger.warning(
                "chunking failed for %s — falling back to character chunking",
                file_path, exc_info=True,
            )
            return _character_chunks(source, file_path, lang)

        # Precompute newline offsets so each chunk's line lookup is O(log N)
        # via bisect, not O(N) via prefix-slice + count. Naive line counting
        # made large source files (TypeScript .d.ts, etc.) the dominant cost
        # of indexing — see py-spy dump at parse_and_chunk:77 for context.
        newline_offsets: list[int] = []
        pos = source.find("\n")
        while pos != -1:
            newline_offsets.append(pos)
            pos = source.find("\n", pos + 1)

        def line_at(offset: int) -> int:
            # 1-indexed line: count of newlines strictly before `offset`, +1.
            return bisect.bisect_left(newline_offsets, offset) + 1

        results = []
        for c in raw_chunks:
            results.append({
                "text": c.text,
                "file_path": file_path,
                "language": lang,
                "start_line": line_at(c.start_index),
                "end_line": line_at(c.end_index),
            })
        return results
