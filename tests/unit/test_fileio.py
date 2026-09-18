"""Tests for infrastructure.fileio.read_source.

Detroit-style: exercises real behavior with tmp_path; no mocks.
"""
import logging

import pytest

from treeloom.infrastructure import metrics
from treeloom.infrastructure.fileio import read_source


class TestReadSourceValidUtf8:
    """Valid UTF-8 files round-trip exactly and do not increment the counter."""

    def test_valid_utf8_round_trips(self, tmp_path):
        content = "# café\ndef greet():\n    return 'bonjour'\n"
        f = tmp_path / "valid.py"
        f.write_text(content, encoding="utf-8")

        before = metrics.encoding_fallbacks._value.get()
        result = read_source(str(f))
        after = metrics.encoding_fallbacks._value.get()

        assert result == content
        assert after == before  # counter must not advance


class TestReadSourceInvalidUtf8:
    """Non-UTF-8 files are indexed with replacement chars, loudly."""

    def test_returns_text_with_replacement_char(self, tmp_path):
        raw = b"# caf\xe9 comment\ndef f():\n    return 1\n"
        f = tmp_path / "latin1.py"
        f.write_bytes(raw)

        result = read_source(str(f))

        assert "�" in result  # U+FFFD replacement char present

    def test_increments_counter_by_one(self, tmp_path):
        raw = b"# caf\xe9 comment\ndef f():\n    return 1\n"
        f = tmp_path / "counter_check.py"
        f.write_bytes(raw)

        before = metrics.encoding_fallbacks._value.get()
        read_source(str(f))
        after = metrics.encoding_fallbacks._value.get()

        assert after == before + 1

    def test_emits_warning_containing_not_valid_utf8(self, tmp_path, caplog):
        raw = b"# caf\xe9 comment\ndef f():\n    return 1\n"
        f = tmp_path / "warn_check.py"
        f.write_bytes(raw)

        with caplog.at_level(logging.WARNING, logger="treeloom.infrastructure.fileio"):
            read_source(str(f))

        assert any("not valid UTF-8" in r.message for r in caplog.records)


class TestReadSourceMissing:
    """Missing files raise FileNotFoundError (an OSError subclass)."""

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.py"
        with pytest.raises((FileNotFoundError, OSError)):
            read_source(str(missing))


class TestReadSourceIntegrationCodeIndexer:
    """CodeIndexer.parse_and_chunk still returns chunks for a non-UTF-8 file."""

    def test_non_utf8_file_yields_chunks(self, tmp_path):
        # Bytes that are not valid UTF-8 but decode fine with errors="replace".
        # The file is syntactically valid Python after replacement (comment +
        # a plain function body).
        raw = b"# caf\xe9 comment\ndef f():\n    return 1\n"
        f = tmp_path / "legacy.py"
        f.write_bytes(raw)

        from treeloom.indexer import CodeIndexer

        chunks = CodeIndexer().parse_and_chunk(str(f))

        assert len(chunks) > 0
