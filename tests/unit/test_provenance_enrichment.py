"""Unit tests for _attach_provenance in indexer_service.

Detroit-style: stubs only the out-of-process _source_repo port.
Does NOT mock internal functions (build_provenance, etc.).
"""
from __future__ import annotations

import pytest
from treeloom.application import indexer_runners as _runners
from treeloom.application import lifecycle as _life
from treeloom.application import indexer_state
from treeloom.application import indexer_state as _state
from datetime import datetime, timezone

from treeloom.domain.sources import SourceRecord
from treeloom.application import indexer_service


def _make_record(sid: str = "src-abc") -> SourceRecord:
    return SourceRecord(
        id=sid,
        path="/home/user/repo",
        url="https://github.com/acme/repo.git",
        branch="main",
        commit_sha="deadbeef1234567890ab",
        indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_attach_provenance_populates_sources_and_citation(monkeypatch):
    """Basic happy path: sources map and per-chunk citation are populated."""
    sid = "src-abc"
    rec = _make_record(sid)

    async def fake_get_by_id(source_id):
        if source_id == sid:
            return rec
        return None

    monkeypatch.setattr(indexer_state._source_repo, "get_by_id", fake_get_by_id)

    result = {
        "source_id": sid,
        "chunks": [
            {
                "file_path": "src/main.py",
                "start_line": 1,
                "end_line": 5,
                # chunk's own source_id is None — falls back to top-level
                "source_id": None,
            }
        ],
    }

    await indexer_service._attach_provenance(result, check_staleness=False)

    # sources map must exist and carry commit_sha
    assert "sources" in result
    assert sid in result["sources"]
    assert result["sources"][sid]["commit_sha"] == "deadbeef1234567890ab"

    # per-chunk citation must be set
    chunk = result["chunks"][0]
    assert "citation" in chunk
    assert chunk["citation"]  # non-empty
    # citation should reference the file path
    assert "src/main.py" in chunk["citation"]

    # commit_sha is NOT repeated per chunk — it lives in the sources
    # map; the chunk's citation embeds the short SHA instead.
    assert "commit_sha" not in chunk


@pytest.mark.asyncio
async def test_attach_provenance_with_check_staleness(monkeypatch):
    """check_staleness=True merges is_stale/current_sha into the sources map."""
    sid = "src-xyz"
    rec = _make_record(sid)

    async def fake_get_by_id(source_id):
        return rec if source_id == sid else None

    async def fake_compute_staleness(src: dict) -> dict:
        return {
            "indexed_sha": rec.commit_sha,
            "current_sha": "newsha9999",
            "is_stale": True,
            "resolved_via": "ls-remote",
            "indexed_at": None,
        }

    monkeypatch.setattr(indexer_state._source_repo, "get_by_id", fake_get_by_id)
    monkeypatch.setattr(
        _runners, "compute_source_staleness", fake_compute_staleness
    )

    result = {
        "source_id": sid,
        "chunks": [
            {
                "file_path": "lib/util.py",
                "start_line": 10,
                "end_line": 20,
                "source_id": sid,
            }
        ],
    }

    await indexer_service._attach_provenance(result, check_staleness=True)

    assert "sources" in result
    src_entry = result["sources"][sid]
    assert src_entry.get("is_stale") is True
    assert src_entry.get("current_sha") == "newsha9999"


@pytest.mark.asyncio
async def test_attach_provenance_source_store_error_leaves_result_intact(monkeypatch):
    """A source-store failure must NOT raise — result is returned unchanged."""
    async def exploding_get_by_id(source_id):
        raise RuntimeError("DB is down")

    monkeypatch.setattr(indexer_state._source_repo, "get_by_id", exploding_get_by_id)

    original_chunks = [
        {"file_path": "a.py", "start_line": 1, "end_line": 5, "source_id": "src-err"}
    ]
    result = {
        "source_id": "src-err",
        "chunks": list(original_chunks),
    }

    # Must not raise
    await indexer_service._attach_provenance(result, check_staleness=False)

    # sources key must not be added (or if it is, must not break things)
    # The chunks must still be present
    assert result["chunks"] == original_chunks


@pytest.mark.asyncio
async def test_attach_provenance_no_source_ids_is_a_noop(monkeypatch):
    """Result with no source_id anywhere leaves result unchanged."""
    called = []

    async def should_not_be_called(source_id):
        called.append(source_id)
        return None

    monkeypatch.setattr(indexer_state._source_repo, "get_by_id", should_not_be_called)

    result = {
        "chunks": [
            {"file_path": "a.py", "start_line": 1, "end_line": 5}
        ]
    }

    await indexer_service._attach_provenance(result, check_staleness=False)

    assert "sources" not in result
    assert called == []


@pytest.mark.asyncio
async def test_attach_provenance_chunk_source_id_resolves_independently(monkeypatch):
    """When a chunk has its own source_id different from the top-level one."""
    sid_top = "src-top"
    sid_chunk = "src-chunk"
    rec_top = _make_record(sid_top)
    rec_chunk = SourceRecord(
        id=sid_chunk,
        path="/home/user/other-repo",
        url="",
        branch="dev",
        commit_sha="aaaa1111bbbb",
        indexed_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    async def fake_get_by_id(source_id):
        return {sid_top: rec_top, sid_chunk: rec_chunk}.get(source_id)

    monkeypatch.setattr(indexer_state._source_repo, "get_by_id", fake_get_by_id)

    result = {
        "source_id": sid_top,
        "chunks": [
            {
                "file_path": "pkg/foo.py",
                "start_line": 5,
                "end_line": 15,
                "source_id": sid_chunk,
            }
        ],
    }

    await indexer_service._attach_provenance(result, check_staleness=False)

    assert sid_top in result["sources"]
    assert sid_chunk in result["sources"]
    # chunk's source resolves independently to rec_chunk's SHA, surfaced via the
    # sources map (no longer repeated on the chunk) and its citation.
    assert "commit_sha" not in result["chunks"][0]
    assert result["sources"][sid_chunk]["commit_sha"] == "aaaa1111bbbb"
    assert "aaaa1111bbbb"[:12] in result["chunks"][0]["citation"]
