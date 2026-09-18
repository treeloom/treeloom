"""Detroit-style unit tests for SourceRepositoryPort → PostgreSourceRepository.

Mock only the asyncpg pool (out-of-process dependency). The real
PostgreSourceRepository class is tested; no internal mocking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from treeloom.domain.sources import SourceRecord, SourceRepositoryPort
from treeloom.domain.shared import make_source_id


# ---------------------------------------------------------------------------
# Helpers — deterministic test records
# ---------------------------------------------------------------------------

def _make_record(
    path: str = "/tmp/test_repo",
    url: str = "",
    branch: str = "main",
    file_count: int = 42,
    chunk_count: int = 128,
) -> SourceRecord:
    sid = make_source_id(path=path, url=url, branch=branch)
    return SourceRecord(
        id=sid,
        path=path,
        url=url,
        branch=branch,
        indexed_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        file_count=file_count,
        chunk_count=chunk_count,
    )


# ---------------------------------------------------------------------------
# Mock asyncpg pool (out-of-process dep only)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pool() -> AsyncMock:
    """A mock asyncpg.Pool that tracks executed SQL and returned rows."""
    pool = AsyncMock()

    # Internal storage dict that simulates the database
    _store: dict[str, dict] = {}

    # Mock connection
    conn = AsyncMock()

    async def _fetchrow(query, *args):
        """Simulate SELECT by id — return one row or None."""
        if "SELECT" in query and "FROM source_records" in query:
            rid = args[0] if args else None
            if rid and rid in _store:
                row = _store[rid]
                mock_row = MagicMock()
                mock_row.__getitem__ = lambda self, k: row.get(k)
                mock_row.keys.return_value = list(row.keys())
                # Support .get() access too
                mock_row.get = row.get
                mock_row.values.return_value = list(row.values())
                return mock_row
        return None

    async def _fetch(query, *args):
        """Simulate SELECT list_all — return all rows ordered by indexed_at DESC."""
        if "SELECT" in query and "FROM source_records" in query:
            rows = sorted(_store.values(), key=lambda r: r.get("indexed_at", datetime.min), reverse=True)
            results = []
            for row_data in rows:
                mock_row = MagicMock()
                mock_row.__getitem__ = lambda self, k, rd=row_data: rd.get(k)
                mock_row.keys.return_value = list(row_data.keys())
                mock_row.get = row_data.get
                mock_row.values.return_value = list(row_data.values())
                results.append(mock_row)
            return results
        return []

    async def _execute(query, *args):
        """Simulate INSERT/UPDATE/DELETE."""
        if "INSERT" in query and "source_records" in query:
            # args: id, path, url, branch, indexed_at, file_count, chunk_count, created_by
            rid = args[0]
            _store[rid] = {
                "id": args[0],
                "path": args[1] if len(args) > 1 else "",
                "url": args[2] if len(args) > 2 else "",
                "branch": args[3] if len(args) > 3 else "",
                "indexed_at": args[4] if len(args) > 4 else datetime.now(timezone.utc),
                "file_count": args[5] if len(args) > 5 else 0,
                "chunk_count": args[6] if len(args) > 6 else 0,
                "created_by": args[7] if len(args) > 7 else None,
            }
        elif "DELETE" in query and "source_records" in query:
            rid = args[0] if args else None
            if rid and rid in _store:
                del _store[rid]
                return "DELETE 1"
            return "DELETE 0"
        return "OK"

    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.fetch = AsyncMock(side_effect=_fetch)
    conn.execute = AsyncMock(side_effect=_execute)

    # Make pool.acquire() return conn as an async context manager
    _acquire_ctx = MagicMock()
    _acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    _acquire_ctx.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=_acquire_ctx)

    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSourceRepositorySaveAndGet:
    """T011: save() inserts a SourceRecord; get_by_id() retrieves it."""

    @pytest.mark.asyncio
    async def test_source_repository_save_and_get(self, mock_pool):
        """Save a SourceRecord, then retrieve it by id."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)
        record = _make_record(path="/tmp/myproject", branch="main")

        # Save
        await repo.save(record)

        # Get back
        result = await repo.get_by_id(record.id)
        assert result is not None
        assert result.id == record.id
        assert result.path == record.path
        assert result.url == record.url
        assert result.branch == record.branch
        assert result.file_count == record.file_count
        assert result.chunk_count == record.chunk_count

    @pytest.mark.asyncio
    async def test_source_repository_get_nonexistent(self, mock_pool):
        """get_by_id returns None for nonexistent source."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)
        result = await repo.get_by_id("nonexistent-id")
        assert result is None


class TestSourceRepositoryUpsert:
    """T011: upsert — second save with same id updates the record."""

    @pytest.mark.asyncio
    async def test_source_repository_upsert_on_reindex(self, mock_pool):
        """Save same id twice → second save updates existing record."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)
        sid = make_source_id(path="/tmp/upsert-test")

        # First save
        record1 = SourceRecord(
            id=sid,
            path="/tmp/upsert-test",
            branch="main",
            file_count=10,
            chunk_count=50,
        )
        await repo.save(record1)

        # Second save — same id, different counts
        record2 = SourceRecord(
            id=sid,
            path="/tmp/upsert-test",
            branch="main",
            file_count=20,
            chunk_count=100,
        )
        await repo.save(record2)

        # Retrieve — should reflect second save
        result = await repo.get_by_id(sid)
        assert result is not None
        assert result.file_count == 20
        assert result.chunk_count == 100


class TestSourceRepositoryListAndDelete:
    """T011: list_all returns all records ordered by indexed_at DESC;
    delete removes a record and returns True."""

    @pytest.mark.asyncio
    async def test_source_repository_list_and_delete(self, mock_pool):
        """Save two records, list_all returns both, delete removes one."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)

        # Save two records
        record_a = SourceRecord(
            id=make_source_id(path="/tmp/proj-a"),
            path="/tmp/proj-a",
            file_count=5,
            chunk_count=10,
        )
        record_b = SourceRecord(
            id=make_source_id(path="/tmp/proj-b"),
            path="/tmp/proj-b",
            file_count=15,
            chunk_count=30,
        )
        await repo.save(record_a)
        await repo.save(record_b)

        # List all
        all_sources = await repo.list_all()
        assert len(all_sources) == 2
        ids = {s.id for s in all_sources}
        assert record_a.id in ids
        assert record_b.id in ids

        # Delete record_a
        deleted = await repo.delete(record_a.id)
        assert deleted is True

        # List again — only record_b remains
        remaining = await repo.list_all()
        assert len(remaining) == 1
        assert remaining[0].id == record_b.id

    @pytest.mark.asyncio
    async def test_source_repository_delete_nonexistent(self, mock_pool):
        """Delete nonexistent id returns False."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)
        deleted = await repo.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_source_repository_list_empty(self, mock_pool):
        """list_all returns empty list when no records exist."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)
        result = await repo.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_source_repository_list_ordered_by_indexed_at(self, mock_pool):
        """list_all returns records ordered by indexed_at DESC (most recent first)."""
        from treeloom.adapters.sources.repository import PostgreSourceRepository

        repo = PostgreSourceRepository(pool=mock_pool)

        older = SourceRecord(
            id=make_source_id(path="/tmp/older"),
            path="/tmp/older",
            indexed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        newer = SourceRecord(
            id=make_source_id(path="/tmp/newer"),
            path="/tmp/newer",
            indexed_at=datetime(2025, 6, 15, tzinfo=timezone.utc),
        )
        await repo.save(older)
        await repo.save(newer)

        results = await repo.list_all()
        assert len(results) == 2
        # Most recently indexed first
        assert results[0].id == newer.id
        assert results[1].id == older.id


class TestRowToRecordKeysIterator:
    """Regression: asyncpg's Record.keys() is a single-use iterator. The
    repeated `"col" in keys` membership checks in _row_to_record must not
    consume it — otherwise a column (e.g. commit_sha) positioned in table order
    BEFORE one checked earlier reads as "missing" and silently falls back to its
    default, so every source reports an empty commit_sha regardless of storage.
    """

    class _OneShotKeysRow:
        """Minimal asyncpg.Record stand-in whose keys() returns a fresh
        single-use iterator each call (the real Record.keys() is a
        tuple_iterator), so the consuming-membership bug reproduces."""

        def __init__(self, data: dict):
            self._d = data

        def keys(self):
            return iter(self._d.keys())

        def __getitem__(self, k):
            return self._d[k]

        def get(self, k, default=None):
            return self._d.get(k, default)

    def test_commit_sha_read_when_keys_is_one_shot_iterator(self):
        from treeloom.adapters.sources.repository import _row_to_record

        # Column order mirrors the real table: commit_sha precedes the
        # graph_indexed* columns, which _row_to_record checks first.
        row = self._OneShotKeysRow({
            "id": "abc123",
            "path": "/repo",
            "url": "https://example.com/org/repo",
            "branch": "main",
            "indexed_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "file_count": 3,
            "chunk_count": 9,
            "commit_sha": "a" * 40,
            "created_by": None,
            "graph_indexed": True,
            "graph_indexed_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "kind": "directory",
        })

        rec = _row_to_record(row)

        # The fields guarded by `"col" in keys` must all be read, not defaulted.
        assert rec.commit_sha == "a" * 40
        assert rec.kind == "directory"
        assert rec.graph_indexed is True
        assert rec.graph_indexed_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
