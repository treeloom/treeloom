"""Five findings that are correctness bugs first and security bugs second.

Each had been silently wrong for as long as it existed, which is the point:
none of them raised, logged, or failed a test.
"""

import sqlite3

import pytest


class TestSourceIdEncoding:
    """finding 42 CWE-20. `"|".join(parts)` is ambiguous: a url of "h/r|main" with no
    branch produced the same key — and the same source_id — as url "h/r" on
    branch "main". source_id keys the dedup short-circuit, the ACL grants and
    the vector-store filters, so a crafted url could take over an existing
    source's identity."""

    def test_the_collision_is_closed(self):
        from treeloom.domain.shared import make_source_id

        assert make_source_id(url="h/r|main") != make_source_id(
            url="h/r", branch="main"
        )

    def test_backslash_cannot_forge_the_escape(self):
        """Escaping "|" is only unambiguous if the escape character is escaped
        too, or `h/r\\` + branch collides with `h/r\\|main`."""
        from treeloom.domain.shared import make_source_id

        assert make_source_id(url="h/r\\", branch="main") != make_source_id(
            url="h/r\\|main"
        )

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (dict(url="https://github.com/a/b.git", branch="main"), "c604aac9f038a40f"),
            (dict(url="https://github.com/a/b.git"), "b0467180bfe714c5"),
            (dict(), "0a92fab3230134cc"),
        ],
    )
    def test_existing_ids_are_byte_for_byte_unchanged(self, kwargs, expected):
        """The decisive constraint on the fix. source_id is persisted in
        source_records, referenced by every ACL grant and embedded in every
        Milvus row, so changing the scheme would mean a full reindex and
        orphaned grants. These are the ids the OLD implementation produced,
        pinned as literals."""
        from treeloom.domain.shared import make_source_id

        assert make_source_id(**kwargs) == expected

    def test_determinism_still_holds(self):
        from treeloom.domain.shared import make_source_id

        a = make_source_id(url="https://x/y.git", branch="main")
        b = make_source_id(url="https://x/y.git/", branch=" main ")
        assert a == b, "trailing slash and whitespace still normalise"


class TestStaleTempTable:
    """finding 36 CWE-362. `CREATE TEMP TABLE IF NOT EXISTS _doomed AS SELECT ...`
    does not refresh an existing table — IF NOT EXISTS short-circuits the whole
    statement, SELECT included. The sqlite connection is a reused singleton, so
    from the second delete_source() on, _doomed still held the PREVIOUS
    source's ids."""

    def _doomed(self, conn, src, guarded):
        if guarded:
            conn.execute("DROP TABLE IF EXISTS _doomed")
            stmt = "CREATE TEMP TABLE _doomed AS"
        else:
            stmt = "CREATE TEMP TABLE IF NOT EXISTS _doomed AS"
        conn.execute(
            stmt + """
            SELECT se.entity_id AS id FROM source_entities se
            WHERE se.source_id = ?1
              AND NOT EXISTS (SELECT 1 FROM source_entities o
                              WHERE o.entity_id = se.entity_id
                                AND o.source_id <> ?1)""",
            (src,),
        )
        return sorted(r[0] for r in conn.execute("SELECT id FROM _doomed"))

    @pytest.fixture
    def conn(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE source_entities (source_id TEXT, entity_id TEXT)")
        c.executemany(
            "INSERT INTO source_entities VALUES (?,?)",
            [("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", "b2")],
        )
        return c

    def test_the_bug_reproduces_without_the_drop(self, conn):
        """Pins WHY the DROP is there, so removing it fails loudly."""
        assert self._doomed(conn, "A", guarded=False) == ["a1", "a2"]
        assert self._doomed(conn, "B", guarded=False) == ["a1", "a2"]

    def test_the_drop_fixes_it(self, conn):
        assert self._doomed(conn, "A", guarded=True) == ["a1", "a2"]
        assert self._doomed(conn, "B", guarded=True) == ["b1", "b2"]

    def test_the_store_actually_drops_first(self):
        import inspect

        from treeloom.adapters.sqlite import graph_store

        src = inspect.getsource(graph_store.delete_source)
        assert "DROP TABLE IF EXISTS _doomed" in src
        assert "CREATE TEMP TABLE IF NOT EXISTS _doomed" not in src


class TestEntityDeletionIsScopedByRelationship:
    """finding 39 CWE-284. delete_entities_by_file matched on an `e.source_id`
    PROPERTY that store_graph never writes — the association is the
    (:Source)-[:CONTAINS]->(:Entity) edge. So it matched nothing, always, and
    a file removed via the webhook kept its graph nodes forever."""

    def test_query_traverses_contains_not_a_property(self):
        import inspect

        from treeloom.adapters.neo4j import graph_store

        src = inspect.getsource(graph_store.delete_entities_by_file)
        assert "[:CONTAINS]->" in src
        assert "e.source_id = $source_id" not in src

    def test_store_graph_still_does_not_write_that_property(self):
        """The assumption the fix rests on. If a later change starts writing
        e.source_id, this test says so rather than leaving two mechanisms."""
        import inspect

        from treeloom.adapters.neo4j import graph_store

        src = inspect.getsource(graph_store)
        assert "entity.source_id =" not in src


class TestRerankerIndexValidation:
    """finding 38 CWE-129. `results[idx]` with idx from the reranker — a remote
    service for the cloud providers. Python accepts a negative index silently,
    so -1 scores the LAST chunk as if it were the first and quietly corrupts
    the ranking; an over-range one raises IndexError and fails the search."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_idx", [-1, -5, 99, 3])
    async def test_out_of_range_indices_are_skipped(self, mocker, bad_idx):
        from treeloom.application import retrieval

        results = [
            {"entity": {"chunk_text": f"c{i}"}, "id": i} for i in range(3)
        ]
        mocker.patch.object(
            retrieval._reranker, "rerank_texts",
            new=mocker.AsyncMock(return_value=[(0, 0.9), (bad_idx, 0.99)]),
        )
        out = await retrieval.rerank("q", results, top_k=5)
        assert [r["id"] for r in out] == [0], "only the valid pair scored"
        assert "relevance_score" not in results[2] or bad_idx == 2

    @pytest.mark.asyncio
    async def test_valid_indices_are_unaffected(self, mocker):
        from treeloom.application import retrieval

        results = [
            {"entity": {"chunk_text": f"c{i}"}, "id": i} for i in range(3)
        ]
        mocker.patch.object(
            retrieval._reranker, "rerank_texts",
            new=mocker.AsyncMock(return_value=[(0, 0.1), (1, 0.9), (2, 0.5)]),
        )
        out = await retrieval.rerank("q", results, top_k=3)
        assert [r["id"] for r in out] == [1, 2, 0]


class TestJobStorePoolGuard:
    """finding 50 CWE-754. Ten of eleven call sites went straight from get_pool() to
    pool.acquire(), so a None pool surfaced as `AttributeError: 'NoneType'
    object has no attribute 'acquire'` from a background worker — a traceback
    that names neither the database nor the job, while job state is lost."""

    @pytest.mark.asyncio
    async def test_raises_a_message_that_names_the_cause(self, mocker):
        from treeloom.adapters.postgresql import job_store

        mocker.patch.object(
            job_store, "get_pool", new=mocker.AsyncMock(return_value=None)
        )
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            await job_store._require_pool()

    def test_no_call_site_uses_the_raw_getter(self):
        import inspect

        from treeloom.adapters.postgresql import job_store

        src = inspect.getsource(job_store)
        body = src.split("async def _require_pool")[1]
        rest = src.replace("async def _require_pool" + body.split("\n\n\n")[0], "")
        assert "await get_pool()" not in rest, (
            "a call site is bypassing the guard"
        )
