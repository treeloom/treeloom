"""CWE-943: Cypher injection via an unvalidated relationship type.

Neo4j has no query placeholder for a relationship type, so `upsert_relationship`
interpolates it into the MERGE. Unvalidated, a caller supplying

    FOO]->(x) DETACH DELETE x //

closes the pattern and appends arbitrary Cypher to a query running with the
indexer's credentials.

`store_graph` validated its types from the outset; `upsert_relationship` did
not, and that asymmetry is the bug — the same shape as the Milvus
`_build_filter` fix, where one path escaped its input and its sibling did not.

Not exploitable as it stood: the function has no callers in the tree, only a
re-export in `treeloom.graph_store`. It is one caller away from being so, and
relationship types derive from tree-sitter output over indexed source, which
is attacker-influenced for any repository the operator indexes.
"""

import pytest

from treeloom.domain.graph import is_valid_rel_type, require_valid_rel_type


class TestTheRule:
    @pytest.mark.parametrize("t", ["CALLS", "IMPORTS", "INHERITS", "DEFINES",
                                   "_private", "A1", "a_b_2"])
    def test_identifiers_accepted(self, t):
        assert is_valid_rel_type(t) is True

    @pytest.mark.parametrize(
        "t",
        [
            "FOO]->(x) DETACH DELETE x //",   # the injection
            "CALLS]->() MATCH (n) DELETE n",
            "has-dash",                        # dash ends the identifier
            "has space",
            "1STARTS_WITH_DIGIT",
            "back`tick",
            "",
            None,
            123,
            ["CALLS"],
        ],
    )
    def test_everything_else_refused(self, t):
        assert is_valid_rel_type(t) is False

    def test_require_returns_the_value_when_valid(self):
        assert require_valid_rel_type("CALLS") == "CALLS"

    def test_require_raises_with_the_offending_value(self):
        with pytest.raises(ValueError, match="invalid relationship type"):
            require_valid_rel_type("FOO]->(x) DETACH DELETE x //")

    def test_the_error_explains_why_it_cannot_be_parameterised(self):
        """A caller hitting this needs to know it is not an arbitrary
        restriction — Cypher genuinely cannot bind a relationship type."""
        with pytest.raises(ValueError, match="cannot be parameterised"):
            require_valid_rel_type("bad type")


class TestNeo4jUpsertRejectsBeforeQuerying:
    @pytest.mark.asyncio
    async def test_injection_raises_and_runs_no_query(self, mocker):
        """The decisive test: it must refuse BEFORE opening a session, or the
        malicious Cypher has already been built."""
        from treeloom.adapters.neo4j import graph_store as gs

        session = mocker.patch.object(gs, "_get_session")
        with pytest.raises(ValueError):
            await gs.upsert_relationship("a", "b", "FOO]->(x) DETACH DELETE x //")
        session.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_valid_type_still_reaches_the_query(self, mocker):
        """The guard must not break the ordinary path."""
        from treeloom.adapters.neo4j import graph_store as gs

        run = mocker.AsyncMock()
        ctx = mocker.MagicMock()
        ctx.__aenter__ = mocker.AsyncMock(return_value=mocker.Mock(run=run))
        ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch.object(gs, "_get_session", return_value=ctx)

        await gs.upsert_relationship("a", "b", "CALLS")

        run.assert_awaited_once()
        cypher = run.await_args[0][0]
        assert "[r:CALLS]" in cypher
        # endpoints stay parameters — only the type is ever interpolated
        assert "$source_id" in cypher and "$target_id" in cypher


class TestBothBackendsAgree:
    """Same port, same contract. SQLite binds the type as a column value so it
    cannot inject — but a type with no Cypher representation would mean the
    same source produced a different graph depending on which store is
    configured."""

    @pytest.mark.asyncio
    async def test_sqlite_refuses_the_same_input(self, mocker):
        from treeloom.adapters.sqlite import graph_store as gs

        conn = mocker.patch.object(gs, "_get_conn", new=mocker.AsyncMock())
        with pytest.raises(ValueError):
            await gs.upsert_relationship("a", "b", "FOO]->(x) DETACH DELETE x //")
        conn.assert_not_awaited()

    def test_neither_adapter_keeps_its_own_copy_of_the_rule(self):
        """Two regexes drift. The neo4j adapter used to own one; it now
        delegates, so there is a single definition to change."""
        import inspect

        from treeloom.adapters.neo4j import graph_store as n4
        from treeloom.adapters.sqlite import graph_store as sq

        for mod in (n4, sq):
            src = inspect.getsource(mod)
            assert "_VALID_REL_TYPE = re.compile" not in src, (
                f"{mod.__name__} defines its own relationship-type pattern"
            )


class TestBatchPathIsUnchanged:
    """store_graph SKIPS invalid types rather than raising — correct for a
    batch, where one malformed edge should not lose the other 200. Pinned so
    the single-edge fix does not get copied onto it."""

    def test_store_graph_still_skips_rather_than_raises(self):
        import inspect

        from treeloom.adapters.neo4j import graph_store as gs

        src = inspect.getsource(gs.store_graph)
        assert "_is_valid_rel_type" in src
        assert "continue" in src
        assert "require_valid_rel_type" not in src
