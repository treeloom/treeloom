"""SQLite graph store — real tmp-file DB (in-process = no mock).

Focus: the load-bearing Neo4j MERGE semantics CLAUDE.md warns about
(mid-job Source creation, ExternalModule auto-typing), delete_source
exclusivity, traversal, and the community round-trip that
run_post_index_signals depends on.
"""
import pytest
import pytest_asyncio

from treeloom.adapters.sqlite import graph_store as gs


@pytest_asyncio.fixture(autouse=True)
async def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "GRAPH_DB_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setattr(gs, "_conn", None)
    monkeypatch.setattr(gs, "_conn_lock", None)
    await gs.ensure_schema()
    yield
    await gs.close()


def _entity(i: int, **kw) -> dict:
    base = {
        "id": f"py:///repo/f{i}.py#func_{i}",
        "type": "Function",
        "name": f"func_{i}",
        "file_path": f"/repo/f{i}.py",
        "start_line": 1,
        "end_line": 20,
        "signature": f"def func_{i}()",
        "language": "python",
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
async def test_store_graph_creates_source_mid_job():
    """Entities stored BEFORE upsert_source must not be dropped — the
    MATCH-vs-MERGE Source regression from CLAUDE.md."""
    await gs.store_graph([_entity(1)], [], source_id="src-1")
    ents = await gs.get_entities_by_file("/repo/f1.py", source_id="src-1")
    assert [e["name"] for e in ents] == ["func_1"]
    # the bare source row exists and upsert_source later enriches it
    await gs.upsert_source({"id": "src-1", "path": "/repo", "file_count": 9})
    sources = await gs.list_sources()
    assert sources[0]["id"] == "src-1" and sources[0]["file_count"] == 9


@pytest.mark.asyncio
async def test_store_graph_upsert_idempotent():
    await gs.store_graph([_entity(1)], [], source_id="src-1")
    await gs.store_graph([_entity(1, end_line=99)], [], source_id="src-1")
    ents = await gs.get_entities_by_file("/repo/f1.py")
    assert len(ents) == 1 and ents[0]["end_line"] == 99


@pytest.mark.asyncio
async def test_relationship_pass_auto_creates_external_module_targets():
    e = _entity(1)
    await gs.store_graph(
        [e],
        [{"source_id": e["id"], "target_id": "ext://numpy", "type": "IMPORTS"}],
        source_id="src-1",
    )
    ext = await gs.get_entity_by_id("ext://numpy")
    assert ext is not None and ext["type"] == "ExternalModule"
    # a later real definition of the target keeps its real type
    await gs.store_graph(
        [_entity(2, id="ext://numpy", type="Module", name="numpy")], [],
        source_id="src-2",
    )
    assert (await gs.get_entity_by_id("ext://numpy"))["type"] == "Module"


@pytest.mark.asyncio
async def test_external_typing_never_overwrites_existing_type():
    e1, e2 = _entity(1), _entity(2)
    await gs.store_graph([e1, e2], [], source_id="src-1")
    await gs.store_graph(
        [], [{"source_id": e1["id"], "target_id": e2["id"], "type": "CALLS"}],
        source_id="src-1",
    )
    assert (await gs.get_entity_by_id(e2["id"]))["type"] == "Function"


@pytest.mark.asyncio
async def test_invalid_rel_type_skipped():
    e1, e2 = _entity(1), _entity(2)
    await gs.store_graph(
        [e1, e2],
        [{"source_id": e1["id"], "target_id": e2["id"], "type": "BAD TYPE!"}],
        source_id="src-1",
    )
    assert await gs.find_references(e2["id"]) == []


@pytest.mark.asyncio
async def test_delete_source_keeps_shared_entities():
    shared = _entity(1)
    only_a = _entity(2)
    await gs.store_graph([shared, only_a], [], source_id="src-a")
    await gs.store_graph([shared], [], source_id="src-b")
    await gs.delete_source("src-a")
    assert await gs.get_entity_by_id(only_a["id"]) is None
    assert await gs.get_entity_by_id(shared["id"]) is not None
    assert [s["id"] for s in await gs.list_sources()] == ["src-b"]


@pytest.mark.asyncio
async def test_callers_callees_references_with_rel_metadata():
    a, b, c = _entity(1), _entity(2), _entity(3, type="Class", name="Base")
    await gs.store_graph(
        [a, b, c],
        [
            {"source_id": a["id"], "target_id": b["id"], "type": "CALLS"},
            {"source_id": b["id"], "target_id": c["id"], "type": "INHERITS"},
        ],
        source_id="src-1",
    )
    callers = await gs.find_callers(b["id"])
    assert [(x["id"], x["_rel_type"], x["_rel_direction"]) for x in callers] == [
        (a["id"], "CALLS", "in")
    ]
    callees = await gs.find_callees(a["id"])
    assert [x["id"] for x in callees] == [b["id"]]
    refs = await gs.find_references(c["id"])
    assert [(x["id"], x["_rel_type"]) for x in refs] == [(b["id"], "INHERITS")]


@pytest.mark.asyncio
async def test_traverse_depth_and_neighbors_shape():
    a, b, c = _entity(1), _entity(2), _entity(3)
    await gs.store_graph(
        [a, b, c],
        [
            {"source_id": a["id"], "target_id": b["id"], "type": "CALLS"},
            {"source_id": b["id"], "target_id": c["id"], "type": "CALLS"},
        ],
        source_id="src-1",
    )
    nodes = await gs.traverse(a["id"], depth=1)
    by_id = {n["id"]: n for n in nodes}
    assert set(by_id) == {a["id"], b["id"]}  # depth 1: a + neighbor b
    nb = by_id[a["id"]]["_neighbors"][0]
    assert nb["id"] == b["id"] and nb["_rel_direction"] == "out"
    deep = await gs.traverse(a["id"], depth=2)
    assert {n["id"] for n in deep} == {a["id"], b["id"], c["id"]}


@pytest.mark.asyncio
async def test_get_neighbors_batch_skips_unknown_ids():
    a, b = _entity(1), _entity(2)
    await gs.store_graph(
        [a, b],
        [{"source_id": a["id"], "target_id": b["id"], "type": "CALLS"}],
        source_id="src-1",
    )
    out = await gs.get_neighbors_batch([a["id"], "missing"])
    assert set(out) == {a["id"]}
    assert [n["id"] for n in out[a["id"]]] == [b["id"]]


@pytest.mark.asyncio
async def test_find_entity_innermost_by_line():
    outer = _entity(1, id="outer", name="Outer", type="Class",
                    start_line=1, end_line=100)
    inner = _entity(1, id="inner", name="method", type="Method",
                    start_line=10, end_line=20)
    await gs.store_graph([outer, inner], [], source_id="src-1")
    hit = await gs.find_entity("/repo/f1.py", 15)
    assert hit["id"] == "inner"
    assert (await gs.find_entity("/repo/f1.py", 999)) is None


@pytest.mark.asyncio
async def test_find_entities_by_name_scoped_and_typed():
    await gs.store_graph([_entity(1)], [], source_id="src-a")
    await gs.store_graph([_entity(1, id="other#func_1")], [], source_id="src-b")
    all_hits = await gs.find_entities_by_name("func_1")
    assert len(all_hits) == 2
    scoped = await gs.find_entities_by_name("func_1", source_id="src-a")
    assert [e["id"] for e in scoped] == ["py:///repo/f1.py#func_1"]
    assert await gs.find_entities_by_name("func_1", entity_type="Class") == []


@pytest.mark.asyncio
async def test_community_round_trip_for_post_index_signals():
    """The exact write/read cycle run_post_index_signals + search use."""
    a, b = _entity(1), _entity(2)
    await gs.store_graph([a, b], [], source_id="src-1")
    await gs.set_community_ids_batch({a["id"]: "src-1#0", b["id"]: "src-1#1"})
    await gs.set_centralities_batch({a["id"]: 0.5})
    await gs.upsert_community_summaries_batch(
        {"src-1#0": "2 entities: func_1", "src-1#1": "1 entity: func_2"}
    )
    await gs.set_community_embeddings_batch({"src-1#0": [0.1, 0.2]})

    assert await gs.get_community_ids([a["id"], b["id"]]) == {"src-1#0", "src-1#1"}
    assert (await gs.get_entity_by_id(a["id"]))["centrality"] == 0.5
    assert (await gs.get_community_summaries_map(["src-1#0"])) == {
        "src-1#0": "2 entities: func_1"
    }
    assert len(await gs.get_all_community_summaries()) == 2
    assert await gs.load_community_embeddings() == {"src-1#0": [0.1, 0.2]}
    # re-run clears stale communities (count shrinkage case)
    await gs.delete_source_communities("src-1")
    assert await gs.get_all_community_summaries() == {}


@pytest.mark.asyncio
async def test_pull_graph_shape_for_louvain():
    a, b = _entity(1), _entity(2)
    await gs.store_graph(
        [a, b],
        [{"source_id": a["id"], "target_id": b["id"], "type": "CALLS"}],
        source_id="src-1",
    )
    nodes = await gs.pull_graph(source_id="src-1")
    by_id = {n["id"]: n for n in nodes}
    assert set(by_id) == {a["id"], b["id"]}
    assert by_id[a["id"]]["_rels"] == [{"type": "CALLS", "target_id": b["id"]}]
    assert set(by_id[a["id"]]) == {"id", "name", "type", "file_path", "_rels"}


@pytest.mark.asyncio
async def test_delete_entities_by_file_scoped_count():
    a = _entity(1)
    b = _entity(2, file_path="/repo/f1.py", id="b#x")
    other_src = _entity(3, file_path="/repo/f1.py", id="c#y")
    await gs.store_graph([a, b], [], source_id="src-a")
    await gs.store_graph([other_src], [], source_id="src-b")
    n = await gs.delete_entities_by_file("src-a", "/repo/f1.py")
    assert n == 2
    assert await gs.get_entity_by_id("c#y") is not None


@pytest.mark.asyncio
async def test_list_entities_filters():
    await gs.store_graph(
        [
            _entity(1),
            _entity(2, type="Class", name="Foo", file_path="/repo/tests/t.py"),
        ],
        [],
        source_id="src-1",
    )
    out = await gs.list_entities(["Function", "Class"], source_id="src-1",
                                 exclude=["/tests/"])
    assert [e["name"] for e in out] == ["func_1"]
    out = await gs.list_entities(["Class"], path_prefix="/repo/tests/")
    assert [e["name"] for e in out] == ["Foo"]


@pytest.mark.asyncio
async def test_list_entities_default_excludes_build_artifacts():
    """DEFAULT_EXCLUDES must drop build/dist/minified ground truth.

    A query anchored to a generated bundle (e.g. build/three.core.js) is
    unanswerable — no agent reads a minified bundle — which silently zeroes
    recall for every arm.
    """
    from treeloom.application.benchmark.enhanced_query_gen import DEFAULT_EXCLUDES

    assert "/build/" in DEFAULT_EXCLUDES
    assert "/dist/" in DEFAULT_EXCLUDES
    assert ".min." in DEFAULT_EXCLUDES

    await gs.store_graph(
        [
            _entity(1, file_path="/repo/src/widget.js"),
            _entity(2, name="bundled", file_path="/repo/build/three.core.js"),
            _entity(3, name="dist_fn", file_path="/repo/dist/app.js"),
            _entity(4, name="minified", file_path="/repo/static/app.min.js"),
        ],
        [],
        source_id="src-1",
    )
    out = await gs.list_entities(
        ["Function", "Class"], source_id="src-1", exclude=DEFAULT_EXCLUDES
    )
    assert [e["name"] for e in out] == ["func_1"]


@pytest.mark.asyncio
async def test_clear_all_and_upsert_relationship_match_semantics():
    a, b = _entity(1), _entity(2)
    await gs.store_graph([a], [], source_id="src-1")
    # MATCH...MATCH parity: edge to a nonexistent node is a no-op
    await gs.upsert_relationship(a["id"], "ghost", "CALLS")
    assert await gs.find_callees(a["id"]) == []
    await gs.store_graph([b], [], source_id="src-1")
    await gs.upsert_relationship(a["id"], b["id"], "CALLS")
    assert [x["id"] for x in await gs.find_callees(a["id"])] == [b["id"]]
    await gs.clear_all()
    assert await gs.list_sources() == []
    assert await gs.get_entity_by_id(a["id"]) is None


# ── per-source authorization scoping ───────────


@pytest.mark.asyncio
async def test_neighbors_and_callers_scoped_by_source():
    """A relationship crossing two sources must not leak the other source's
    entity into a query scoped to (or excluding) one of them."""
    a = _entity(1)  # lives in src-a
    b = _entity(2)  # lives in src-b
    await gs.store_graph([a], [], source_id="src-a")
    await gs.store_graph(
        [b],
        [{"source_id": a["id"], "target_id": b["id"], "type": "CALLS"}],
        source_id="src-b",
    )

    # neighbors of A: B is the only neighbor, and it lives in src-b.
    same = await gs.get_neighbors_batch([a["id"]], source_id="src-b")
    assert [n["id"] for n in same[a["id"]]] == [b["id"]]
    other = await gs.get_neighbors_batch([a["id"]], source_id="src-a")
    assert other[a["id"]] == []  # B not in src-a -> hidden
    excluded = await gs.get_neighbors_batch(
        [a["id"]], exclude_source_ids=["src-b"]
    )
    assert excluded[a["id"]] == []  # B excluded

    # callers of B: A lives in src-a.
    assert [c["id"] for c in await gs.find_callers(b["id"], source_id="src-a")] == [a["id"]]
    assert await gs.find_callers(b["id"], source_id="src-b") == []
    assert await gs.find_callers(b["id"], exclude_source_ids=["src-a"]) == []


@pytest.mark.asyncio
async def test_find_entities_by_name_excludes_denied_source():
    shared = _entity(1, id="dup", name="dup")
    await gs.store_graph([shared], [], source_id="src-a")
    # same-named entity also registered under a denied source
    await gs.store_graph([_entity(2, id="dup2", name="dup")], [], source_id="src-secret")
    names = await gs.find_entities_by_name("dup", exclude_source_ids=["src-secret"])
    assert {e["id"] for e in names} == {"dup"}


# ── orphan-entity leak regression ──────────────────────────
#
# An "orphan" is an entity with NO source membership: in sqlite that means a
# row in `entities` with no matching row in `source_entities`; in Neo4j it
# means an :Entity with no incoming (:Source)-[:CONTAINS]->. Two leak paths:
#   1. delete_source strands an entity that WAS contained by the source (the
#      regression these tests guard — must NOT happen).
#   2. the relationship pass MERGEs bare external-reference TARGETS that are
#      never linked to a source (EXPECTED — characterized, not "fixed"; see
#      scripts/audit_orphan_entities.py for the cleanup policy).


async def _orphan_entity_ids() -> set[str]:
    """Entity ids present in `entities` but absent from `source_entities`.

    The sqlite analogue of the Neo4j orphan query
    `MATCH (e:Entity) WHERE NOT EXISTS { (:Source)-[:CONTAINS]->(e) }`.
    """
    conn = await gs._get_conn()
    cur = await conn.execute(
        "SELECT e.id FROM entities e "
        "WHERE NOT EXISTS (SELECT 1 FROM source_entities se "
        "                  WHERE se.entity_id = e.id)"
    )
    return {r["id"] for r in await cur.fetchall()}


@pytest.mark.asyncio
async def test_delete_source_leaves_no_orphan_from_its_own_entities():
    """After deleting the only source, NONE of its indexed entities remain as
    orphan rows. This is the core invariant for the live delete path."""
    ents = [_entity(i) for i in range(1, 6)]
    await gs.store_graph(
        ents,
        [{"source_id": ents[0]["id"], "target_id": ents[1]["id"], "type": "CALLS"}],
        source_id="src-1",
    )
    # all five entities are source-contained -> zero orphans pre-delete
    assert await _orphan_entity_ids() == set()

    await gs.delete_source("src-1")

    # every indexed entity is gone (not stranded), and there are no orphans
    for e in ents:
        assert await gs.get_entity_by_id(e["id"]) is None
    assert await _orphan_entity_ids() == set()
    # the entities table is empty of this source's rows entirely
    conn = await gs._get_conn()
    cur = await conn.execute("SELECT COUNT(*) AS n FROM entities")
    assert (await cur.fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_delete_source_does_not_orphan_shared_or_other_source_entities():
    """delete_source must not strand entities that belong to ANOTHER source —
    they stay both present AND source-contained (never become orphans)."""
    shared = _entity(1)
    only_a = _entity(2)
    only_b = _entity(3)
    await gs.store_graph([shared, only_a], [], source_id="src-a")
    await gs.store_graph([shared, only_b], [], source_id="src-b")

    await gs.delete_source("src-a")

    # only_a removed; shared + only_b survive AND remain contained (no orphan)
    assert await gs.get_entity_by_id(only_a["id"]) is None
    assert await gs.get_entity_by_id(shared["id"]) is not None
    assert await gs.get_entity_by_id(only_b["id"]) is not None
    assert await _orphan_entity_ids() == set()


@pytest.mark.asyncio
async def test_merge_target_orphan_is_characterized_not_reaped_by_delete():
    """CHARACTERIZATION (not a "should be empty" assertion): the relationship
    pass auto-creates a bare ExternalModule TARGET that is never added to
    source_entities, so it is already an orphan the moment it's created and
    survives delete_source. This is the ~202k-row MERGE-target population seen in production.

    The cleanup policy (scripts/cleanup_orphan_entities.py) KEEPS these by
    default because they back IMPORTS/CALLS traversal to external symbols; this
    test pins the current behavior so a future store_graph change is a
    conscious decision, not an accident."""
    e = _entity(1)
    await gs.store_graph(
        [e],
        [{"source_id": e["id"], "target_id": "ext://numpy", "type": "IMPORTS"}],
        source_id="src-1",
    )
    # the external target is an orphan from birth (no source_entities row)
    assert "ext://numpy" in await _orphan_entity_ids()
    assert (await gs.get_entity_by_id("ext://numpy"))["type"] == "ExternalModule"

    # deleting the source reaps the real entity but the external target is
    # NOT reaped via delete_source (it was never source-contained). The
    # relationship is dropped because its endpoint `e` was deleted; the bare
    # ExternalModule node persists as a dangling orphan -> cleanup-script turf.
    await gs.delete_source("src-1")
    assert await gs.get_entity_by_id(e["id"]) is None
    assert await gs.get_entity_by_id("ext://numpy") is not None  # survives
    assert "ext://numpy" in await _orphan_entity_ids()
