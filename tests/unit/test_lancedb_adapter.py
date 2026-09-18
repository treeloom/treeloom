"""LanceDB adapter — real embedded DB in a tmp dir (in-process = no mock).

Covers the milvus-module-surface contract: milvus-shaped hits, hybrid RRF
fusion with a BM25 leg, LIKE path_prefix prefilter, nullable summary
vectors, per-source/per-file deletes, resume support, dim guard.
"""
import asyncio

import pytest

from treeloom.adapters.lancedb import vector_store as vs

DIM = 8


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setattr(vs, "TABLE_NAME", "test_chunks")
    monkeypatch.setattr(vs, "VECTOR_DIM", DIM)
    monkeypatch.setattr(vs, "_db", None)
    monkeypatch.setattr(vs, "_table", None)
    yield


def _vec(seed: int) -> list[float]:
    return [((seed * 17 + i * 13) % 100) / 100.0 for i in range(DIM)]


def _chunk(i: int, text: str | None = None, source_id: str = "src-1") -> dict:
    return {
        "text": text or f"def func_{i}():\n    return {i}",
        "file_path": f"/repo/src/mod_{i % 3}/file_{i}.py",
        "language": "python",
        "start_line": 1 + i,
        "end_line": 10 + i,
        "source_id": source_id,
    }


async def _seed(n: int = 10, **kw):
    await vs.init_collection()
    chunks = [_chunk(i, **kw) for i in range(n)]
    await vs.insert_chunks(chunks, [_vec(i) for i in range(n)])
    return chunks


@pytest.mark.asyncio
async def test_init_collection_idempotent():
    await vs.init_collection()
    await vs.init_collection()
    assert vs._get_table().count_rows() == 0


@pytest.mark.asyncio
async def test_search_returns_milvus_shaped_hits():
    await _seed(5)
    hits = vs.search(_vec(2), top_k=3)
    assert len(hits) == 3
    top = hits[0]
    assert set(top) == {"id", "distance", "entity"}
    assert set(top["entity"]) == {
        "chunk_text", "file_path", "language", "start_line",
        "end_line", "source_id",
    }
    # self-match: cosine similarity ~1.0, and it ranks first
    assert top["entity"]["file_path"] == "/repo/src/mod_2/file_2.py"
    assert top["distance"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.asyncio
async def test_hybrid_bm25_leg_surfaces_keyword_needle():
    await vs.init_collection()
    chunks = [_chunk(i) for i in range(20)]
    chunks[13]["text"] = "def frobnicate_zanzibar():\n    pass"
    await vs.insert_chunks(chunks, [_vec(i) for i in range(20)])
    # dense query points at chunk 2; text query names the needle in 13
    hits = vs.hybrid_search("frobnicate_zanzibar", _vec(2), top_k=5)
    paths = [h["entity"]["file_path"] for h in hits]
    assert "/repo/src/mod_1/file_13.py" in paths  # BM25 leg
    assert "/repo/src/mod_2/file_2.py" in paths   # dense leg
    assert all(hits[i]["distance"] >= hits[i + 1]["distance"]
               for i in range(len(hits) - 1))


@pytest.mark.asyncio
async def test_path_prefix_is_native_prefilter():
    await _seed(9)
    hits = vs.search(_vec(0), top_k=9, path_prefix="/repo/src/mod_1/")
    assert hits and all(
        h["entity"]["file_path"].startswith("/repo/src/mod_1/") for h in hits
    )


@pytest.mark.asyncio
async def test_source_and_language_filters():
    await _seed(4, source_id="src-a")
    await vs.insert_chunks(
        [_chunk(99, source_id="src-b")], [_vec(99)]
    )
    hits = vs.search(_vec(99), top_k=10, source_id="src-b")
    assert [h["entity"]["source_id"] for h in hits] == ["src-b"]
    assert vs.search(_vec(0), top_k=10, language="rust") == []


@pytest.mark.asyncio
async def test_summary_vector_leg_with_nullable_column():
    await vs.init_collection()
    chunks = [_chunk(i) for i in range(6)]
    summaries: list[list[float] | None] = [
        _vec(100 + i) if i % 2 == 0 else None for i in range(6)
    ]
    await vs.insert_chunks(chunks, [_vec(i) for i in range(6)], summaries)
    hits = vs.hybrid_search(
        "anything", _vec(0), query_summary_dense=_vec(104), top_k=6
    )
    assert hits  # the null summary rows must not break the summary leg
    paths = [h["entity"]["file_path"] for h in hits]
    assert "/repo/src/mod_1/file_4.py" in paths  # summary self-match for i=4


@pytest.mark.asyncio
async def test_delete_chunks_by_file_returns_count():
    await _seed(5)
    assert vs.delete_chunks_by_file("src-1", "/repo/src/mod_1/file_1.py") == 1
    assert vs.delete_chunks_by_file("src-1", "/repo/src/mod_1/file_1.py") == 0
    assert vs._get_table().count_rows() == 4


@pytest.mark.asyncio
async def test_delete_chunks_by_source():
    await _seed(4, source_id="src-a")
    await vs.insert_chunks([_chunk(50, source_id="src-b")], [_vec(50)])
    vs.delete_chunks_by_source("src-a")
    assert vs._get_table().count_rows() == 1


@pytest.mark.asyncio
async def test_list_indexed_paths_for_resume():
    await _seed(5)
    paths = await vs.list_indexed_paths("src-1")
    assert paths == {f"/repo/src/mod_{i % 3}/file_{i}.py" for i in range(5)}
    assert await vs.list_indexed_paths("nope") == set()


@pytest.mark.asyncio
async def test_dim_mismatch_fails_loud(monkeypatch):
    await _seed(2)
    monkeypatch.setattr(vs, "VECTOR_DIM", DIM * 2)
    monkeypatch.setattr(vs, "_table", None)
    with pytest.raises(RuntimeError, match="vector dim"):
        await vs.init_collection()


@pytest.mark.asyncio
async def test_sql_injection_escaped_in_filters():
    await _seed(2)
    # must not raise / must not match anything
    assert vs.search(_vec(0), top_k=5, source_id="x' OR '1'='1") == []
    assert vs.delete_chunks_by_file("src-1", "p' OR '1'='1") == 0
    assert vs._get_table().count_rows() == 2
