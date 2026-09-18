"""Live-Milvus regression for the filter-injection fix (CWE-943).

Integration, not unit: it needs a real Milvus, because the defect was in how
the SERVER parses the expression we build. The unit tests in
tests/unit/test_retriever_domain.py check the expression's shape; only this
file proves what Milvus does with it — and the distinction mattered. The
obvious fix (template every value) is wrong: Milvus 2.5.4 rejects a templated
`like` operand, so it would have broken every path-prefixed search while
looking correct in unit tests.

Run against a standalone instance:

    docker compose --profile local-infra up -d milvus
    MILVUS_TEST_URI=http://localhost:19530 \
      env -u PYTHONPATH python -m pytest tests/integration/test_milvus_filter_injection.py -v

Skipped unless MILVUS_TEST_URI is set — no service, no silent pass.
"""

from __future__ import annotations

import asyncio
import os

import pytest

URI = os.environ.get("MILVUS_TEST_URI")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not URI, reason="set MILVUS_TEST_URI to run"),
]

COLL = "treeloom_test_filter_injection"
VEC = [1.0, 0.0, 0.0, 0.0]

# A caller-supplied path_prefix that closes the literal and opens an `or`.
# Unescaped, this rendered as
#   file_path like "%" or source_id == "secret-repo%"
#     and source_id not in ["secret-repo"]
# and `and` binds tighter than `or`, so everything matched and the ACL
# prefilter never applied.
INJECTION = '%" or source_id == "secret-repo'


@pytest.fixture(scope="module")
def adapter():
    os.environ.setdefault("VECTOR_DIM", "4")
    from pymilvus import MilvusClient

    from treeloom.adapters.milvus import vector_store as vs

    raw = MilvusClient(uri=URI)
    if raw.has_collection(COLL):
        raw.drop_collection(COLL)

    host, _, port = URI.split("://", 1)[1].partition(":")
    a = vs.MilvusAdapter(
        host=host, port=port or "19530", collection_name=COLL, vector_dim=4
    )
    a.uri = URI  # the adapter assumes https; the local standalone is plain http
    asyncio.run(a.init_collection())
    asyncio.run(
        a.insert(
            [
                {"text": "ok", "file_path": "/srv/mine/a.py", "language": "python",
                 "start_line": 1, "end_line": 9, "source_id": "mine"},
                {"text": "CLASSIFIED", "file_path": "/srv/secret/b.py", "language": "python",
                 "start_line": 1, "end_line": 9, "source_id": "secret-repo"},
                {"text": "bs", "file_path": "/srv/back\\slash/c.py", "language": "python",
                 "start_line": 1, "end_line": 9, "source_id": "back\\slash"},
            ],
            [VEC] * 3,
        )
    )
    raw.flush(COLL)
    raw.load_collection(COLL)
    yield a
    raw.drop_collection(COLL)


def _sources(rows) -> list[str]:
    return sorted(r["entity"]["source_id"] for r in rows)


class TestAclPrefilterHoldsUnderInjection:
    """The decisive pair. Before the fix both returned the denied source."""

    def test_search(self, adapter):
        rows = adapter.search(
            VEC, top_k=10, path_prefix=INJECTION, exclude_source_ids=["secret-repo"]
        )
        assert _sources(rows) == []

    def test_hybrid_search(self, adapter):
        rows = adapter.hybrid_search(
            "q", VEC, top_k=10, path_prefix=INJECTION,
            exclude_source_ids=["secret-repo"],
        )
        assert _sources(rows) == []


class TestOrdinarySearchStillWorks:
    """A filter that refuses everything would pass the class above. These are
    what stop the fix from being bought with a broken feature."""

    ALL = ["back\\slash", "mine", "secret-repo"]

    def test_no_filter(self, adapter):
        assert _sources(adapter.search(VEC, top_k=10)) == self.ALL

    def test_path_prefix(self, adapter):
        assert _sources(adapter.search(VEC, top_k=10, path_prefix="/srv/mine")) == ["mine"]

    def test_language(self, adapter):
        assert _sources(adapter.search(VEC, top_k=10, language="python")) == self.ALL

    def test_source_id(self, adapter):
        assert _sources(adapter.search(VEC, top_k=10, source_id="mine")) == ["mine"]

    def test_exclusion_is_applied(self, adapter):
        rows = adapter.search(VEC, top_k=10, exclude_source_ids=["secret-repo"])
        assert _sources(rows) == ["back\\slash", "mine"]

    def test_hybrid_no_filter(self, adapter):
        assert _sources(adapter.hybrid_search("ok", VEC, top_k=10)) == self.ALL

    def test_hybrid_path_prefix(self, adapter):
        rows = adapter.hybrid_search("ok", VEC, top_k=10, path_prefix="/srv/mine")
        assert _sources(rows) == ["mine"]

    def test_hybrid_exclusion_is_applied(self, adapter):
        rows = adapter.hybrid_search("ok", VEC, top_k=10, exclude_source_ids=["secret-repo"])
        assert _sources(rows) == ["back\\slash", "mine"]


class TestBackslashValues:
    """The quote-only escaping these paths used raised Milvus error 1100
    ('cannot parse expression') on any value containing a backslash — so a
    Windows path, or any file with one in its name, failed rather than
    matched. Same root defect as the injection, milder symptom."""

    def test_search_by_source_id(self, adapter):
        rows = adapter.search(VEC, top_k=10, source_id="back\\slash")
        assert _sources(rows) == ["back\\slash"]

    def test_list_indexed_paths(self, adapter):
        assert sorted(adapter.list_indexed_paths("back\\slash")) == [
            "/srv/back\\slash/c.py"
        ]

    def test_get_chunk_bodies(self, adapter):
        bodies = adapter.get_chunk_bodies(
            [("/srv/back\\slash/c.py", 1, 9)], source_id="back\\slash"
        )
        assert [b["source_id"] for b in bodies] == ["back\\slash"]


class TestHydrateRespectsExclusions:
    """/hydrate-chunks called _authorize_scope and threw the result away, so
    the denied-source list never reached the query. A hit_id is only a file
    path plus a line range — guessable for any repo whose layout the caller
    knows — so the bodies came back regardless of grants."""

    def test_denied_source_body_is_not_returned(self, adapter):
        loc = [("/srv/secret/b.py", 1, 9)]
        assert adapter.get_chunk_bodies(loc, None) != [], "control: it is there"
        assert adapter.get_chunk_bodies(
            loc, None, exclude_source_ids=["secret-repo"]
        ) == []

    def test_permitted_source_still_hydrates(self, adapter):
        bodies = adapter.get_chunk_bodies(
            [("/srv/mine/a.py", 1, 9)], None, exclude_source_ids=["secret-repo"]
        )
        assert [b["source_id"] for b in bodies] == ["mine"]


class TestDeletesTargetExactlyOneRow:
    """Deletion runs the same expression builder; a filter that over-matched
    here would destroy another source's chunks."""

    def test_delete_by_file_then_by_source(self, adapter):
        from pymilvus import MilvusClient

        raw = MilvusClient(uri=URI)
        assert adapter.delete_chunks_by_file(
            "back\\slash", "/srv/back\\slash/c.py"
        ) == 1
        raw.flush(COLL)
        adapter.delete_chunks_by_source("secret-repo")
        raw.flush(COLL)
        assert _sources(adapter.search(VEC, top_k=10)) == ["mine"]
