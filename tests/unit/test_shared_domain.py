"""Domain tests for shared models — chunk_hit_from_milvus conversion."""
import pytest

from treeloom.domain.shared import ChunkHit, chunk_hit_from_milvus


def test_chunk_hit_from_milvus_flattens_entity():
    """Raw Milvus hit (nested `entity`, `chunk_text`) -> flat ChunkHit shape.

    Regression: graph_search returns raw Milvus dicts; if they aren't run
    through this converter the MCP server's ChunkHit(**c) blows up because
    `file_path`/`snippet` live under `entity` and the field is `chunk_text`,
    not `snippet`.
    """
    raw = {
        "id": 4666228193,
        "distance": 0.42,
        "relevance_score": 0.91,
        "entity": {
            "file_path": "src/model.py",
            "chunk_text": "def forward(self): ...",
            "language": "python",
            "start_line": 10,
            "end_line": 25,
            "source_id": "src-1",
        },
    }
    hit = chunk_hit_from_milvus(raw)
    assert isinstance(hit, ChunkHit)
    assert hit.file_path == "src/model.py"
    assert hit.snippet == "def forward(self): ..."
    assert hit.start_line == 10
    assert hit.end_line == 25
    assert hit.language == "python"
    assert hit.source_id == "src-1"
    # relevance_score preferred over raw distance
    assert hit.score == pytest.approx(0.91)


def test_chunk_hit_from_milvus_prefers_final_score():
    """When graph rescoring ran, surface final_score (the ranking value)."""
    raw = {
        "relevance_score": 0.5,
        "distance": 0.1,
        "final_score": 0.73,
        "entity": {"file_path": "a.py", "chunk_text": "x = 1", "start_line": 1, "end_line": 1},
    }
    assert chunk_hit_from_milvus(raw).score == pytest.approx(0.73)


def test_chunk_hit_from_milvus_handles_missing_entity():
    """A hit with no entity payload still produces a valid (empty) ChunkHit
    instead of raising — file_path/snippet default to ''."""
    hit = chunk_hit_from_milvus({"id": 1, "distance": 0.0})
    assert hit.file_path == ""
    assert hit.snippet == ""
    assert hit.score == 0.0
