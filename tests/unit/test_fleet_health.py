"""Fleet health rollup aggregation. Pure function, no services."""
from treeloom.application.fleet import build_fleet_health


def _src(id, **kw):
    base = {
        "id": id, "path": f"/src/{id}", "url": "", "branch": "main",
        "indexed_at": 1000, "file_count": 10, "chunk_count": 100,
        "commit_sha": "abc", "graph_indexed": True,
    }
    base.update(kw)
    return base


def _job(source_id, **kw):
    base = {
        "job_id": f"job-{source_id}", "source_id": source_id, "status": "done",
        "error": "", "errors": 0, "finished_at": 1001.0, "start_time": 1000.0,
    }
    base.update(kw)
    return base


def test_summary_totals_and_counts():
    sources = [
        _src("a", chunk_count=100, file_count=10),
        _src("b", chunk_count=50, file_count=5, graph_indexed=False),
    ]
    jobs = [_job("a"), _job("b", status="failed", error="boom")]
    health = build_fleet_health(sources, jobs, now=4600.0)
    s = health["summary"]
    assert s["sources_total"] == 2
    assert s["total_chunks"] == 150
    assert s["total_files"] == 15
    assert s["sources_graph_missing"] == 1   # b
    assert s["sources_with_errors"] == 1     # b failed
    # oldest age = now - min(indexed_at) = 4600 - 1000
    assert s["oldest_index_age_seconds"] == 3600.0
    # staleness/entities absent unless provided
    assert "sources_stale" not in s
    assert "total_entities" not in s


def test_latest_job_per_source_wins():
    sources = [_src("a")]
    jobs = [
        _job("a", job_id="old", status="failed", errors=3, start_time=100.0),
        _job("a", job_id="new", status="done", errors=0, start_time=200.0),
    ]
    health = build_fleet_health(sources, jobs)
    row = health["sources"][0]
    assert row["last_job"]["job_id"] == "new"
    assert row["last_job"]["status"] == "done"
    # The newest job is clean → not counted as error.
    assert health["summary"]["sources_with_errors"] == 0


def test_errors_counter_flags_error_count_even_when_done():
    # A 'done' job that logged per-file errors still counts as unhealthy.
    sources = [_src("a")]
    health = build_fleet_health(sources, [_job("a", status="done", errors=2)])
    assert health["summary"]["sources_with_errors"] == 1


def test_no_job_for_source_is_not_error():
    health = build_fleet_health([_src("a")], [])
    assert health["sources"][0]["last_job"] is None
    assert health["summary"]["sources_with_errors"] == 0


def test_entities_and_staleness_enrichments():
    sources = [_src("a"), _src("b")]
    health = build_fleet_health(
        sources, [],
        staleness_by_id={"a": True, "b": None},
        entity_counts={"a": 42, "b": 7},
        staleness_truncated=False,
    )
    rows = {r["id"]: r for r in health["sources"]}
    assert rows["a"]["entity_count"] == 42
    assert rows["a"]["is_stale"] is True
    assert rows["b"]["is_stale"] is None
    assert health["summary"]["total_entities"] == 49
    assert health["summary"]["sources_stale"] == 1  # only a, None doesn't count


def test_staleness_truncated_flag():
    health = build_fleet_health(
        [_src("a")], [], staleness_by_id={}, staleness_truncated=True
    )
    assert health["summary"]["staleness_truncated"] is True


def test_url_source_label_prefers_url():
    health = build_fleet_health(
        [_src("a", path="", url="https://github.com/x/y.git")], []
    )
    assert health["sources"][0]["label"] == "https://github.com/x/y.git"
