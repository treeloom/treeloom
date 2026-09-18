"""Fleet auto-refresh selection logic. Pure function, no services."""
from treeloom.application.fleet import resolve_reindex_kind, select_sources_to_refresh


def _s(id, is_stale, indexed_at):
    return {"id": id, "is_stale": is_stale, "indexed_at": indexed_at}


def test_only_stale_git_sources_selected():
    sources = [
        _s("stale", True, 100),
        _s("fresh", False, 100),
        _s("nongit", None, 100),  # staleness undetermined → skipped
    ]
    assert select_sources_to_refresh(sources, max_per_tick=10) == ["stale"]


def test_oldest_indexed_first():
    sources = [
        _s("new", True, 300),
        _s("old", True, 100),
        _s("mid", True, 200),
    ]
    assert select_sources_to_refresh(sources, max_per_tick=10) == ["old", "mid", "new"]


def test_cap_limits_per_tick():
    sources = [_s(f"s{i}", True, i) for i in range(10)]
    picked = select_sources_to_refresh(sources, max_per_tick=3)
    assert picked == ["s0", "s1", "s2"]


def test_zero_cap_selects_nothing():
    assert select_sources_to_refresh([_s("a", True, 1)], max_per_tick=0) == []


def test_missing_id_skipped():
    sources = [{"is_stale": True, "indexed_at": 1}, _s("ok", True, 2)]
    assert select_sources_to_refresh(sources, max_per_tick=10) == ["ok"]


def test_none_indexed_at_sorts_last():
    sources = [_s("noidx", True, None), _s("withidx", True, 50)]
    assert select_sources_to_refresh(sources, max_per_tick=10) == ["withidx", "noidx"]


# ── resolve_reindex_kind: re-enqueue the SAME kind, not always repo ──────────

def test_stored_kind_is_trusted():
    for k in ("repo", "directory", "file"):
        assert resolve_reindex_kind(stored_kind=k, has_url=False, path_is_file=True) == k


def test_legacy_url_source_is_repo():
    # No stored kind (legacy row) + a URL → repo job (clone).
    assert resolve_reindex_kind(stored_kind=None, has_url=True, path_is_file=False) == "repo"


def test_legacy_file_path_is_file_not_repo():
    # The bug guard: a legacy file-indexed source must NOT become a repo job
    # (which would fail the runner's isdir check every tick).
    assert resolve_reindex_kind(stored_kind="", has_url=False, path_is_file=True) == "file"


def test_legacy_directory_path_defaults_repo():
    assert resolve_reindex_kind(stored_kind=None, has_url=False, path_is_file=False) == "repo"


def test_unknown_stored_kind_falls_back_to_shape():
    # A garbage/unrecognised kind value is treated as legacy → shape inference.
    assert resolve_reindex_kind(stored_kind="graph", has_url=False, path_is_file=True) == "file"
