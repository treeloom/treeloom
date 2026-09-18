"""Unit tests for domain/provenance.py — pure, no mocks, no services."""

from datetime import datetime, timezone

import pytest

from treeloom.domain.provenance import (
    build_provenance,
    make_citation,
    permalink,
    repo_label,
)
from treeloom.domain.sources import SourceRecord


# ── make_citation ───────────────────────────────────────────────────────


class TestMakeCitation:
    def test_full_with_sha_and_lines(self):
        result = make_citation("mui_material-ui", "a1b2c3d4e5f6aaaa", "src/Foo.tsx", 10, 42)
        assert result == "mui_material-ui@a1b2c3d4e5f6:src/Foo.tsx:10-42"

    def test_sha_truncated_to_12(self):
        result = make_citation("repo", "abcdef123456789xyz", "f.py", 1, 2)
        # exactly 12 chars used — the 13th char ("7") must not appear after "@"
        assert "@abcdef1234567" not in result
        assert "@abcdef123456:" in result

    def test_without_sha(self):
        result = make_citation("myrepo", "", "src/bar.py", 5, 10)
        assert "@" not in result
        assert result == "myrepo:src/bar.py:5-10"

    def test_lines_both_zero_omits_line_segment(self):
        result = make_citation("repo", "abc123def456", "src/baz.py", 0, 0)
        assert "-" not in result
        assert result == "repo@abc123def456:src/baz.py"

    def test_no_sha_no_lines(self):
        result = make_citation("repo", "", "src/baz.py", 0, 0)
        assert result == "repo:src/baz.py"

    def test_empty_label_falls_back_to_file_only(self):
        result = make_citation("", "", "src/thing.py", 0, 0)
        assert result == "src/thing.py"

    def test_empty_label_with_sha(self):
        result = make_citation("", "aabbccdd1122", "x.py", 3, 7)
        # label is empty so the sha segment starts with "@"
        assert result.startswith("@aabbccdd1122")
        assert "x.py" in result
        assert "3-7" in result

    def test_sha_exact_12_chars_not_truncated_further(self):
        result = make_citation("r", "123456789012", "f.py", 1, 2)
        assert "@123456789012:" in result


# ── permalink ───────────────────────────────────────────────────────────


class TestPermalink:
    def test_none_when_url_missing(self):
        assert permalink("", "abc123", "src/f.py", 1, 5) is None

    def test_none_when_sha_missing(self):
        assert permalink("https://github.com/org/repo", "", "src/f.py", 1, 5) is None

    def test_git_suffix_stripped(self):
        result = permalink("https://github.com/org/repo.git", "abc123def456", "src/f.py", 1, 5)
        assert result is not None
        assert ".git" not in result

    def test_correct_hash_L_shape(self):
        result = permalink("https://github.com/org/repo", "abc123def456", "src/f.py", 10, 42)
        assert result == "https://github.com/org/repo/blob/abc123def456/src/f.py#L10-L42"

    def test_no_L_when_lines_both_zero(self):
        result = permalink("https://github.com/org/repo", "abc123def456", "src/f.py", 0, 0)
        assert result is not None
        assert "#L" not in result
        assert result == "https://github.com/org/repo/blob/abc123def456/src/f.py"

    def test_trailing_slash_stripped_from_url(self):
        result = permalink("https://github.com/org/repo/", "abc123", "a.py", 0, 0)
        assert result is not None
        assert "repo//blob" not in result
        assert "/blob/abc123/" in result

    def test_gitea_url_works(self):
        result = permalink("https://gitea.example.com/user/project", "deadbeef1234", "main.go", 1, 10)
        assert result == "https://gitea.example.com/user/project/blob/deadbeef1234/main.go#L1-L10"


# ── repo_label ──────────────────────────────────────────────────────────


class TestRepoLabel:
    def _make_record(self, **kwargs) -> SourceRecord:
        return SourceRecord(id="testid", **kwargs)

    def test_url_beats_path(self):
        rec = self._make_record(url="https://github.com/org/awesome-lib.git", path="/home/user/awesome-lib")
        assert repo_label(rec) == "awesome-lib"

    def test_git_stripped_from_url_basename(self):
        rec = self._make_record(url="https://github.com/org/my-repo.git")
        assert repo_label(rec) == "my-repo"

    def test_url_without_git_suffix(self):
        rec = self._make_record(url="https://github.com/org/my-repo")
        assert repo_label(rec) == "my-repo"

    def test_path_beats_id_when_no_url(self):
        rec = self._make_record(path="/home/user/projects/cool-tool")
        assert repo_label(rec) == "cool-tool"

    def test_id_fallback_when_no_url_no_path(self):
        rec = self._make_record()
        assert repo_label(rec) == "testid"

    def test_works_with_plain_dict_url(self):
        d = {"url": "https://github.com/org/dict-repo.git", "path": "/p", "id": "xid"}
        assert repo_label(d) == "dict-repo"

    def test_works_with_plain_dict_path(self):
        d = {"path": "/home/user/dict-path", "id": "xid"}
        assert repo_label(d) == "dict-path"

    def test_works_with_plain_dict_id(self):
        d = {"id": "abc123"}
        assert repo_label(d) == "abc123"

    def test_works_with_plain_dict_source_id(self):
        d = {"source_id": "src456"}
        assert repo_label(d) == "src456"


# ── build_provenance ────────────────────────────────────────────────────


_DT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _record(sid: str, *, url: str = "", path: str = "", branch: str = "", commit_sha: str = "") -> SourceRecord:
    return SourceRecord(
        id=sid,
        url=url,
        path=path,
        branch=branch,
        commit_sha=commit_sha,
        indexed_at=_DT,
    )


class TestBuildProvenanceScoped:
    """Scoped (single-source) result — source_id hoisted to result-level."""

    def setup_method(self):
        self.sid = "aabbccdd11223344"
        self.rec = _record(
            self.sid,
            url="https://github.com/org/myrepo.git",
            commit_sha="deadbeef1234abcd",
            branch="main",
        )
        self.result = {
            "source_id": self.sid,
            "chunks": [
                {"file_path": "a.py", "start_line": 1, "end_line": 5, "source_id": None},
                {"file_path": "b.py", "start_line": 10, "end_line": 20, "source_id": None},
            ],
        }
        build_provenance(self.result, {self.sid: self.rec})

    def test_chunks_do_not_repeat_commit_sha(self):
        # commit_sha is no longer written per chunk — it's emitted once in the
        # sources map to avoid repeating the same SHA ~6× per single-source
        # search (token optimization). The citation still carries it.
        for chunk in self.result["chunks"]:
            assert "commit_sha" not in chunk

    def test_chunks_get_citation(self):
        c0 = self.result["chunks"][0]
        assert c0["citation"] == "myrepo@deadbeef1234:a.py:1-5"

        c1 = self.result["chunks"][1]
        assert c1["citation"] == "myrepo@deadbeef1234:b.py:10-20"

    def test_sources_entry_exists(self):
        assert self.sid in self.result["sources"]

    def test_sources_entry_has_commit_sha(self):
        assert self.result["sources"][self.sid]["commit_sha"] == "deadbeef1234abcd"

    def test_sources_entry_has_indexed_at_iso(self):
        assert self.result["sources"][self.sid]["indexed_at"] == _DT.isoformat()

    def test_sources_entry_has_path_url_branch(self):
        entry = self.result["sources"][self.sid]
        assert entry["url"] == "https://github.com/org/myrepo.git"
        assert entry["path"] == ""
        assert entry["branch"] == "main"

    def test_sources_entry_has_permalink_base(self):
        entry = self.result["sources"][self.sid]
        assert "permalink_base" in entry
        assert entry["permalink_base"] == "https://github.com/org/myrepo/blob/deadbeef1234abcd"

    def test_no_staleness_keys_when_staleness_not_provided(self):
        entry = self.result["sources"][self.sid]
        assert "is_stale" not in entry
        assert "current_sha" not in entry


class TestBuildProvenanceCrossRepo:
    """Cross-repo result — chunks carry their own source_id."""

    def setup_method(self):
        self.sid1 = "source111"
        self.sid2 = "source222"
        self.rec1 = _record(
            self.sid1,
            url="https://github.com/org/repo-one.git",
            commit_sha="aabbccdd11223344",
        )
        self.rec2 = _record(
            self.sid2,
            path="/home/user/local-repo",
            commit_sha="",  # non-git source
        )
        self.result = {
            "chunks": [
                {"file_path": "x.py", "start_line": 1, "end_line": 3, "source_id": self.sid1},
                {"file_path": "y.py", "start_line": 5, "end_line": 8, "source_id": self.sid2},
            ],
        }
        build_provenance(self.result, {self.sid1: self.rec1, self.sid2: self.rec2})

    def test_two_sources_entries(self):
        assert self.sid1 in self.result["sources"]
        assert self.sid2 in self.result["sources"]

    def test_each_chunk_cited_to_its_own_source(self):
        c0 = self.result["chunks"][0]
        assert "repo-one@aabbccdd1122:x.py:1-3" == c0["citation"]
        # commit_sha is no longer written per chunk — it lives in the
        # sources map, and the citation embeds the short SHA.
        assert "commit_sha" not in c0
        assert self.result["sources"][self.sid1]["commit_sha"] == "aabbccdd11223344"

    def test_no_commit_sha_key_on_chunks(self):
        # commit_sha is never written onto chunks anymore.
        for chunk in self.result["chunks"]:
            assert "commit_sha" not in chunk

    def test_no_permalink_base_when_no_url(self):
        entry2 = self.result["sources"][self.sid2]
        assert "permalink_base" not in entry2

    def test_permalink_base_present_for_url_source(self):
        entry1 = self.result["sources"][self.sid1]
        assert "permalink_base" in entry1
        assert entry1["permalink_base"] == "https://github.com/org/repo-one/blob/aabbccdd11223344"


class TestBuildProvenanceStaleness:
    """staleness_by_id merges is_stale and current_sha."""

    def setup_method(self):
        self.sid = "staleid"
        self.rec = _record(self.sid, url="https://github.com/org/repo.git", commit_sha="oldsha1234567890")
        self.result = {
            "source_id": self.sid,
            "chunks": [{"file_path": "f.py", "start_line": 1, "end_line": 2, "source_id": None}],
        }

    def test_staleness_merged_when_provided(self):
        staleness = {self.sid: {"is_stale": True, "current_sha": "newsha9876543210"}}
        build_provenance(self.result, {self.sid: self.rec}, staleness_by_id=staleness)
        entry = self.result["sources"][self.sid]
        assert entry["is_stale"] is True
        assert entry["current_sha"] == "newsha9876543210"

    def test_no_staleness_when_not_in_staleness_dict(self):
        staleness = {"other_sid": {"is_stale": True, "current_sha": "xyz"}}
        build_provenance(self.result, {self.sid: self.rec}, staleness_by_id=staleness)
        entry = self.result["sources"][self.sid]
        assert "is_stale" not in entry

    def test_staleness_none_is_safe(self):
        build_provenance(self.result, {self.sid: self.rec}, staleness_by_id=None)
        entry = self.result["sources"][self.sid]
        assert "is_stale" not in entry


class TestBuildProvenanceDefensive:
    """build_provenance must never KeyError on incomplete input."""

    def test_missing_file_path_in_chunk_is_safe(self):
        sid = "s1"
        rec = _record(sid, commit_sha="abc123", url="https://github.com/x/y")
        result = {"source_id": sid, "chunks": [{"start_line": 1, "end_line": 2, "source_id": None}]}
        build_provenance(result, {sid: rec})  # must not raise
        # citation still produced, just with empty file_path
        assert "citation" in result["chunks"][0]

    def test_unknown_source_id_skipped(self):
        result = {
            "chunks": [{"file_path": "f.py", "start_line": 1, "end_line": 2, "source_id": "missing_sid"}]
        }
        build_provenance(result, {})  # must not raise
        assert result["chunks"][0].get("citation") is None

    def test_empty_chunks_list(self):
        sid = "s1"
        rec = _record(sid, commit_sha="abc123")
        result = {"source_id": sid, "chunks": []}
        build_provenance(result, {sid: rec})
        # sources still created because top-level source_id is resolvable
        assert sid in result["sources"]
