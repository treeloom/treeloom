"""Unit tests for branch pattern filtering."""

import pytest
from treeloom.domain.shared import branch_matches, SourceInfo


class TestBranchMatches:
    def test_exact_match(self):
        assert branch_matches("main", ["main", "develop"]) is True

    def test_no_match(self):
        assert branch_matches("feature/foo", ["main", "develop"]) is False

    def test_glob_match(self):
        assert branch_matches("release/v2.1", ["main", "release/*"]) is True

    def test_glob_no_match(self):
        assert branch_matches("hotfix/critical", ["main", "release/*"]) is False

    def test_empty_patterns_matches_everything(self):
        """Empty or None patterns = no filtering (opt-in to restrict)."""
        assert branch_matches("feature/foo", []) is True
        assert branch_matches("feature/foo", None) is True

    def test_default_patterns_main_master(self):
        """Default: only main and master."""
        patterns = ["main", "master"]
        assert branch_matches("main", patterns) is True
        assert branch_matches("master", patterns) is True
        assert branch_matches("develop", patterns) is False
        assert branch_matches("feature/xyz", patterns) is False

    def test_develop_pattern(self):
        patterns = ["main", "develop", "release/*"]
        assert branch_matches("develop", patterns) is True
        assert branch_matches("release/2026-q2", patterns) is True
        assert branch_matches("release-next", patterns) is False  # not glob

    def test_single_star_matches_anything(self):
        assert branch_matches("anything", ["*"]) is True

    def test_case_sensitive(self):
        """Branch matching is case-sensitive."""
        assert branch_matches("Main", ["main"]) is False

    def test_none_branch_defaults(self):
        """None branch = assume default."""
        assert branch_matches(None, ["main", "master"]) is True


class TestSourceInfoBranchPatterns:
    def test_default_patterns(self):
        s = SourceInfo(id="abc", url="https://example.com/repo")
        assert s.branch_patterns == ["main", "master"]

    def test_custom_patterns(self):
        s = SourceInfo(
            id="abc",
            url="https://example.com/repo",
            branch_patterns=["main", "develop", "release/*"],
        )
        assert s.branch_patterns == ["main", "develop", "release/*"]

    def test_serialization_roundtrip(self):
        s = SourceInfo(
            id="abc",
            url="https://example.com/repo",
            branch="main",
            branch_patterns=["main", "release/*"],
            file_count=10,
            chunk_count=50,
        )
        data = s.model_dump()
        restored = SourceInfo(**data)
        assert restored.branch_patterns == ["main", "release/*"]
