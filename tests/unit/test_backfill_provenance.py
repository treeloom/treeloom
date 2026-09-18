"""Unit tests for treeloom.backfill_provenance.

Detroit-style: the git-resolution tests run against a real temporary git
repo built with subprocess `git`; the plan tests use a stub resolver so the
decision logic is exercised without any git or DB.
"""

from __future__ import annotations

import subprocess

import pytest

from treeloom.backfill_provenance import (
    BackfillOutcome,
    ResolvedProvenance,
    git_branch,
    git_origin_url,
    normalize_git_url,
    plan_source,
    resolve_provenance,
)
from treeloom.domain.sources import SourceRecord


# ── fixtures ──────────────────────────────────────────────────────────────


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(cwd),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )


@pytest.fixture
def git_repo(tmp_path):
    """A temp git repo on branch `mainline` with an origin remote + 1 commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "mainline")
    (repo / "a.py").write_text("print('hi')\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "remote", "add", "origin", "https://github.com/acme/widget.git")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, sha


# ── normalize_git_url ─────────────────────────────────────────────────────


class TestNormalizeGitUrl:
    def test_https_strips_dotgit(self):
        assert normalize_git_url("https://github.com/acme/widget.git") == (
            "https://github.com/acme/widget"
        )

    def test_scp_ssh_to_https(self):
        assert normalize_git_url("git@github.com:acme/widget.git") == (
            "https://github.com/acme/widget"
        )

    def test_ssh_scheme_to_https(self):
        assert normalize_git_url("ssh://git@gitlab.com/acme/widget.git") == (
            "https://gitlab.com/acme/widget"
        )

    def test_trailing_slash_trimmed(self):
        assert normalize_git_url("https://host/a/b/") == "https://host/a/b"

    def test_empty(self):
        assert normalize_git_url("") == ""
        assert normalize_git_url(None) == ""


# ── git resolution against a real repo ────────────────────────────────────


class TestResolveProvenanceRealRepo:
    def test_resolves_sha_url_branch(self, git_repo):
        repo, sha = git_repo
        resolved = resolve_provenance(str(repo))
        assert resolved.is_git
        assert resolved.commit_sha == sha
        assert resolved.url == "https://github.com/acme/widget"
        assert resolved.branch == "mainline"

    def test_origin_url_helper(self, git_repo):
        repo, _ = git_repo
        assert git_origin_url(str(repo)) == "https://github.com/acme/widget.git"

    def test_branch_helper(self, git_repo):
        repo, _ = git_repo
        assert git_branch(str(repo)) == "mainline"

    def test_non_git_path_is_not_git(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "f.txt").write_text("x")
        resolved = resolve_provenance(str(plain))
        assert resolved.is_git is False
        assert resolved.commit_sha == ""
        assert resolved.url == ""
        assert resolved.branch == ""

    def test_repo_without_origin_resolves_sha_but_empty_url(self, tmp_path):
        repo = tmp_path / "noremote"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "trunk")
        (repo / "a.py").write_text("x\n")
        _git(repo, "add", "a.py")
        _git(repo, "commit", "-q", "-m", "init")
        resolved = resolve_provenance(str(repo))
        assert resolved.is_git
        assert resolved.commit_sha
        assert resolved.url == ""
        assert resolved.branch == "trunk"


# ── plan_source decision logic (stub resolver, no git/DB) ─────────────────


def _rec(**kw):
    base = dict(id="abc", path="/data/repos/widget", commit_sha="")
    base.update(kw)
    return SourceRecord(**base)


class TestPlanSource:
    def test_updates_empty_sha_git_path(self):
        resolved = ResolvedProvenance("deadbeef", "https://h/o/r", "main")
        out = plan_source(_rec(), resolve=lambda p: resolved)
        assert out.action == "update"
        assert out.commit_sha == "deadbeef"
        assert out.url == "https://h/o/r"
        assert out.branch == "main"

    def test_skips_non_git(self):
        out = plan_source(_rec(), resolve=lambda p: ResolvedProvenance("", "", ""))
        assert out.action == "skip-non-git"

    def test_skips_no_path(self):
        out = plan_source(
            _rec(path="", url="https://h/o/r.git"),
            resolve=lambda p: pytest.fail("resolve must not run for empty path"),
        )
        assert out.action == "skip-no-path"

    def test_skips_row_with_existing_sha(self):
        out = plan_source(
            _rec(commit_sha="already"),
            resolve=lambda p: pytest.fail("resolve must not run when sha present"),
        )
        assert out.action == "skip-has-sha"

    def test_force_reresolves_existing_sha(self):
        resolved = ResolvedProvenance("new", "https://h/o/r", "main")
        out = plan_source(_rec(commit_sha="old"), resolve=lambda p: resolved, force=True)
        assert out.action == "update"
        assert out.commit_sha == "new"

    def test_end_to_end_real_repo_through_plan(self, git_repo):
        repo, sha = git_repo
        out = plan_source(_rec(path=str(repo)))  # default real resolver
        assert out.action == "update"
        assert out.commit_sha == sha
        assert out.url == "https://github.com/acme/widget"
        assert out.branch == "mainline"
