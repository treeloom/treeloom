"""Manifest parsing for fleet onboarding. Pure, no I/O."""
import pytest

from treeloom.domain.sources.manifest import ManifestEntry, parse_manifest


def test_bare_list_yaml():
    entries = parse_manifest(
        """
        - path: /src/a
        - url: https://github.com/acme/b.git
          branch: release
        """
    )
    assert [e.label for e in entries] == ["/src/a", "https://github.com/acme/b.git"]
    assert entries[0].path == "/src/a" and entries[0].url == ""
    assert entries[1].url == "https://github.com/acme/b.git"
    assert entries[1].branch == "release"


def test_json_form_parses():
    # JSON is a subset of YAML, so the same parser handles it.
    entries = parse_manifest('{"repos": [{"path": "/src/a"}]}')
    assert len(entries) == 1
    assert entries[0].path == "/src/a"


def test_defaults_merged_and_overridden():
    entries = parse_manifest(
        """
        defaults:
          branch: main
          skip_graph: true
        repos:
          - path: /src/a
          - path: /src/b
            branch: dev
        """
    )
    assert entries[0].branch == "main" and entries[0].skip_graph is True
    # Per-entry value wins over the default.
    assert entries[1].branch == "dev" and entries[1].skip_graph is True


def test_empty_manifest_is_empty_list():
    assert parse_manifest("") == []
    assert parse_manifest("repos: []") == []


def test_path_xor_url_required():
    with pytest.raises(ValueError, match="exactly one"):
        parse_manifest("- path: /src/a\n  url: https://x/y.git")
    with pytest.raises(ValueError, match="exactly one"):
        parse_manifest("- branch: main")  # neither path nor url


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        parse_manifest("- path: /src/a\n  bogus: 1")


def test_skip_patterns_must_be_list_of_strings():
    with pytest.raises(ValueError, match="skip_patterns"):
        parse_manifest("- path: /src/a\n  skip_patterns: notalist")


def test_repos_key_required_in_mapping_form():
    with pytest.raises(ValueError, match="repos"):
        parse_manifest("defaults:\n  branch: main")


def test_to_index_request_shapes_body():
    entry = ManifestEntry(
        url="https://github.com/acme/b.git",
        branch="main",
        skip_patterns=["vendor/**"],
        skip_graph=True,
    )
    body = entry.to_index_request()
    assert body == {
        "force": False,
        "url": "https://github.com/acme/b.git",
        "branch": "main",
        "skip_patterns": ["vendor/**"],
        "skip_graph": True,
    }


def test_force_override_ors_with_entry_force():
    assert ManifestEntry(path="/a").to_index_request(force_override=True)["force"] is True
    assert ManifestEntry(path="/a", force=True).to_index_request()["force"] is True
    assert ManifestEntry(path="/a").to_index_request()["force"] is False
