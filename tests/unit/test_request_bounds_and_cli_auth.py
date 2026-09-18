"""Unbounded request inputs (CWE-400), the .env parser (CWE-20), and the
bearer token in argv (CWE-214).

/search and /graph-explore sit in the auth middleware's PUBLIC_PATHS, so
their fields are UNAUTHENTICATED input. They were unbounded: `query` is
embedded (a TEI round-trip the caller sizes), `top_k`/`rerank_pool` size the
vector search and the cross-encoder batch, and `depth` fans out
multiplicatively per graph hop.
"""

import pytest
from fastapi.testclient import TestClient

from treeloom.application import indexer_service as svc


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    return TestClient(svc.app, raise_server_exceptions=False)


class TestReadEndpointBounds:
    @pytest.mark.parametrize(
        "label,body",
        [
            ("multi-megabyte query", {"query": "x" * 200_000}),
            ("top_k past the cap", {"query": "q", "top_k": 10_000_000}),
            ("top_k of zero", {"query": "q", "top_k": 0}),
            ("negative top_k", {"query": "q", "top_k": -1}),
            ("rerank_pool past the cap", {"query": "q", "rerank_pool": 99_999}),
        ],
    )
    def test_search_rejects_at_the_boundary(self, client, label, body):
        resp = client.post("/search", json={**body, "cross_repo": True})
        assert resp.status_code == 422, label

    def test_graph_explore_bounds_traversal_depth(self, client):
        resp = client.post(
            "/graph-explore", json={"query": "q", "depth": 99, "cross_repo": True}
        )
        assert resp.status_code == 422

    def test_hydrate_list_is_bounded(self, client):
        """Each id costs its own Milvus query."""
        resp = client.post(
            "/hydrate-chunks", json={"ids": ["a.py:1-2"] * 5000}
        )
        assert resp.status_code == 422

    def test_an_ordinary_request_is_not_rejected(self, client, mocker):
        """The caps sit where more stops being useful, not where a real query
        lives — this fails if someone tightens them into the working range."""
        assert len("def parse_manifest(path)") < svc.MAX_QUERY_CHARS
        assert svc.MAX_TOP_K >= 50, "the documented rerank pool default"
        assert svc.MAX_RERANK_POOL >= 300, "the top of the measured pool sweep"


class TestDotenvValueParsing:
    """`AUTH_ENABLED="true"  # turn auth on` produced the value `"true"` —
    quotes included — because the parser tested for a closing quote before
    stripping the comment, and a commented line does not end in one. Every
    `== "true"` check in the codebase then reads that as auth OFF, on a line
    that looks exactly like auth on.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('"true"   # turn auth on', "true"),
            ("'true' # single quotes", "true"),
            ('"true"', "true"),
            ("true", "true"),
            ("val # comment", "val"),
            ("val#withhash", "val#withhash"),   # no space: literal, as before
            ('"a#b" # c', "a#b"),               # hash inside quotes is data
            ('""', ""),
            ("  spaced  ", "spaced"),
        ],
    )
    def test_values(self, raw, expected):
        from treeloom import _parse_dotenv_value

        assert _parse_dotenv_value(raw) == expected

    def test_the_auth_flag_case_end_to_end(self):
        """The one that matters, stated as the comparison the code makes."""
        from treeloom import _parse_dotenv_value

        assert _parse_dotenv_value('"true"  # enable auth').lower() == "true"

    @pytest.mark.parametrize("raw", ['"', '"unbalanced'])
    def test_unbalanced_quotes_stay_literal(self, raw):
        """Truncating to empty would be worse than passing it through: an
        empty DATABASE_URL fails loudly, an empty allow-list may not."""
        from treeloom import _parse_dotenv_value

        assert _parse_dotenv_value(raw) == raw


class TestCliKeyResolution:
    """--key puts the token in the process argument list, readable by any
    local user via ps or /proc for the life of the command, and it lands in
    shell history afterwards. Neither is visible to whoever typed it."""

    def test_key_file_is_preferred(self, tmp_path, monkeypatch):
        from treeloom.infrastructure.cli_auth import resolve_key

        monkeypatch.setenv("TREELOOM_KEY", "from-env")
        f = tmp_path / "token"
        f.write_text("from-file\n")
        assert resolve_key("from-argv", str(f)) == "from-file"

    def test_stdin_sentinel(self, monkeypatch, mocker):
        from treeloom.infrastructure import cli_auth

        mocker.patch.object(cli_auth.sys, "stdin", mocker.Mock(read=lambda: " piped \n"))
        assert cli_auth.resolve_key("-") == "piped"

    def test_literal_key_still_works_but_warns(self, capsys, monkeypatch):
        """Back-compat matters: breaking every documented invocation to fix a
        local disclosure would be a poor trade. Warn instead."""
        from treeloom.infrastructure.cli_auth import resolve_key

        monkeypatch.delenv("TREELOOM_KEY", raising=False)
        assert resolve_key("literal-token") == "literal-token"
        err = capsys.readouterr().err
        assert "ps" in err and "--key-file" in err

    def test_env_fallback_order(self, monkeypatch):
        from treeloom.infrastructure.cli_auth import resolve_key

        monkeypatch.delenv("TREELOOM_KEY", raising=False)
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "mcp-key")
        assert resolve_key(None) == "mcp-key"
        monkeypatch.setenv("TREELOOM_KEY", "primary")
        assert resolve_key(None) == "primary"

    def test_no_token_anywhere_is_none_not_empty_string(self, monkeypatch):
        """Callers build the Authorization header from this; an empty string
        would send `Bearer ` rather than omitting the header."""
        from treeloom.infrastructure.cli_auth import resolve_key

        monkeypatch.delenv("TREELOOM_KEY", raising=False)
        monkeypatch.delenv("TREELOOM_MCP_API_KEY", raising=False)
        assert resolve_key(None) is None

    def test_an_empty_key_file_is_none(self, tmp_path, monkeypatch):
        from treeloom.infrastructure.cli_auth import resolve_key

        monkeypatch.delenv("TREELOOM_KEY", raising=False)
        f = tmp_path / "empty"
        f.write_text("\n")
        assert resolve_key(None, str(f)) is None
