"""MCP read tools forward the caller's bearer, never the service key.

The indexer enforces per-principal authorization against request.state.user, so
read tools must carry the *caller's* credential. A missing credential must
forward nothing — never silently escalate to TREELOOM_MCP_API_KEY.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("INDEXER_URL", "http://indexer.test")

import treeloom.application.mcp_server as mcp_server  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_mcp_credential_env(monkeypatch):
    """Make this module hermetic against a developer's ambient environment.

    A test in this file leaving one of these three vars set (from a prior
    test, or from the developer's own shell — e.g. TREELOOM_MCP_TOKEN for a
    real PAT) has now caused a false pass/false failure at four separate
    sites across three review rounds, because the credential-precedence
    logic under test reads straight from os.environ. Clearing all three
    before every test — rather than patching yet another individual test —
    closes the whole defect class instead of its latest instance. Tests that
    want one of these vars set do so via monkeypatch.setenv() after this
    fixture has run, which still wins (monkeypatch undoes in LIFO order).
    """
    monkeypatch.delenv("TREELOOM_MCP_TOKEN", raising=False)
    monkeypatch.delenv("TREELOOM_MCP_API_KEY", raising=False)
    monkeypatch.delenv("TREELOOM_MCP_SERVICE_ACCOUNT", raising=False)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Captures the headers passed to the indexer call.

    POST (search tools) gets a search-shaped dict; GET (find tools) gets the
    list of entities they expect.
    """

    def __init__(self, captured):
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self._captured["headers"] = headers
        # Superset payload: satisfies both the search-shaped reads and the
        # job-shaped writes, so one fake serves every tool under test.
        return _FakeResp(
            {
                "chunks": [],
                "neighbors": [],
                "community_summaries": {},
                "job_id": "job-1",
                "status": "queued",
                "source_id": "src-1",
            }
        )

    async def get(self, url, params=None, headers=None):
        self._captured["headers"] = headers
        return _FakeResp([])


def _ctx_with_bearer(value: str | None):
    headers = {"authorization": value} if value else {}
    request = SimpleNamespace(headers=headers)
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


@pytest.fixture
def capture(mocker):
    captured: dict = {}

    def _factory(*a, **k):
        return _FakeClient(captured)

    mocker.patch.object(mcp_server.httpx, "AsyncClient", _factory)
    return captured


# ── _caller_auth_headers unit behavior ─────────────────────────────

def test_forwards_inbound_bearer():
    ctx = _ctx_with_bearer("Bearer user-token-123")
    assert mcp_server._caller_auth_headers(ctx) == {"Authorization": "Bearer user-token-123"}


def test_no_ctx_forwards_nothing():
    assert mcp_server._caller_auth_headers(None) == {}


def test_no_inbound_bearer_forwards_nothing_even_with_service_key(monkeypatch):
    monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
    ctx = _ctx_with_bearer(None)
    # Must NOT fall back to the service key for reads.
    assert mcp_server._caller_auth_headers(ctx) == {}


# ── end-to-end through the tools ───────────────────────────────────

@pytest.mark.asyncio
async def test_search_code_forwards_caller_bearer(capture):
    await mcp_server.search_code(
        "q", source_id="repoA", ctx=_ctx_with_bearer("Bearer alice")
    )
    assert capture["headers"] == {"Authorization": "Bearer alice"}


@pytest.mark.asyncio
async def test_find_definition_forwards_caller_bearer(capture):
    await mcp_server.find_definition("Foo", ctx=_ctx_with_bearer("Bearer bob"))
    assert capture["headers"] == {"Authorization": "Bearer bob"}


@pytest.mark.asyncio
async def test_read_tool_without_credential_sends_no_auth(capture, monkeypatch):
    monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
    await mcp_server.search_code("q", source_id="repoA", ctx=None)
    assert capture["headers"] == {}


# All nine per-repo-scoped read tools, with the minimum kwargs each needs
# (every tool accepts ctx= separately). A first pass of this branch routed
# all nine through `_tool_auth_headers` and passed every test that existed
# at the time — only `search_code` had a regression guard, so the other
# eight had no test standing between them and that exact mistake recurring.
_READ_TOOL_MIN_ARGS: list[tuple[str, dict]] = [
    ("search_code", dict(query="q", source_id="repoA")),
    ("search_code_enhanced", dict(query="q", source_id="repoA")),
    ("hydrate_chunks", dict(ids=["chunk-1"], source_id="repoA")),
    ("explain_code", dict(query="q", source_id="repoA")),
    ("graph_explore", dict(query="q", source_id="repoA")),
    ("find_definition", dict(name="Foo")),
    ("find_callers", dict(name_or_id="Foo")),
    ("find_references", dict(name_or_id="Foo")),
    ("list_job_errors", dict(job_id="job-1")),
    # Not repo-scoped but per-principal filtered: GET /sources and its
    # staleness view return only the caller's own records. A service-account
    # fallback here would answer with the whole fleet.
    ("list_indexed_sources", dict()),
    ("source_staleness", dict(source_id="repoA")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,kwargs",
    _READ_TOOL_MIN_ARGS,
    ids=[name for name, _ in _READ_TOOL_MIN_ARGS],
)
async def test_read_tool_never_uses_service_account_even_when_opted_in(
    capture, monkeypatch, tool_name, kwargs
):
    """Per-repo-scoped reads must never fall through to the shared service
    account, even when an operator has opted in TREELOOM_MCP_SERVICE_ACCOUNT=
    true for write/admin tools. Regression guard for `_read_tool_auth_headers`
    staying distinct from `_tool_auth_headers`, parametrized
    over every read tool rather than just `search_code`.
    """
    monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
    monkeypatch.setenv("TREELOOM_MCP_SERVICE_ACCOUNT", "true")
    tool = getattr(mcp_server, tool_name)
    await tool(**kwargs, ctx=None)
    assert "Authorization" not in (capture["headers"] or {})


# ── Write/admin tools: same credential policy as reads ──────────────────
# These tools used to inject TREELOOM_MCP_API_KEY unconditionally, so an
# anonymous caller on the HTTP transport inherited service-account authority
# over the indexer (index any path, delete any source). The caller's own
# credential is now the only one used by default.


class TestWriteToolCredentialPolicy:
    @pytest.mark.asyncio
    async def test_caller_bearer_is_forwarded(self, capture, monkeypatch):
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
        await mcp_server.index_file(
            file_path="/repo/a.py", ctx=_ctx_with_bearer("Bearer caller-token")
        )
        assert capture["headers"]["Authorization"] == "Bearer caller-token"

    @pytest.mark.asyncio
    async def test_no_caller_credential_sends_nothing(self, capture, monkeypatch):
        """The service key must NOT stand in for an uncredentialed caller."""
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=_ctx_with_bearer(None))
        assert "Authorization" not in (capture["headers"] or {})

    @pytest.mark.asyncio
    async def test_service_account_used_only_when_opted_in(self, capture, monkeypatch):
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
        monkeypatch.setenv("TREELOOM_MCP_SERVICE_ACCOUNT", "true")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer service-secret"

    @pytest.mark.asyncio
    async def test_caller_credential_wins_over_service_account(self, capture, monkeypatch):
        """Opting in must never let the service account override a real caller."""
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
        monkeypatch.setenv("TREELOOM_MCP_SERVICE_ACCOUNT", "true")
        await mcp_server.index_file(
            file_path="/repo/a.py", ctx=_ctx_with_bearer("Bearer caller-token")
        )
        assert capture["headers"]["Authorization"] == "Bearer caller-token"

    @pytest.mark.asyncio
    async def test_opt_in_is_exact_match_not_truthy(self, capture, monkeypatch):
        """A stray value like '0' or 'no' must not enable the fallback."""
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
        for value in ("0", "no", "false", "1", "yes"):
            monkeypatch.setenv("TREELOOM_MCP_SERVICE_ACCOUNT", value)
            await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
            sent = (capture["headers"] or {}).get("Authorization")
            assert sent is None, f"{value!r} must not enable the service account"


class TestStdioEnvCredential:
    """On stdio there is no HTTP request, so the caller's credential arrives
    via the environment — the model the MCP spec prescribes for stdio."""

    @pytest.mark.asyncio
    async def test_env_token_used_when_no_caller_bearer(self, capture, monkeypatch):
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "user-token")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer user-token"

    @pytest.mark.asyncio
    async def test_env_token_applies_to_read_tools_too(self, capture, monkeypatch):
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "user-token")
        await mcp_server.search_code(query="auth", source_id="repoA", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer user-token"

    @pytest.mark.asyncio
    async def test_caller_bearer_wins_over_env_token(self, capture, monkeypatch):
        """On a shared HTTP transport the per-request identity must win."""
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "user-token")
        await mcp_server.index_file(
            file_path="/repo/a.py", ctx=_ctx_with_bearer("Bearer caller-token")
        )
        assert capture["headers"]["Authorization"] == "Bearer caller-token"

    @pytest.mark.asyncio
    async def test_env_token_wins_over_service_account(self, capture, monkeypatch):
        """A real user credential outranks the shared service account."""
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "user-token")
        monkeypatch.setenv("TREELOOM_MCP_API_KEY", "service-secret")
        monkeypatch.setenv("TREELOOM_MCP_SERVICE_ACCOUNT", "true")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer user-token"

    @pytest.mark.asyncio
    async def test_blank_env_token_is_not_a_credential(self, capture, monkeypatch):
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "   ")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert "Authorization" not in (capture["headers"] or {})

    @pytest.mark.asyncio
    async def test_env_token_with_bearer_prefix_is_not_doubled(self, capture, monkeypatch):
        """A user pasting the full header value ('Bearer xyz') instead of the
        bare token must not produce 'Authorization: Bearer Bearer xyz'."""
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "Bearer user-token")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer user-token"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pasted", ["bearer user-token", "BEARER user-token"])
    async def test_bearer_prefix_strip_is_case_insensitive(
        self, capture, monkeypatch, pasted
    ):
        """RFC 7235 makes the auth scheme case-insensitive, so a copied
        'bearer xyz' is a legitimate value. A case-sensitive strip would send
        'Bearer bearer xyz' and 401 for a reason nothing in the logs explains."""
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", pasted)
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer user-token"

    @pytest.mark.asyncio
    async def test_a_token_that_merely_starts_with_bearer_is_untouched(
        self, capture, monkeypatch
    ):
        """Only the scheme prefix is stripped — 'bearerish' is a whole token,
        and the space is what distinguishes them."""
        monkeypatch.setenv("TREELOOM_MCP_TOKEN", "bearerish-token")
        await mcp_server.index_file(file_path="/repo/a.py", ctx=None)
        assert capture["headers"]["Authorization"] == "Bearer bearerish-token"
