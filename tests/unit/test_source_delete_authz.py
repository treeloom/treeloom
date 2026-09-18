"""Authorization on DELETE /sources/{source_id}.

The handler used to take no `request` parameter at all, so it could not check
anything. source_ids are enumerable from GET /sources, and the deletion is
bulk and irreversible — chunks from the vector store, entities and community
nodes from the graph, and the registry row. So any caller who could reach the
indexer could wipe every indexed source (CWE-862), and with auth on, any
authenticated user could wipe another principal's data (CWE-639 / IDOR).

It is now routed through `_authorize_scope`, the same per-source choke point
the read endpoints use: 401 without a credential, 403 when the ACL denies,
and an audit record either way.

Two behaviours these tests pin deliberately, because both look like gaps:

* With AUTH_ENABLED off the guard is a no-op. That matches every other
  endpoint — enforcement is globally opt-in — and what keeps deletion off the
  network by default is the loopback INDEXER_HOST bind, not this check.
* The ACL has a single allow/deny axis, so a principal merely *authorized* for
  a source can delete it. That is the documented model, not an oversight here;
  a read/write split would be a policy change.
"""

import pytest
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


@pytest.fixture(autouse=True)
def auth_off_by_default(monkeypatch):
    """Most tests here drive the ACL gate directly. AUTH_ENABLED off makes
    `_require_scope` a no-op so they reach it; the scope tests set it on."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)


@pytest.fixture
def no_side_effects(mocker):
    """Stub every destructive call so a test that wrongly reaches the body
    fails loudly on an assertion rather than by touching a real store."""
    return {
        "chunks": mocker.patch.object(svc, "delete_chunks_by_source"),
        "graph": mocker.patch.object(
            svc.graph_store, "delete_source", new=mocker.AsyncMock()
        ),
        "communities": mocker.patch.object(
            svc.graph_store, "delete_source_communities", new=mocker.AsyncMock()
        ),
        "registry": mocker.patch.object(
            _state._source_repo, "delete", new=mocker.AsyncMock()
        ),
    }


class TestGuardIsInvoked:
    @pytest.mark.asyncio
    async def test_denied_caller_gets_403_and_deletes_nothing(
        self, mocker, no_side_effects
    ):
        """The decisive test: a 403 must abort BEFORE any store is touched.
        Deletion is irreversible, so 'authorized late' is the same as
        unauthorized."""
        mocker.patch.object(
            _authz,
            "_authorize_scope",
            new=mocker.AsyncMock(
                side_effect=HTTPException(403, "Not authorized for this source")
            ),
        )
        with pytest.raises(HTTPException) as exc:
            await svc.remove_source("victim-source", mocker.Mock())
        assert exc.value.status_code == 403

        no_side_effects["chunks"].assert_not_called()
        no_side_effects["graph"].assert_not_called()
        no_side_effects["communities"].assert_not_called()
        no_side_effects["registry"].assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthenticated_401_deletes_nothing(self, mocker, no_side_effects):
        mocker.patch.object(
            _authz,
            "_authorize_scope",
            new=mocker.AsyncMock(
                side_effect=HTTPException(401, "Authentication required")
            ),
        )
        with pytest.raises(HTTPException) as exc:
            await svc.remove_source("victim-source", mocker.Mock())
        assert exc.value.status_code == 401
        no_side_effects["chunks"].assert_not_called()
        no_side_effects["graph"].assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_caller_deletes_everything(self, mocker, no_side_effects):
        """All four stores must still be cleared — community nodes are
        namespaced separately and are easy to leave behind."""
        authorize = mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        result = await svc.remove_source("my-source", mocker.Mock())

        assert result == {"status": "deleted", "source_id": "my-source"}
        no_side_effects["chunks"].assert_called_once_with("my-source")
        no_side_effects["graph"].assert_awaited_once_with("my-source")
        no_side_effects["communities"].assert_awaited_once_with("my-source")
        no_side_effects["registry"].assert_awaited_once_with("my-source")
        # The guard must be asked about THIS source, not a default.
        assert authorize.await_args[0][1] == "my-source"

    @pytest.mark.asyncio
    async def test_guard_receives_a_distinguishable_action(self, mocker, no_side_effects):
        """The audit trail should distinguish a deletion from a search."""
        authorize = mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        await svc.remove_source("my-source", mocker.Mock())
        assert authorize.await_args.kwargs.get("action") == "remove_source"


class TestResilienceIsPreservedAfterTheGuard:
    """Best-effort cleanup semantics must survive the change: a store that is
    down should not strand the other three."""

    @pytest.mark.asyncio
    async def test_vector_store_failure_still_clears_graph_and_registry(
        self, mocker, no_side_effects
    ):
        mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        no_side_effects["chunks"].side_effect = RuntimeError("milvus down")

        result = await svc.remove_source("my-source", mocker.Mock())

        assert result["status"] == "deleted"
        no_side_effects["graph"].assert_awaited_once()
        no_side_effects["registry"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_registry_failure_is_swallowed(self, mocker, no_side_effects):
        mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        no_side_effects["registry"].side_effect = RuntimeError("pg down")
        result = await svc.remove_source("my-source", mocker.Mock())
        assert result["status"] == "deleted"


class TestEnforcementOffIsUnchanged:
    @pytest.mark.asyncio
    async def test_open_mode_still_deletes(self, mocker, monkeypatch, no_side_effects):
        """_authorize_scope returns None when enforcement is off. Pinned so a
        future edit cannot quietly make deletion require auth in open mode —
        that would be a behaviour change, not a fix."""
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        result = await svc.remove_source("my-source", mocker.Mock())
        assert result["status"] == "deleted"
        no_side_effects["chunks"].assert_called_once_with("my-source")


class TestTokenScopeIsCheckedToo:
    """Two independent gates. The grant check asks whether the *principal* is
    authorized for this source; the scope check asks what this *credential* is
    permitted to do. A search-only PAT owned by someone with full access would
    clear the first on its own — which is exactly what scopes exist to stop.
    """

    @pytest.mark.asyncio
    async def test_search_only_credential_cannot_delete(
        self, mocker, monkeypatch, no_side_effects
    ):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        authorize = mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"search"}

        with pytest.raises(HTTPException) as exc:
            await svc.remove_source("my-source", request)

        assert exc.value.status_code == 403
        assert "index" in str(exc.value.detail)
        no_side_effects["chunks"].assert_not_called()
        authorize.assert_not_awaited(), "scope is cheap — check it before the ACL"

    @pytest.mark.asyncio
    async def test_index_scoped_credential_may_delete(
        self, mocker, monkeypatch, no_side_effects
    ):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"search", "index"}

        result = await svc.remove_source("my-source", request)
        assert result["status"] == "deleted"
        no_side_effects["chunks"].assert_called_once_with("my-source")

    @pytest.mark.asyncio
    async def test_no_credential_is_401_not_403(
        self, mocker, monkeypatch, no_side_effects
    ):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = None

        with pytest.raises(HTTPException) as exc:
            await svc.remove_source("my-source", request)
        assert exc.value.status_code == 401
        no_side_effects["chunks"].assert_not_called()


class TestSearchOpenDoesNotOpenDeletion:
    """TREELOOM_SEARCH_OPEN makes *search* anonymous while indexing stays
    authenticated. Routing deletion through the search predicate meant setting
    it also reverted this endpoint to "any caller may destroy any source" —
    the exact CWE-862 the guard closes, re-opened by a flag whose name
    mentions only search.
    """

    def test_write_enforcement_ignores_the_search_escape_hatch(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("TREELOOM_SEARCH_OPEN", "1")
        assert _authz._search_auth_enforced() is False
        assert _authz._write_auth_enforced() is True

    def test_auth_disabled_still_turns_everything_off(self, monkeypatch):
        """AUTH_ENABLED stays the single global switch — the write predicate
        is narrower than the search one, not independent of it."""
        monkeypatch.setenv("AUTH_ENABLED", "false")
        monkeypatch.setenv("TREELOOM_SEARCH_OPEN", "1")
        assert _authz._write_auth_enforced() is False

    @pytest.mark.asyncio
    async def test_delete_still_checks_the_acl_under_search_open(
        self, mocker, monkeypatch, no_side_effects
    ):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("TREELOOM_SEARCH_OPEN", "1")
        mocker.patch.object(
            _authz,
            "_require_authenticated_user",
            new=mocker.AsyncMock(
                side_effect=HTTPException(401, "Authentication required")
            ),
        )
        # Clear the scope gate so the ACL gate is what this test measures.
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"index"}

        with pytest.raises(HTTPException) as exc:
            await svc.remove_source("victim-source", request)
        assert exc.value.status_code == 401
        no_side_effects["chunks"].assert_not_called()
        no_side_effects["graph"].assert_not_called()
