"""Enforcement-wiring tests for _authorize_scope.

Exercises the indexer choke point directly: open-mode bypass, single-source
deny -> 403, shared without all_access -> 403, and the all_access exclusion set
flowing back to the caller. Patches the module-level grant store + owner lookup
so no DB or search pipeline is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state
from treeloom.domain.authorization import GrantStorePort, Role, User


class FakeGrantStore(GrantStorePort):
    def __init__(self) -> None:
        self._grants: dict[tuple[str, str, str], str] = {}

    def add(self, ptype, pid, sid, effect):
        self._grants[(ptype, pid, sid)] = effect

    async def grant(self, principal_type, principal_id, source_id, effect):
        self.add(principal_type, principal_id, source_id, effect)
        return True

    async def revoke(self, principal_type, principal_id, source_id):
        return self._grants.pop((principal_type, principal_id, source_id), None) is not None

    async def list_for_source(self, source_id):
        return []

    async def effects_for_source(self, principals, source_id):
        ps = set(principals)
        return {e for (pt, pid, sid), e in self._grants.items()
                if sid == source_id and (pt, pid) in ps}

    async def denied_sources(self, principals):
        ps = set(principals)
        return [sid for (pt, pid, sid), e in self._grants.items()
                if e == "deny" and (pt, pid) in ps]


def _req(user: User | None) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(user=user), headers={})


class FakeAudit:
    def __init__(self):
        self.records = []

    async def record(self, entry):
        self.records.append(entry)

    async def recent(self, **k):
        return [r.to_dict() for r in self.records]


@pytest.fixture
def audit(mocker):
    store = FakeAudit()
    mocker.patch.object(_state, "_search_audit", store)
    return store


@pytest.fixture
def grants(mocker, audit):
    store = FakeGrantStore()
    mocker.patch.object(_state, "_grant_store", store)
    # No source has an owner unless a test says so.
    mocker.patch.object(_authz, "_load_source", mocker.AsyncMock(return_value=None))
    return store


@pytest.fixture
def auth_on(mocker):
    mocker.patch.dict("os.environ", {"AUTH_ENABLED": "true"}, clear=False)


@pytest.mark.asyncio
async def test_open_mode_returns_none(mocker, grants):
    mocker.patch.dict("os.environ", {"AUTH_ENABLED": "false"}, clear=False)
    # No user, enforcement off -> no checks.
    assert await _authz._authorize_scope(_req(None), "repoA") is None


@pytest.mark.asyncio
async def test_search_open_escape_hatch(mocker, grants):
    mocker.patch.dict(
        "os.environ", {"AUTH_ENABLED": "true", "TREELOOM_SEARCH_OPEN": "1"}, clear=False
    )
    u = User(id="u1")
    assert await _authz._authorize_scope(_req(u), "repoA") is None


@pytest.mark.asyncio
async def test_single_source_denied_raises_403(auth_on, grants):
    u = User(id="u1")  # no grant, not owner
    with pytest.raises(HTTPException) as exc:
        await _authz._authorize_scope(_req(u), "repoA")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_single_source_granted_returns_none(auth_on, grants):
    grants.add("user", "u1", "repoA", "allow")
    assert await _authz._authorize_scope(_req(User(id="u1")), "repoA") is None


@pytest.mark.asyncio
async def test_shared_without_all_access_raises_403(auth_on, grants):
    with pytest.raises(HTTPException) as exc:
        await _authz._authorize_scope(_req(User(id="u1")), None)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_shared_all_access_returns_exclusions(auth_on, grants):
    grants.add("user", "ci", "secret", "deny")
    u = User(id="ci", all_access=True)
    excl = await _authz._authorize_scope(_req(u), None)
    assert excl == ["secret"]


@pytest.mark.asyncio
async def test_admin_bypasses_everything(auth_on, grants):
    admin = User(id="a", role=Role.ADMIN)
    assert await _authz._authorize_scope(_req(admin), "any-repo") is None
    assert await _authz._authorize_scope(_req(admin), None) == []


# ── a derived non-admin principal (a search-scoped key owned by an
# admin) must NOT get the admin bypass; it is governed by all_access + grants ──


@pytest.mark.asyncio
async def test_derived_non_admin_no_admin_bypass(auth_on, grants):
    """A search-scoped key whose owner is an admin resolves to role USER.

    With no grant and no all_access, it must be denied on a pinned source — it
    does NOT inherit the admin see-everything bypass.
    """
    derived = User(id="adminowner", role=Role.USER, all_access=False)
    with pytest.raises(HTTPException) as exc:
        await _authz._authorize_scope(_req(derived), "repoA")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_derived_non_admin_governed_by_grant(auth_on, grants):
    """An explicit allow grant lets the derived USER principal through."""
    grants.add("user", "adminowner", "repoA", "allow")
    derived = User(id="adminowner", role=Role.USER, all_access=False)
    assert await _authz._authorize_scope(_req(derived), "repoA") is None


@pytest.mark.asyncio
async def test_derived_non_admin_all_access_still_denied_repo(auth_on, grants):
    """all_access carries over but an explicit deny still wins (admin > deny ...).

    A derived principal that kept the owner's all_access can run shared queries
    but an explicit per-source deny still carves out the sensitive repo.
    """
    grants.add("user", "adminowner", "secret", "deny")
    derived = User(id="adminowner", role=Role.USER, all_access=True)
    # Shared query: allowed, with the denied source excluded.
    excl = await _authz._authorize_scope(_req(derived), None)
    assert excl == ["secret"]
    # Pinned to the denied source: 403 despite all_access (deny beats all_access).
    with pytest.raises(HTTPException) as exc:
        await _authz._authorize_scope(_req(derived), "secret")
    assert exc.value.status_code == 403


# ── audit emission ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_records_deny(auth_on, grants, audit):
    with pytest.raises(HTTPException):
        await _authz._authorize_scope(_req(User(id="u1")), "repoA", action="search")
    assert len(audit.records) == 1
    r = audit.records[0]
    assert (r.user_id, r.action, r.scope, r.decision, r.source_id) == (
        "u1", "search", "source", "deny", "repoA"
    )


@pytest.mark.asyncio
async def test_audit_records_allow_with_groups(auth_on, grants, audit):
    grants.add("group", "team", "repoB", "allow")
    u = User(id="u1", group_ids=["team"])
    await _authz._authorize_scope(_req(u), "repoB", action="find_definition")
    r = audit.records[0]
    assert r.decision == "allow" and r.action == "find_definition"
    assert r.principal_groups == ("team",)


@pytest.mark.asyncio
async def test_open_mode_emits_no_audit(mocker, grants, audit):
    mocker.patch.dict("os.environ", {"AUTH_ENABLED": "false"}, clear=False)
    await _authz._authorize_scope(_req(None), "repoA")
    assert audit.records == []
