"""Decision-table tests for the per-source AuthorizationService.

Detroit-style: a real in-memory GrantStorePort fake and a plain async owner
lookup — no mocking. Exercises the precedence rules directly:
admin > deny > all_access > allow > owner > deny.
"""

from __future__ import annotations

import pytest

from treeloom.domain.authorization import GrantStorePort, Role, User
from treeloom.domain.authorization.access import (
    AuthorizationService,
    decide_shared,
    decide_single_source,
    principals_for,
)


class FakeGrantStore(GrantStorePort):
    """In-memory grants keyed by (principal_type, principal_id, source_id)."""

    def __init__(self) -> None:
        self._grants: dict[tuple[str, str, str], str] = {}

    def add(self, ptype: str, pid: str, source_id: str, effect: str) -> None:
        self._grants[(ptype, pid, source_id)] = effect

    async def grant(self, principal_type, principal_id, source_id, effect) -> bool:
        self.add(principal_type, principal_id, source_id, effect)
        return True

    async def revoke(self, principal_type, principal_id, source_id) -> bool:
        return self._grants.pop((principal_type, principal_id, source_id), None) is not None

    async def list_for_source(self, source_id) -> list[dict]:
        return [
            {"principal_type": pt, "principal_id": pid, "source_id": sid, "effect": e}
            for (pt, pid, sid), e in self._grants.items()
            if sid == source_id
        ]

    async def effects_for_source(self, principals, source_id) -> set[str]:
        ps = set(principals)
        return {
            e for (pt, pid, sid), e in self._grants.items()
            if sid == source_id and (pt, pid) in ps
        }

    async def denied_sources(self, principals) -> list[str]:
        ps = set(principals)
        return [
            sid for (pt, pid, sid), e in self._grants.items()
            if e == "deny" and (pt, pid) in ps
        ]


def _user(**kw) -> User:
    return User(id=kw.pop("id", "u1"), **kw)


def _service(grants: FakeGrantStore, owner: str | None = None) -> AuthorizationService:
    async def owner_of(source_id: str):
        return owner
    return AuthorizationService(grants=grants, owner_of=owner_of)


# ── principals_for ─────────────────────────────────────────────────

def test_principals_include_user_and_groups():
    u = _user(id="u1", group_ids=["g1", "g2"])
    assert principals_for(u) == [("user", "u1"), ("group", "g1"), ("group", "g2")]


# ── single-source decision table (pure) ────────────────────────────

def test_pure_admin_allows_even_with_deny():
    u = _user(role=Role.ADMIN)
    d = decide_single_source(user=u, source_id="s", effects={"deny"}, owner_id=None)
    assert d.allowed and d.reason == "admin"


def test_pure_deny_beats_all_access():
    u = _user(all_access=True)
    d = decide_single_source(user=u, source_id="s", effects={"deny"}, owner_id=None)
    assert not d.allowed and d.reason == "explicit-deny"


def test_pure_all_access_allows():
    u = _user(all_access=True)
    d = decide_single_source(user=u, source_id="s", effects=set(), owner_id=None)
    assert d.allowed and d.reason == "all_access"


def test_pure_explicit_allow():
    u = _user()
    d = decide_single_source(user=u, source_id="s", effects={"allow"}, owner_id=None)
    assert d.allowed and d.reason == "explicit-allow"


def test_pure_owner_allows():
    u = _user(id="owner1")
    d = decide_single_source(user=u, source_id="s", effects=set(), owner_id="owner1")
    assert d.allowed and d.reason == "owner"


def test_pure_no_grant_denies():
    u = _user(id="someone")
    d = decide_single_source(user=u, source_id="s", effects=set(), owner_id="other")
    assert not d.allowed and d.reason == "no-grant"


# ── shared decision table (pure) ───────────────────────────────────

def test_pure_shared_admin_no_exclusions():
    d = decide_shared(user=_user(role=Role.ADMIN), denied_source_ids=["x"])
    assert d.allowed and d.excluded_source_ids == ()


def test_pure_shared_requires_all_access():
    d = decide_shared(user=_user(all_access=False), denied_source_ids=[])
    assert not d.allowed and d.reason == "no-all-access"


def test_pure_shared_all_access_returns_exclusions_sorted_unique():
    d = decide_shared(user=_user(all_access=True), denied_source_ids=["b", "a", "b"])
    assert d.allowed and d.excluded_source_ids == ("a", "b")


# ── service wiring (async, real fake store) ────────────────────────

@pytest.mark.asyncio
async def test_service_user_grant_allows():
    g = FakeGrantStore()
    g.add("user", "u1", "repoA", "allow")
    d = await _service(g).authorize_source(_user(id="u1"), "repoA")
    assert d.allowed


@pytest.mark.asyncio
async def test_service_group_grant_allows():
    g = FakeGrantStore()
    g.add("group", "team", "repoB", "allow")
    u = _user(id="u1", group_ids=["team"])
    d = await _service(g).authorize_source(u, "repoB")
    assert d.allowed


@pytest.mark.asyncio
async def test_service_unauthorized_denies():
    g = FakeGrantStore()
    d = await _service(g, owner="someone-else").authorize_source(_user(id="u1"), "repoX")
    assert not d.allowed


@pytest.mark.asyncio
async def test_service_owner_allows_without_grant():
    g = FakeGrantStore()
    d = await _service(g, owner="u1").authorize_source(_user(id="u1"), "repoMine")
    assert d.allowed


@pytest.mark.asyncio
async def test_service_ci_all_access_minus_exclusion():
    """The CI-agent case: all_access principal, one repo explicitly denied."""
    g = FakeGrantStore()
    g.add("group", "ci", "secret-repo", "deny")
    u = _user(id="ci-bot", all_access=True, group_ids=["ci"])
    # shared query: allowed, but the secret repo is excluded
    shared = await _service(g).authorize_shared(u)
    assert shared.allowed and shared.excluded_source_ids == ("secret-repo",)
    # and a direct pin to the denied repo is rejected
    pinned = await _service(g).authorize_source(u, "secret-repo")
    assert not pinned.allowed


@pytest.mark.asyncio
async def test_service_shared_without_all_access_denied():
    g = FakeGrantStore()
    d = await _service(g).authorize_shared(_user(id="u1"))
    assert not d.allowed
