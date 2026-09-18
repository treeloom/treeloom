"""Tests for ACL store hydration + graceful degradation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from treeloom.adapters.authorization.user_store import _row_to_user, PostgreSQLUserStore
from treeloom.adapters.authorization.group_store import PostgreSQLGroupStore
from treeloom.adapters.authorization.grant_store import PostgreSQLGrantStore


def _base_row(**over) -> dict:
    row = {
        "id": "u1",
        "username": "alice",
        "email": "a@x.io",
        "api_key_hash": "h",
        "role": "user",
        "active": True,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    row.update(over)
    return row


def test_row_to_user_plain_row_defaults_authz_empty():
    u = _row_to_user(_base_row())
    assert u.all_access is False
    assert u.group_ids == []


def test_row_to_user_hydrated_effective_all_access_and_groups():
    row = _base_row(
        all_access=False,
        effective_all_access=True,  # OR'd from a group
        group_ids=["g1", "g2"],
    )
    u = _row_to_user(row)
    assert u.all_access is True
    assert u.group_ids == ["g1", "g2"]


def test_row_to_user_falls_back_to_own_all_access_column():
    # No effective_all_access key (e.g. list_users SELECT *), but column present.
    row = _base_row(all_access=True, group_ids=None)
    u = _row_to_user(row)
    assert u.all_access is True
    assert u.group_ids == []


# ── graceful degradation: no pool -> no-op returns ─────────────────

@pytest.mark.asyncio
async def test_user_store_update_all_access_no_pool():
    store = PostgreSQLUserStore(pool=None)
    assert await store.update_all_access("u1", True) is False


@pytest.mark.asyncio
async def test_group_store_degrades_without_pool():
    store = PostgreSQLGroupStore(pool=None)
    assert await store.create_group("team") is None
    assert await store.list_groups() == []
    assert await store.add_member("g", "u") is False
    assert await store.list_members("g") == []


@pytest.mark.asyncio
async def test_grant_store_writes_and_listing_degrade_without_pool():
    """Management operations still degrade gracefully — a failed write reports
    failure, and an admin listing an empty set is harmless."""
    store = PostgreSQLGrantStore(pool=None)
    assert await store.grant("user", "u1", "s1", "allow") is False
    assert await store.list_for_source("s1") == []


@pytest.mark.asyncio
async def test_authorization_reads_fail_loud_without_pool():
    """The two hot-path reads must NOT degrade, and this test previously
    asserted that they did.

    An empty result from these is a real answer about a real caller — "this
    principal has no deny grants" — so it cannot double as an error signal.
    For an all_access caller the deny grant is the only thing withholding a
    source, so returning empty on a database blip grants exactly the access
    the grant exists to prevent (CWE-636).
    """
    from treeloom.adapters.authorization.grant_store import GrantLookupUnavailable

    store = PostgreSQLGrantStore(pool=None)
    with pytest.raises(GrantLookupUnavailable):
        await store.effects_for_source([("user", "u1")], "s1")
    with pytest.raises(GrantLookupUnavailable):
        await store.denied_sources([("user", "u1")])


@pytest.mark.asyncio
async def test_grant_store_rejects_bad_input():
    store = PostgreSQLGrantStore(pool=None)
    with pytest.raises(ValueError):
        await store.grant("robot", "u1", "s1", "allow")
    with pytest.raises(ValueError):
        await store.grant("user", "u1", "s1", "maybe")


@pytest.mark.asyncio
async def test_grant_store_empty_principals_short_circuit():
    store = PostgreSQLGrantStore(pool=None)
    # No principals -> empty result without touching the (absent) pool.
    assert await store.effects_for_source([], "s1") == set()
    assert await store.denied_sources([]) == []
