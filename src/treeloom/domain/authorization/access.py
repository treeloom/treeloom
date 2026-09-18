"""Per-source authorization decisions.

The model is **scope-level rejection**, not per-hit result filtering:

* A query pinned to a single ``source_id`` is allowed only if the caller is
  authorized for that source.
* A shared / cross-repo query (``cross_repo=true`` or a ``path_prefix`` that
  isn't source-pinned) is allowed only if the caller has ``all_access``; the
  sources they are explicitly *denied* are returned so the query layer can add
  a ``source_id NOT IN (...)`` prefilter.

Precedence for a single source:  admin > deny > all_access > allow > owner > deny.
``deny`` beats ``all_access`` so an all-access principal (e.g. a CI agent) can
still be carved out of individual sensitive repos.

The decision *table* lives in the two pure functions ``decide_single_source``
and ``decide_shared`` (sync, no I/O — trivially unit-testable). The async
``AuthorizationService`` only fetches the inputs (grants, owner) and delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from treeloom.domain.authorization import GrantStorePort, Role, User

# A principal is (type, id), e.g. ("user", "abc") or ("group", "team").
Principal = tuple[str, str]

# Async lookup of a source's owner (source_records.created_by), or None.
OwnerLookup = Callable[[str], Awaitable[Optional[str]]]


@dataclass(frozen=True)
class ScopeDecision:
    """Outcome of an authorization check.

    ``excluded_source_ids`` is only meaningful for shared-index decisions: it
    lists sources the (all_access) caller is denied and must be filtered out.
    """

    allowed: bool
    excluded_source_ids: tuple[str, ...] = ()
    reason: str = ""


def principals_for(user: User) -> list[Principal]:
    """The user themself plus each group they belong to."""
    principals: list[Principal] = [("user", user.id)]
    principals.extend(("group", gid) for gid in user.group_ids)
    return principals


def decide_single_source(
    *,
    user: User,
    source_id: str,
    effects: set[str],
    owner_id: Optional[str],
) -> ScopeDecision:
    """Pure decision for a query pinned to one source.

    ``effects`` is the set of grant effects ({'allow'}, {'deny'}, both, or
    empty) that any of the caller's principals hold on ``source_id``.
    ``owner_id`` is the source's ``created_by`` (or None).
    """
    if user.role == Role.ADMIN:
        return ScopeDecision(True, reason="admin")
    if "deny" in effects:
        return ScopeDecision(False, reason="explicit-deny")
    if user.all_access:
        return ScopeDecision(True, reason="all_access")
    if "allow" in effects:
        return ScopeDecision(True, reason="explicit-allow")
    if owner_id and owner_id == user.id:
        return ScopeDecision(True, reason="owner")
    return ScopeDecision(False, reason="no-grant")


def decide_shared(
    *, user: User, denied_source_ids: list[str]
) -> ScopeDecision:
    """Pure decision for a shared / cross-repo query.

    Allowed only with all_access (or admin); the denied set is surfaced as the
    exclusion list to prefilter at the store.
    """
    if user.role == Role.ADMIN:
        return ScopeDecision(True, reason="admin")
    if not user.all_access:
        return ScopeDecision(False, reason="no-all-access")
    return ScopeDecision(
        True,
        excluded_source_ids=tuple(sorted(set(denied_source_ids))),
        reason="all_access",
    )


@dataclass
class AuthorizationService:
    """Resolves authorization inputs and applies the decision table.

    Detroit-testable: inject a real (in-memory) ``GrantStorePort`` and a plain
    async ``owner_of`` callable in tests — no mocking required.
    """

    grants: GrantStorePort
    owner_of: OwnerLookup

    async def authorize_source(self, user: User, source_id: str) -> ScopeDecision:
        """Authorize a single-source query."""
        if user.role == Role.ADMIN:
            return ScopeDecision(True, reason="admin")
        principals = principals_for(user)
        effects = await self.grants.effects_for_source(principals, source_id)
        owner_id: Optional[str] = None
        # Owner is only consulted when nothing else has decided it.
        if "deny" not in effects and not user.all_access and "allow" not in effects:
            owner_id = await self.owner_of(source_id)
        return decide_single_source(
            user=user, source_id=source_id, effects=effects, owner_id=owner_id
        )

    async def authorize_shared(self, user: User) -> ScopeDecision:
        """Authorize a shared / cross-repo query."""
        if user.role == Role.ADMIN:
            return ScopeDecision(True, reason="admin")
        if not user.all_access:
            return ScopeDecision(False, reason="no-all-access")
        denied = await self.grants.denied_sources(principals_for(user))
        return decide_shared(user=user, denied_source_ids=denied)
