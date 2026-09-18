"""Graph-domain rules shared by the Neo4j and SQLite backends.

This package used to be shadowed by a sibling `domain/graph.py` defining a
`GraphStorePort` protocol. The package won the import, so that file was
unreachable and nothing referenced the protocol; it has since been deleted.
"""

from __future__ import annotations

import re

# A Cypher relationship type must be a bare identifier. It is the one part of
# a MERGE that CANNOT be a query parameter — Neo4j provides no placeholder for
# a relationship type — so it is interpolated into the query string. Anything
# that is not an identifier is therefore either a syntax error or injection.
#
# The rule lives in the domain because both graph backends must agree on what
# a relationship type is. SQLite binds it as a column value and would happily
# store `FOO]->(x) DETACH DELETE x //`, producing a graph that cannot exist in
# Neo4j — so the same source indexed on two backends would differ. SQLite also
# cannot import this from the Neo4j adapter: that module require_env's its
# service vars at import and would crash simple mode.
_VALID_REL_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_rel_type(t: object) -> bool:
    """True when *t* is usable as a Cypher relationship type."""
    return isinstance(t, str) and bool(_VALID_REL_TYPE.match(t))


def require_valid_rel_type(t: object) -> str:
    """Return *t*, or raise ValueError if it is not a valid relationship type.

    Raises rather than skipping, unlike `store_graph`. That function processes
    a batch, where dropping one malformed edge with a warning and continuing is
    the right call. A single-edge upsert that quietly did nothing would report
    success for an edge that was never written.
    """
    if not is_valid_rel_type(t):
        raise ValueError(
            f"invalid relationship type {t!r}: must match "
            f"{_VALID_REL_TYPE.pattern}. It is interpolated into Cypher, "
            f"where relationship types cannot be parameterised."
        )
    return t  # type: ignore[return-value]
