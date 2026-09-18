"""Search audit domain.

A who-searched-what trail, distinct from the LLM-operation audit in
``domain/audit.py`` (which has no user/resource concept). One record per
authorization decision on a read endpoint.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class SearchAuditRecord:
    """One authorization decision on a read endpoint."""

    user_id: Optional[str]
    action: str           # search | graph_explore | find_definition | ...
    scope: str            # 'source' | 'shared'
    decision: str         # 'allow' | 'deny'
    source_id: Optional[str] = None
    principal_groups: tuple[str, ...] = ()
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "action": self.action,
            "scope": self.scope,
            "decision": self.decision,
            "source_id": self.source_id,
            "principal_groups": list(self.principal_groups),
            "occurred_at": self.occurred_at.isoformat(),
        }


class SearchAuditPort(abc.ABC):
    """Persistence contract for the search audit trail."""

    @abc.abstractmethod
    async def record(self, entry: SearchAuditRecord) -> None:
        """Append one decision. Best-effort — never raises to the caller."""
        ...

    @abc.abstractmethod
    async def recent(
        self,
        user_id: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent decisions, optionally filtered by user or source."""
        ...
