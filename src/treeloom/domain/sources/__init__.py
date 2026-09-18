"""Source persistence bounded context.

Domain ports and value objects for recording what repositories and branches
have been indexed by treeloom. The persistence adapter (PostgreSQL) implements
SourceRepositoryPort; the domain stays ignorant of storage details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SourceRecord:
    """A repository or directory that has been indexed by treeloom.

    Attributes:
        id: Deterministic SHA-1 hash from path/url + branch (same as Milvus source_id).
        path: Local filesystem path (empty if indexed from URL).
        url: Remote repository URL (empty if indexed from path).
        branch: Git branch name (empty string for directory indexing).
        indexed_at: When the last index job completed.
        file_count: Total files in the source.
        chunk_count: Total chunks produced.
        commit_sha: Git HEAD SHA at index time (empty for non-git inputs).
        created_by: User ID who requested the index (None for pre-auth migrations).
    """

    id: str
    path: str = ""
    url: str = ""
    branch: str = ""
    indexed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    file_count: int = 0
    chunk_count: int = 0
    commit_sha: str = ""
    created_by: Optional[str] = None
    # How the source was indexed: "repo" | "directory" | "file". Lets the
    # fleet auto-refresh loop re-enqueue the same kind of job rather
    # than blindly re-indexing everything as a repo. Defaults to "repo".
    kind: str = "repo"
    # Two-pass indexing state. TRUE means graph extraction has been
    # completed for this source; FALSE means chunks/embeddings exist but
    # graph data does not (operator hasn't run /index-graph yet).
    graph_indexed: bool = True
    graph_indexed_at: Optional[datetime] = None


class SourceRepositoryPort(ABC):
    """Persistence contract for source metadata.

    Implementations store SourceRecord in PostgreSQL, in-memory, or any
    other durable store. All methods may raise adapter-specific exceptions;
    application code should handle them generically.
    """

    @abstractmethod
    async def save(self, record: SourceRecord) -> None:
        """Insert or update a source record (upsert by id)."""
        ...

    @abstractmethod
    async def get_by_id(self, source_id: str) -> Optional[SourceRecord]:
        """Retrieve a single source by its deterministic id."""
        ...

    @abstractmethod
    async def list_all(self) -> list[SourceRecord]:
        """Return all persisted sources, most recently indexed first."""
        ...

    @abstractmethod
    async def delete(self, source_id: str) -> bool:
        """Remove a source record. Returns True if a record was deleted."""
        ...

    @abstractmethod
    async def set_graph_indexed(
        self,
        source_id: str,
        indexed: bool,
        when: Optional[datetime] = None,
    ) -> bool:
        """Set the graph-indexing state for a source. Returns True if updated."""
        ...
