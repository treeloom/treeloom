"""Job domain model — pure data, no I/O dependencies."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Job:
    id: str
    source: str = ""
    source_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    kind: str = "repo"
    total_files: int = 0
    processed_files: int = 0
    committed_files: int = 0
    total_chunks: int = 0
    start_time: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str = ""
    message: str = ""
    current_file: str = ""
    source_path: str = ""
    source_url: str = ""
    source_branch: str = ""
    commit_sha: str = ""
    errors: int = 0
    payload: dict | None = None
    attempts: int = 0
    group_id: str | None = None  # parent job_group ("Job" in the UI); see 020 migration

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "status": self.status.value,
            "kind": self.kind,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "committed_files": self.committed_files,
            "total_chunks": self.total_chunks,
            "start_time": self.start_time,
            "finished_at": self.finished_at,
            "error": self.error,
            "message": self.message,
            "current_file": self.current_file,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "source_branch": self.source_branch,
            "commit_sha": self.commit_sha,
            "errors": self.errors,
            "payload": self.payload,
            "attempts": self.attempts,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(
            id=d.get("id") or d.get("job_id", ""),
            source=d.get("source", ""),
            source_id=d.get("source_id", ""),
            status=JobStatus(d.get("status", "queued")),
            kind=d.get("kind") or "repo",
            total_files=d.get("total_files", 0),
            processed_files=d.get("processed_files", 0),
            committed_files=d.get("committed_files", 0),
            total_chunks=d.get("total_chunks", 0),
            start_time=d.get("start_time", 0.0) or time.time(),
            finished_at=d.get("finished_at"),
            error=d.get("error", "") or "",
            message=d.get("message", ""),
            current_file=d.get("current_file", ""),
            source_path=d.get("source_path", ""),
            source_url=d.get("source_url", ""),
            source_branch=d.get("source_branch", ""),
            commit_sha=d.get("commit_sha", ""),
            errors=d.get("errors", 0),
            payload=d.get("payload"),
            attempts=d.get("attempts", 0),
            group_id=d.get("group_id"),
        )


@dataclass
class JobGroup:
    """A submission of 1..N index Tasks — called a "Job" in the operator UI.

    Pure data; persistence lives in adapters/postgresql/job_group_store.py and
    the schema in migration 020_job_groups.sql. Each row in the `jobs` table
    (a "Task") links to its parent group via `jobs.group_id`. The physical
    `jobs` table is deliberately NOT renamed.
    """

    id: str
    label: str
    kind: str = "repo"          # fleet | repo | directory | file | webhook | graph
    created_at: float = field(default_factory=time.time)
    created_by: str | None = None
    task_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "task_count": self.task_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JobGroup:
        return cls(
            id=d.get("id", ""),
            label=d.get("label", ""),
            kind=d.get("kind") or "repo",
            created_at=d.get("created_at", 0.0) or time.time(),
            created_by=d.get("created_by"),
            task_count=d.get("task_count", 0),
        )


import abc
import asyncio


class JobQueue(abc.ABC):
    """Abstract job queue — enqueue persisted job_ids for background processing.

    Implementations are expected to be safe for cross-process pickup: the
    queue carries only job_ids, and workers look up the actual Job state
    from a shared JobStore. Coroutine reconstruction happens worker-side
    via a dispatcher keyed on `Job.kind`.
    """

    @abc.abstractmethod
    async def enqueue(self, job_id: str) -> None:
        """Add a persisted job_id to the queue."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Start worker tasks pulling from the queue."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop worker tasks gracefully."""

    @abc.abstractmethod
    def pending_count(self) -> int:
        """Number of jobs waiting in the queue (may be a local-view snapshot)."""

    @abc.abstractmethod
    def active_count(self) -> int:
        """Number of jobs currently being processed by this process."""

    async def depth(self) -> int:
        """Return the number of jobs currently in the queue (cross-process).

        Default implementation returns 0 — override for accurate counts.
        """
        return 0
