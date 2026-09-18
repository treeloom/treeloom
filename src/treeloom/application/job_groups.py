"""Pure rollup of a job_group ("Job") and its Tasks for the operator UI v2.

Aggregates a JobGroup plus its member jobs (Tasks) into the summary the Jobs
feed renders: per-status counts, summed file progress, and a single derived
status. No I/O -- unit-tested in tests/unit/test_job_group_rollup.py.
"""

from __future__ import annotations

from treeloom.domain.jobs import Job, JobGroup, JobStatus

_STATUS_KEYS = [s.value for s in JobStatus]  # queued, running, done, failed, dead_letter


def _derive_status(counts: dict[str, int], total: int) -> str:
    """Single status for the whole group.

    running if any Task is active (running/queued); else failed if any Task
    failed/dead-lettered; else done if there are Tasks (all done); else queued
    (a group with no Tasks yet).
    """
    if counts["running"] + counts["queued"] > 0:
        return "running"
    if counts["failed"] + counts["dead_letter"] > 0:
        return "failed"
    if total > 0:
        return "done"
    return "queued"


def summarize_group(group: JobGroup, tasks: list[Job]) -> dict:
    """Build the Jobs-feed summary dict for *group* given its member *tasks*.

    `task_count` reflects the group's recorded submission size; the derived
    counts/progress/status come from the actual Tasks supplied.
    """
    counts = {k: 0 for k in _STATUS_KEYS}
    processed = 0
    total_files = 0
    for t in tasks:
        counts[t.status.value] = counts.get(t.status.value, 0) + 1
        processed += t.processed_files
        total_files += t.total_files
    return {
        "id": group.id,
        "label": group.label,
        "kind": group.kind,
        "created_at": group.created_at,
        "created_by": group.created_by,
        "task_count": group.task_count,
        "status_counts": counts,
        "progress": {"processed_files": processed, "total_files": total_files},
        "status": _derive_status(counts, len(tasks)),
    }
