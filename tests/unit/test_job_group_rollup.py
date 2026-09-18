"""Unit tests for the pure job-group rollup."""

from treeloom.application.job_groups import summarize_group
from treeloom.domain.jobs import Job, JobGroup, JobStatus


def _grp(task_count=0):
    return JobGroup(id="grp_1", label="featbit", kind="fleet",
                    created_at=1000.0, created_by="u1", task_count=task_count)


def _job(status, processed=0, total=0):
    return Job(id="j", status=status, group_id="grp_1",
               processed_files=processed, total_files=total)


def test_empty_group_is_queued_with_zero_counts():
    s = summarize_group(_grp(task_count=3), [])
    assert s["status"] == "queued"
    assert s["status_counts"] == {
        "queued": 0, "running": 0, "done": 0, "failed": 0, "dead_letter": 0,
    }
    assert s["progress"] == {"processed_files": 0, "total_files": 0}
    assert s["task_count"] == 3  # from the group row, not the (empty) task list


def test_all_done_is_done():
    s = summarize_group(_grp(2), [_job(JobStatus.DONE, 10, 10), _job(JobStatus.DONE, 5, 5)])
    assert s["status"] == "done"
    assert s["status_counts"]["done"] == 2
    assert s["progress"] == {"processed_files": 15, "total_files": 15}


def test_any_active_is_running():
    s = summarize_group(_grp(2), [_job(JobStatus.DONE, 10, 10), _job(JobStatus.RUNNING, 3, 10)])
    assert s["status"] == "running"
    assert s["status_counts"]["running"] == 1
    assert s["status_counts"]["done"] == 1


def test_queued_counts_as_active():
    s = summarize_group(_grp(1), [_job(JobStatus.QUEUED)])
    assert s["status"] == "running"


def test_failed_without_active_is_failed():
    s = summarize_group(_grp(2), [_job(JobStatus.DONE), _job(JobStatus.FAILED)])
    assert s["status"] == "failed"
    assert s["status_counts"]["failed"] == 1


def test_dead_letter_without_active_is_failed():
    s = summarize_group(_grp(1), [_job(JobStatus.DEAD_LETTER)])
    assert s["status"] == "failed"


def test_metadata_passthrough():
    s = summarize_group(_grp(5), [])
    assert s["id"] == "grp_1"
    assert s["label"] == "featbit"
    assert s["kind"] == "fleet"
    assert s["created_at"] == 1000.0
    assert s["created_by"] == "u1"
    assert "tasks" not in s  # list endpoint omits per-task bodies
