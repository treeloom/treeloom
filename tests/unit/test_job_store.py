"""Tests for Job domain model.

The JobStore adapter itself is now Postgres-backed (see
src/treeloom/adapters/postgresql/job_store.py) and needs a live database
fixture to test, so storage-level tests are covered by integration runs
rather than this unit module.
"""
import pytest
from treeloom.domain.jobs import Job, JobStatus


class TestJobSerialization:
    def test_job_to_dict(self):
        j = Job(id="abc-123", source="/tmp/repo", source_id="src-id",
                status=JobStatus.QUEUED, total_files=10)
        d = j.to_dict()
        assert d["id"] == "abc-123"
        assert d["status"] == "queued"
        assert d["total_files"] == 10

    def test_job_from_dict(self):
        d = {"id": "xyz-456", "source": "/tmp/other", "status": "running",
             "total_files": 5, "processed_files": 3, "total_chunks": 12,
             "errors": 1}
        j = Job.from_dict(d)
        assert j.id == "xyz-456"
        assert j.status == JobStatus.RUNNING
        assert j.total_files == 5
