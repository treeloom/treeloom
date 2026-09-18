"""Guards for the two bugs that made every queued `repo` job hang forever.

1. `dispatch_job` used `IndexRepoRequest` without importing it, so the `repo`
   branch -- the main indexing path -- raised NameError at runtime. The other
   four kinds were unaffected, so nothing else surfaced it.
2. `PostgresJobQueue._worker` wrapped `_run_one` in a try that caught only
   `CancelledError`. The NameError therefore escaped the `while` loop and
   killed that worker permanently, with no log line and no `failed` status:
   the job simply sat at `queued` while the queue row was already gone.

Together those made a total indexing outage look like an idle queue.
"""

import asyncio
import inspect
import logging

import pytest

from treeloom.application import indexer_runners
from treeloom.adapters.queue.postgres_queue import PostgresJobQueue


class _Job:
    """Minimal stand-in for the Job row dispatch_job consumes."""
    def __init__(self, kind, **fields):
        self.id = "job-1"
        self.kind = kind
        self.payload = fields.pop("payload", None)
        self._d = {"job_id": "job-1", "source_id": "src-1", **fields}

    def to_dict(self):
        return dict(self._d)


class TestDispatchJobRepo:
    def test_repo_job_dispatches_without_nameerror(self, tmp_path):
        """The regression: this raised NameError: name 'IndexRepoRequest'."""
        coro = indexer_runners.dispatch_job(
            _Job("repo", source_path=str(tmp_path), source_url="", source_branch="")
        )
        assert coro is not None, "repo job produced no runner coroutine"
        assert inspect.iscoroutine(coro)
        coro.close()

    def test_repo_job_by_url_dispatches(self):
        coro = indexer_runners.dispatch_job(
            _Job("repo", source_path="", source_url="https://example.com/r.git",
                 source_branch="main")
        )
        assert coro is not None
        coro.close()

    def test_repo_job_with_neither_path_nor_url_is_undispatchable(self):
        """Must return None (-> caller marks it failed), not raise."""
        assert indexer_runners.dispatch_job(
            _Job("repo", source_path="", source_url="", source_branch="")
        ) is None


class TestWorkerSurvivesPoisonJob:
    def test_worker_logs_and_continues_when_run_one_raises(self, caplog):
        """A raising _run_one must not kill the worker loop."""
        q = PostgresJobQueue()
        q._running = True
        q._poll_interval = 0

        claims = ["poison-job", None]
        async def fake_claim():
            if claims:
                v = claims.pop(0)
                if v is None:
                    q._running = False   # stop after the second pass
                return v
            q._running = False
            return None

        calls = []
        async def fake_run_one(worker_id, job_id):
            calls.append(job_id)
            raise NameError("name 'IndexRepoRequest' is not defined")

        q._claim_one = fake_claim
        q._run_one = fake_run_one

        with caplog.at_level(logging.ERROR):
            asyncio.run(q._worker(0))

        assert calls == ["poison-job"], "worker died instead of continuing"
        assert any("poison-job" in r.message or "poison-job" in r.getMessage()
                   for r in caplog.records), "the failure was not logged"
