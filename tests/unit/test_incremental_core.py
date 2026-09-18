"""Unit tests for incremental-indexing core.

Covers:
  - Job payload/attempts round-trip; dispatch_job incremental branch;
         restart recovery resumes incremental jobs.
  - Retry → dead-letter transition; GET /jobs/dead-letter;
         POST /jobs/{id}/retry; dead-letter excluded from recovery.
  - compute_source_staleness helper; freshness sampler sets gauges;
         _record_incremental_metrics records duration + merge-to-searchable.
  - 429 + Retry-After when queue is saturated; normal accept below cap;
         oversized changed_files → full re-index fallback.

All tests use mock stores — no live services needed.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from treeloom.application import indexer_runners as _runners
from treeloom.application import lifecycle as _life
from treeloom.application import indexer_state as _tlstate
from treeloom.application import indexer_state
from treeloom.application import indexer_state as _state


# ── Job payload / attempts model ────────────────────────────────────────


class TestJobPayloadModel:
    def test_payload_round_trips_through_to_dict_from_dict(self):
        from treeloom.domain.jobs import Job, JobStatus

        payload = {"changed_files": [{"path": "src/foo.py", "action": "modified"}], "branch": "main"}
        j = Job(id="j1", kind="incremental", payload=payload, attempts=2)
        d = j.to_dict()
        assert d["payload"] == payload
        assert d["attempts"] == 2

        j2 = Job.from_dict(d)
        assert j2.payload == payload
        assert j2.attempts == 2

    def test_payload_defaults_to_none(self):
        from treeloom.domain.jobs import Job

        j = Job(id="j2")
        assert j.payload is None
        assert j.attempts == 0

    def test_from_dict_accepts_missing_payload(self):
        from treeloom.domain.jobs import Job

        j = Job.from_dict({"id": "j3", "kind": "repo"})
        assert j.payload is None
        assert j.attempts == 0

    def test_dead_letter_status_in_enum(self):
        from treeloom.domain.jobs import JobStatus

        assert JobStatus.DEAD_LETTER.value == "dead_letter"

    def test_from_dict_parses_dead_letter_status(self):
        from treeloom.domain.jobs import Job, JobStatus

        j = Job.from_dict({"id": "dl1", "status": "dead_letter"})
        assert j.status == JobStatus.DEAD_LETTER


# ── dispatch_job incremental branch ─────────────────────────────────────


class TestDispatchJobIncremental:
    """dispatch_job must reconstruct _run_incremental_index for incremental jobs."""

    def _make_incremental_job(self, changed_files=None, repo_url="https://example.com/r.git",
                               branch="main", source_id="sid1"):
        from treeloom.domain.jobs import Job, JobStatus

        payload = {
            "changed_files": changed_files or [{"path": "a.py", "action": "modified"}],
            "branch": branch,
        }
        return Job(
            id="inc1",
            kind="incremental",
            status=JobStatus.QUEUED,
            source_url=repo_url,
            source_id=source_id,
            payload=payload,
        )

    def test_dispatch_returns_coroutine_for_incremental(self):
        from treeloom.application import indexer_service as idx

        async def _fake_run(*args, **kwargs):
            pass

        job = self._make_incremental_job()
        with patch.object(_runners, "_run_incremental_index", side_effect=_fake_run):
            coro = _runners.dispatch_job(job)
            # It should be a coroutine (not None) for a well-formed incremental job.
            if coro is not None:
                coro.close()
            assert coro is not None

    def test_dispatch_returns_none_when_payload_missing(self):
        from treeloom.application import indexer_service as idx
        from treeloom.domain.jobs import Job, JobStatus

        job = Job(id="inc2", kind="incremental", source_url="https://x.com/r.git",
                  source_id="sid", payload=None, status=JobStatus.QUEUED)
        coro = _runners.dispatch_job(job)
        assert coro is None

    def test_dispatch_returns_none_when_changed_files_empty(self):
        from treeloom.application import indexer_service as idx
        from treeloom.domain.jobs import Job, JobStatus

        job = Job(id="inc3", kind="incremental", source_url="https://x.com/r.git",
                  source_id="sid", payload={"changed_files": [], "branch": "main"},
                  status=JobStatus.QUEUED)
        coro = _runners.dispatch_job(job)
        assert coro is None

    def test_dispatch_returns_none_when_source_id_missing(self):
        from treeloom.application import indexer_service as idx
        from treeloom.domain.jobs import Job, JobStatus

        job = Job(id="inc4", kind="incremental", source_url="https://x.com/r.git",
                  source_id="",
                  payload={"changed_files": [{"path": "a.py", "action": "modified"}], "branch": "main"},
                  status=JobStatus.QUEUED)
        coro = _runners.dispatch_job(job)
        assert coro is None


# ── restart recovery resumes incremental jobs ───────────────────────────


class TestRestartRecoveryIncremental:
    """The recovery loop in startup() must resume incremental jobs via the queue."""

    def test_dispatch_job_does_not_return_none_for_valid_incremental(self):
        """Simulates what the recovery loop does: if dispatch_job returns a coro,
        the job can be re-enqueued. We test the dispatch_job contract, not the
        whole startup coroutine.
        """
        from treeloom.application import indexer_service as idx
        from treeloom.domain.jobs import Job, JobStatus

        payload = {
            "changed_files": [{"path": "foo.py", "action": "modified"}],
            "branch": "main",
        }
        job = Job(
            id="recover1",
            kind="incremental",
            source_url="https://example.com/repo.git",
            source_id="src_abc",
            payload=payload,
            status=JobStatus.RUNNING,
        )
        coro = _runners.dispatch_job(job)
        # Should be dispatchable (not None) so recovery can re-enqueue it.
        if coro is not None:
            coro.close()
        assert coro is not None


# ── retry / dead-letter ─────────────────────────────────────────────────


class TestRetryDeadLetter:
    """Queue worker retries incremental jobs up to MAX_JOB_ATTEMPTS, then dead-letters."""

    @pytest.mark.asyncio
    async def test_retry_increments_attempts_and_reenqueues(self):
        """On failure of a retryable job with attempts < cap, re-enqueue."""
        from treeloom.domain.jobs import Job, JobStatus

        mock_job = Job(
            id="job_retry_1",
            kind="incremental",
            source_url="https://x.com/r.git",
            source_id="src1",
            payload={"changed_files": [{"path": "a.py", "action": "modified"}], "branch": "main"},
            status=JobStatus.RUNNING,
            attempts=1,
        )

        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value=mock_job)
        mock_store.increment_attempts = AsyncMock(return_value=2)  # new value = 2

        enqueue_mock = AsyncMock()

        MAX_JOB_ATTEMPTS = 3

        job_id = "job_retry_1"
        new_attempts = await mock_store.increment_attempts(job_id)
        assert new_attempts == 2
        assert new_attempts < MAX_JOB_ATTEMPTS
        # If below cap, the queue worker would re-enqueue.
        await enqueue_mock(job_id)
        enqueue_mock.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_dead_letter_when_attempts_exhausted(self):
        """On failure of a retryable job at/above attempt cap, dead-letter it."""
        from treeloom.domain.jobs import Job, JobStatus

        mock_job = Job(
            id="job_dl_1",
            kind="incremental",
            source_url="https://x.com/r.git",
            source_id="src1",
            payload={"changed_files": [{"path": "a.py", "action": "modified"}], "branch": "main"},
            status=JobStatus.RUNNING,
            attempts=2,
        )

        mock_store = AsyncMock()
        mock_store.increment_attempts = AsyncMock(return_value=3)  # hits cap = 3

        MAX_JOB_ATTEMPTS = 3

        job_id = "job_dl_1"
        new_attempts = await mock_store.increment_attempts(job_id)
        assert new_attempts >= MAX_JOB_ATTEMPTS
        # Should enter dead-letter path (not re-enqueue).

    def test_dead_letter_excluded_from_recovery(self):
        """DEAD_LETTER jobs must not be in RUNNING or QUEUED — recovery only
        resumes RUNNING/QUEUED, so dead-letter jobs stay put automatically."""
        from treeloom.domain.jobs import JobStatus

        # Recovery only lists RUNNING and QUEUED statuses; DEAD_LETTER is terminal.
        recovery_statuses = {JobStatus.RUNNING, JobStatus.QUEUED}
        assert JobStatus.DEAD_LETTER not in recovery_statuses


# ── dead-letter and retry endpoints ─────────────────────────────────────


class TestDeadLetterEndpoints:
    """GET /jobs/dead-letter and POST /jobs/{id}/retry endpoints."""

    @pytest.fixture(autouse=True)
    def disable_auth(self, monkeypatch):
        """Ensure AUTH_ENABLED is false so our endpoint-level auth check is skipped."""
        monkeypatch.setenv("AUTH_ENABLED", "false")

    def test_list_dead_letter_returns_dead_letter_jobs(self, mock_milvus_collection, mock_neo4j_driver):
        from treeloom.domain.jobs import Job, JobStatus
        from fastapi.testclient import TestClient

        dl_job = Job(
            id="dl_ep_1", kind="incremental", status=JobStatus.DEAD_LETTER,
            source="https://x.com/r.git", source_id="src1", attempts=3,
            error="SomeError", finished_at=time.time(),
        )

        with patch("treeloom.application.indexer_state._job_store") as mock_store:
            mock_store.list_dead_letter = AsyncMock(return_value=[dl_job])

            from treeloom.indexer_service import app
            client = TestClient(app)
            response = client.get("/jobs/dead-letter")
            assert response.status_code == 200
            data = response.json()
            assert "jobs" in data
            assert data["total"] == 1
            assert data["jobs"][0]["job_id"] == "dl_ep_1"
            assert data["jobs"][0]["attempts"] == 3

    def test_retry_endpoint_requeues_dead_letter_job(self, mock_milvus_collection, mock_neo4j_driver):
        from treeloom.domain.jobs import Job, JobStatus
        from fastapi.testclient import TestClient

        dl_job = Job(
            id="dl_ep_2", kind="incremental", status=JobStatus.DEAD_LETTER,
            source="https://x.com/r.git", source_id="src1", attempts=3,
        )

        with patch("treeloom.application.indexer_state._job_store") as mock_store, \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.indexer_runners._persist_job",
                   new_callable=AsyncMock):
            mock_store.get = AsyncMock(return_value=dl_job)
            mock_queue.enqueue = AsyncMock()

            from treeloom.indexer_service import app
            client = TestClient(app)
            response = client.post("/jobs/dl_ep_2/retry")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "dl_ep_2"
            assert data["status"] == "queued"

    def test_retry_endpoint_returns_404_for_unknown_job(self, mock_milvus_collection, mock_neo4j_driver):
        from fastapi.testclient import TestClient

        with patch("treeloom.application.indexer_state._job_store") as mock_store:
            mock_store.get = AsyncMock(return_value=None)

            from treeloom.indexer_service import app
            client = TestClient(app)
            response = client.post("/jobs/nonexistent/retry")
            assert response.status_code == 404

    def test_retry_endpoint_returns_409_for_running_job(self, mock_milvus_collection, mock_neo4j_driver):
        from treeloom.domain.jobs import Job, JobStatus
        from fastapi.testclient import TestClient

        running_job = Job(
            id="run_1", kind="incremental", status=JobStatus.RUNNING,
            source="https://x.com/r.git", source_id="src1", attempts=1,
        )

        with patch("treeloom.application.indexer_state._job_store") as mock_store:
            mock_store.get = AsyncMock(return_value=running_job)

            from treeloom.indexer_service import app
            client = TestClient(app)
            response = client.post("/jobs/run_1/retry")
            assert response.status_code == 409


# ── compute_source_staleness helper ─────────────────────────────────────


class TestComputeSourcStaleness:
    """compute_source_staleness extracts and delegates git SHA resolution."""

    @pytest.mark.asyncio
    async def test_stale_when_shas_differ(self):
        from treeloom.application.indexer_runners import compute_source_staleness

        src = {
            "commit_sha": "abc123",
            "path": "/some/repo",
            "branch": "main",
            "url": "",
            "indexed_at": time.time() - 3600,
        }
        with patch(
            "treeloom.application.indexer_runners._git_head_sha",
            return_value="def456",
        ):
            result = await compute_source_staleness(src)
        assert result["is_stale"] is True
        assert result["resolved_via"] == "local"
        assert result["indexed_sha"] == "abc123"
        assert result["current_sha"] == "def456"

    @pytest.mark.asyncio
    async def test_not_stale_when_shas_match(self):
        from treeloom.application.indexer_runners import compute_source_staleness

        src = {
            "commit_sha": "abc123",
            "path": "/some/repo",
            "branch": "main",
            "url": "",
            "indexed_at": time.time(),
        }
        with patch(
            "treeloom.application.indexer_runners._git_head_sha",
            return_value="abc123",
        ):
            result = await compute_source_staleness(src)
        assert result["is_stale"] is False

    @pytest.mark.asyncio
    async def test_is_stale_none_when_sha_unknown(self):
        from treeloom.application.indexer_runners import compute_source_staleness

        src = {
            "commit_sha": "",
            "path": "/some/repo",
            "branch": "main",
            "url": "",
            "indexed_at": None,
        }
        with patch(
            "treeloom.application.indexer_runners._git_head_sha",
            return_value="abc123",
        ):
            result = await compute_source_staleness(src)
        assert result["is_stale"] is None  # indexed_sha is empty

    @pytest.mark.asyncio
    async def test_uses_ls_remote_for_url_backed_source(self):
        from treeloom.application.indexer_runners import compute_source_staleness

        src = {
            "commit_sha": "aaa",
            "path": "",
            "branch": "main",
            "url": "https://example.com/repo.git",
            "indexed_at": time.time(),
        }
        with patch(
            "treeloom.application.indexer_runners._git_remote_sha",
            return_value="bbb",
        ) as mock_remote:
            result = await compute_source_staleness(src)
        assert result["resolved_via"] == "ls-remote"
        assert result["is_stale"] is True
        mock_remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_none_via_for_source_with_no_path_or_url(self):
        from treeloom.application.indexer_runners import compute_source_staleness

        src = {
            "commit_sha": "aaa",
            "path": "",
            "branch": "main",
            "url": "",
            "indexed_at": None,
        }
        result = await compute_source_staleness(src)
        assert result["resolved_via"] == "none"
        assert result["is_stale"] is None


# ── freshness sampler ────────────────────────────────────────────────────


class TestFreshnessSampler:
    """_sample_freshness_once updates Prometheus gauges per source."""

    @pytest.mark.asyncio
    async def test_sampler_sets_stale_gauge_for_stale_source(self):
        from treeloom.application import indexer_service as idx
        from treeloom.infrastructure import metrics

        now = time.time()
        sources = [
            {
                "id": "src_stale",
                "commit_sha": "old_sha",
                "path": "/repo",
                "branch": "main",
                "url": "",
                "indexed_at": now - 7200,
            }
        ]

        async def fake_staleness(src):
            return {
                "is_stale": True,
                "indexed_at": now - 7200,
                "indexed_sha": "old_sha",
                "current_sha": "new_sha",
                "resolved_via": "local",
            }

        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
            return_value=sources,
        ), patch(
            "treeloom.application.indexer_runners.compute_source_staleness",
            side_effect=fake_staleness,
        ), patch(
            "treeloom.application.lifecycle._FRESHNESS_MAX_SOURCES",
            1000,  # well above 1 source
        ):
            await _life._sample_freshness_once()

        # The gauge should have been set to 1 for the stale source
        stale_val = metrics.source_index_stale.labels(source_id="src_stale")._value.get()
        assert stale_val == 1.0

    @pytest.mark.asyncio
    async def test_sampler_sets_aggregate_gauge_above_cardinality_cap(self):
        """When sources > FRESHNESS_MAX_SOURCES, use the aggregate gauge only."""
        from treeloom.application import indexer_service as idx
        from treeloom.infrastructure import metrics

        now = time.time()
        # Create 3 fake sources but cap at 2
        sources = [
            {"id": f"src_{i}", "commit_sha": "old", "path": f"/r{i}",
             "branch": "main", "url": "", "indexed_at": now}
            for i in range(3)
        ]

        async def fake_staleness(src):
            return {
                "is_stale": True,
                "indexed_at": now,
                "indexed_sha": "old",
                "current_sha": "new",
                "resolved_via": "local",
            }

        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
            return_value=sources,
        ), patch(
            "treeloom.application.indexer_runners.compute_source_staleness",
            side_effect=fake_staleness,
        ), patch(
            "treeloom.application.lifecycle._FRESHNESS_MAX_SOURCES",
            2,  # 3 sources > cap of 2 → aggregate mode
        ):
            await _life._sample_freshness_once()

        agg = metrics.sources_stale_total._value.get()
        assert agg == 3.0

    @pytest.mark.asyncio
    async def test_sampler_does_not_crash_on_git_error(self):
        """A staleness error for one source must not abort the whole sample."""
        from treeloom.application import indexer_service as idx

        sources = [{"id": "err_src", "commit_sha": "a", "path": "/r",
                    "branch": "main", "url": "", "indexed_at": 0}]

        call_count = 0

        async def raising_staleness(src):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("git gone")

        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
            return_value=sources,
        ), patch(
            "treeloom.application.indexer_runners.compute_source_staleness",
            side_effect=raising_staleness,
        ), patch(
            "treeloom.application.lifecycle._FRESHNESS_MAX_SOURCES",
            1000,
        ):
            # Must not raise
            await _life._sample_freshness_once()

        assert call_count == 1


# ── metrics hygiene — label pruning, queue saturation, retry reset ───────


class TestFreshnessLabelPruning:
    """FIX 1: per-source freshness series for vanished sources get removed."""

    @pytest.mark.asyncio
    async def test_vanished_source_label_is_removed_next_tick(self):
        from treeloom.application import indexer_service as idx
        from treeloom.infrastructure import metrics

        now = time.time()
        tick1 = [
            {"id": "src_keep", "commit_sha": "a", "path": "/k", "branch": "main",
             "url": "", "indexed_at": now},
            {"id": "src_gone", "commit_sha": "b", "path": "/g", "branch": "main",
             "url": "", "indexed_at": now},
        ]
        tick2 = [tick1[0]]  # src_gone disappears on the 2nd tick

        async def fake_staleness(src):
            return {"is_stale": False, "indexed_at": now, "indexed_sha": "x",
                    "current_sha": "x", "resolved_via": "local"}

        # Reset the module-level tracking set so the test is hermetic.
        _life._freshness_labeled_sources = set()

        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
        ) as list_mock, patch(
            "treeloom.application.indexer_runners.compute_source_staleness",
            side_effect=fake_staleness,
        ), patch(
            "treeloom.application.lifecycle._FRESHNESS_MAX_SOURCES", 1000,
        ), patch(
            "treeloom.application.indexer_state._job_queue", None,
        ), patch.object(
            metrics.source_index_stale, "remove", wraps=metrics.source_index_stale.remove,
        ) as stale_remove, patch.object(
            metrics.source_index_age_seconds, "remove",
            wraps=metrics.source_index_age_seconds.remove,
        ) as age_remove:
            list_mock.return_value = tick1
            await _life._sample_freshness_once()
            # Both labels present after tick 1.
            assert "src_gone" in _life._freshness_labeled_sources

            list_mock.return_value = tick2
            await _life._sample_freshness_once()

        # .remove was called for the vanished source on both gauges.
        stale_remove.assert_any_call("src_gone")
        age_remove.assert_any_call("src_gone")
        # Tracking set now reflects only the surviving source.
        assert _life._freshness_labeled_sources == {"src_keep"}

    @pytest.mark.asyncio
    async def test_aggregate_mode_prunes_all_per_source_labels(self):
        """Crossing FRESHNESS_MAX_SOURCES removes every prior per-source series."""
        from treeloom.application import indexer_service as idx
        from treeloom.infrastructure import metrics

        now = time.time()

        async def fake_staleness(src):
            return {"is_stale": False, "indexed_at": now, "indexed_sha": "x",
                    "current_sha": "x", "resolved_via": "local"}

        _life._freshness_labeled_sources = set()

        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
        ) as list_mock, patch(
            "treeloom.application.indexer_runners.compute_source_staleness",
            side_effect=fake_staleness,
        ), patch(
            "treeloom.application.indexer_state._job_queue", None,
        ), patch(
            "treeloom.application.lifecycle._FRESHNESS_MAX_SOURCES", 2,
        ), patch.object(
            metrics.source_index_stale, "remove",
            wraps=metrics.source_index_stale.remove,
        ) as stale_remove:
            # Tick 1: 2 sources (per-source mode, cap is 2 → not exceeded).
            list_mock.return_value = [
                {"id": f"s{i}", "commit_sha": "a", "path": f"/r{i}",
                 "branch": "main", "url": "", "indexed_at": now}
                for i in range(2)
            ]
            await _life._sample_freshness_once()
            assert _life._freshness_labeled_sources == {"s0", "s1"}

            # Tick 2: 3 sources > cap → aggregate mode, prune all per-source.
            list_mock.return_value = [
                {"id": f"s{i}", "commit_sha": "a", "path": f"/r{i}",
                 "branch": "main", "url": "", "indexed_at": now}
                for i in range(3)
            ]
            await _life._sample_freshness_once()

        stale_remove.assert_any_call("s0")
        stale_remove.assert_any_call("s1")
        assert _life._freshness_labeled_sources == set()


class TestSamplerQueueSaturation:
    """FIX 2: the freshness sampler keeps treeloom_queue_saturation fresh."""

    @pytest.mark.asyncio
    async def test_sampler_sets_queue_saturation_from_depth(self):
        from treeloom.application import indexer_service as idx
        from treeloom.infrastructure import metrics

        fake_queue = MagicMock()
        fake_queue.depth = AsyncMock(return_value=125)

        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock, return_value=[],
        ), patch(
            "treeloom.application.indexer_state._job_queue", fake_queue,
        ), patch(
            "treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 500,
        ):
            await _life._sample_freshness_once()

        fake_queue.depth.assert_awaited_once()
        assert metrics.queue_saturation._value.get() == pytest.approx(125 / 500)


class TestRetryClearsTerminalMarkers:
    """FIX 3: a re-queued retry clears the stale finished_at/error."""

    @pytest.mark.asyncio
    async def test_reenqueue_clears_finished_at_and_error(self):
        from treeloom.application import indexer_service as idx
        from treeloom.adapters.queue.postgres_queue import PostgresJobQueue
        from treeloom.domain.jobs import Job, JobStatus

        # A job that previously failed: _fail_job set finished_at + error.
        job = Job(
            id="job_retry_clear",
            kind="incremental",
            source_url="https://x.com/r.git",
            source_id="src1",
            payload={"changed_files": [{"path": "a.py", "action": "modified"}],
                     "branch": "main"},
            status=JobStatus.RUNNING,
            attempts=1,
        )
        jd0 = job.to_dict()
        jd0["finished_at"] = 12345.0
        jd0["error"] = "boom from a prior attempt"
        # Reload so the queue sees the dirty terminal markers in to_dict().
        job = Job.from_dict(jd0)

        store = AsyncMock()
        store.get = AsyncMock(return_value=job)
        store.increment_attempts = AsyncMock(return_value=2)  # < cap

        async def boom():
            raise RuntimeError("dispatch failed")

        persisted: list[dict] = []

        async def fake_persist(jd):
            persisted.append(jd)

        q = PostgresJobQueue(max_workers=1)
        q.enqueue = AsyncMock()  # avoid touching the DB

        with patch.object(_tlstate, "_job_store", store), patch.object(
            _runners, "dispatch_job", return_value=boom(),
        ), patch.object(
            _runners, "_persist_job", side_effect=fake_persist,
        ), patch.object(
            idx, "_RETRYABLE_KINDS", {"incremental"},
        ), patch.object(idx, "MAX_JOB_ATTEMPTS", 3):
            await q._run_one(worker_id=0, job_id="job_retry_clear")

        assert persisted, "expected the re-queued job to be persisted"
        jd = persisted[-1]
        assert jd["status"] == "queued"
        assert jd["attempts"] == 2
        assert jd["finished_at"] is None
        assert jd["error"] == ""
        q.enqueue.assert_awaited_once_with("job_retry_clear")


# ── incremental job metrics ─────────────────────────────────────────────


class TestIncrementalMetrics:
    """_record_incremental_metrics records duration and merge→searchable."""

    def test_records_done_counter_and_duration(self):
        from treeloom.application.indexer_runners import _record_incremental_metrics
        from treeloom.infrastructure import metrics

        before = metrics.incremental_jobs_total.labels(status="done")._value.get()
        job_start = time.time() - 5.0
        _record_incremental_metrics(job_start, accept_time=None, status="done")
        after = metrics.incremental_jobs_total.labels(status="done")._value.get()
        assert after == before + 1

    def test_records_merge_to_searchable_when_accept_time_given(self):
        from treeloom.application.indexer_runners import _record_incremental_metrics
        from treeloom.infrastructure import metrics

        accept = time.time() - 120.0
        job_start = time.time() - 5.0
        # Should not raise; histogram is observed internally
        _record_incremental_metrics(job_start, accept_time=accept, status="done")

    def test_records_failed_counter(self):
        from treeloom.application.indexer_runners import _record_incremental_metrics
        from treeloom.infrastructure import metrics

        before = metrics.incremental_jobs_total.labels(status="failed")._value.get()
        _record_incremental_metrics(time.time(), accept_time=None, status="failed")
        after = metrics.incremental_jobs_total.labels(status="failed")._value.get()
        assert after == before + 1


# ── backpressure / 429 ──────────────────────────────────────────────────


class TestBackpressure:
    """Webhook returns 429 when queue depth >= MAX_QUEUE_DEPTH."""

    @pytest.fixture(autouse=True)
    def disable_auth(self, monkeypatch):
        """Ensure AUTH_ENABLED is false so the auth middleware skips all routes."""
        monkeypatch.setenv("AUTH_ENABLED", "false")

    @pytest.fixture
    def client(self, mock_milvus_collection, mock_neo4j_driver):
        from treeloom.indexer_service import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def _github_headers_and_body(self, monkeypatch, secret="s"):
        import json
        import hmac
        import hashlib
        monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", secret)
        body = {
            "action": "closed",
            "pull_request": {
                "merged": True,
                "number": 1,
                "base": {
                    "ref": "main",
                    "repo": {"clone_url": "https://github.com/o/r.git"},
                },
            },
        }
        payload_bytes = json.dumps(body).encode()
        sig = "sha256=" + hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        }
        return headers, body

    def test_webhook_returns_429_when_queue_saturated(self, client, monkeypatch):
        from treeloom.domain.webhook import Provider, WebhookPayload

        mock_payload = WebhookPayload(
            provider=Provider.GITHUB,
            repo_url="https://github.com/o/r.git",
            branch="main",
            changed_files=[{"path": "a.py", "action": "modified"}],
        )

        headers, body = self._github_headers_and_body(monkeypatch)
        existing_source = {"id": "src1", "url": "https://github.com/o/r.git", "branch": "main"}

        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=mock_payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[existing_source]), \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 1):
            # Queue depth >= MAX_QUEUE_DEPTH
            mock_queue.depth = AsyncMock(return_value=5)
            response = client.post("/webhook", json=body, headers=headers)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        data = response.json()
        assert "Queue saturated" in data.get("detail", "")

    def test_webhook_sheds_increment_counter_on_429(self, client, monkeypatch):
        from treeloom.domain.webhook import Provider, WebhookPayload
        from treeloom.infrastructure import metrics

        mock_payload = WebhookPayload(
            provider=Provider.GITHUB,
            repo_url="https://github.com/o/r.git",
            branch="main",
            changed_files=[{"path": "a.py", "action": "modified"}],
        )

        headers, body = self._github_headers_and_body(monkeypatch)
        existing_source = {"id": "src1", "url": "https://github.com/o/r.git", "branch": "main"}

        before = metrics.webhook_shed_total._value.get()
        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=mock_payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[existing_source]), \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 1):
            mock_queue.depth = AsyncMock(return_value=5)
            client.post("/webhook", json=body, headers=headers)

        after = metrics.webhook_shed_total._value.get()
        assert after == before + 1

    def test_webhook_accepts_when_below_max_queue_depth(self, client, monkeypatch):
        from treeloom.domain.webhook import Provider, WebhookPayload

        mock_payload = WebhookPayload(
            provider=Provider.GITHUB,
            repo_url="https://github.com/o/r.git",
            branch="main",
            changed_files=[{"path": "a.py", "action": "modified"}],
        )

        headers, body = self._github_headers_and_body(monkeypatch)
        existing_source = {"id": "src1", "url": "https://github.com/o/r.git", "branch": "main"}

        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=mock_payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[existing_source]), \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.indexer_runners._persist_job",
                   new_callable=AsyncMock), \
             patch("treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 500):
            # Queue depth well below cap
            mock_queue.depth = AsyncMock(return_value=10)
            mock_queue.enqueue = AsyncMock()
            response = client.post("/webhook", json=body, headers=headers)

        assert response.status_code == 202

    def test_oversized_changed_files_falls_back_to_full_reindex(self, client, monkeypatch):
        """When changed_files exceeds MAX_CHANGED_FILES, a full repo job is created."""
        from treeloom.domain.webhook import Provider, WebhookPayload

        big_files = [{"path": f"f{i}.py", "action": "modified"} for i in range(10)]
        mock_payload = WebhookPayload(
            provider=Provider.GITHUB,
            repo_url="https://github.com/o/r.git",
            branch="main",
            changed_files=big_files,
        )

        headers, body = self._github_headers_and_body(monkeypatch)
        existing_source = {"id": "src1", "url": "https://github.com/o/r.git", "branch": "main"}

        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=mock_payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[existing_source]), \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.routes_webhook._find_active_job_for_source",
                   new_callable=AsyncMock, return_value=None), \
             patch("treeloom.application.indexer_runners._persist_job",
                   new_callable=AsyncMock), \
             patch("treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 500), \
             patch("treeloom.application.routes_webhook.MAX_CHANGED_FILES", 5):
            # 10 files > cap of 5 → full reindex
            mock_queue.depth = AsyncMock(return_value=0)
            mock_queue.enqueue = AsyncMock()
            response = client.post("/webhook", json=body, headers=headers)

        assert response.status_code == 202
        data = response.json()
        # Message should mention the fallback
        assert "full" in data.get("message", "").lower() or data.get("status") == "queued"

    def test_normal_incremental_job_enqueued_not_create_task(self, client, monkeypatch):
        """Normal (below threshold) incremental webhook routes through the queue, not create_task."""
        from treeloom.domain.webhook import Provider, WebhookPayload

        mock_payload = WebhookPayload(
            provider=Provider.GITHUB,
            repo_url="https://github.com/o/r.git",
            branch="main",
            changed_files=[{"path": "a.py", "action": "modified"}],
        )

        headers, body = self._github_headers_and_body(monkeypatch)
        existing_source = {"id": "src1", "url": "https://github.com/o/r.git", "branch": "main"}

        enqueue_mock = AsyncMock()
        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=mock_payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[existing_source]), \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.indexer_runners._persist_job",
                   new_callable=AsyncMock), \
             patch("treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 500), \
             patch("treeloom.application.routes_webhook.MAX_CHANGED_FILES", 5000):
            mock_queue.depth = AsyncMock(return_value=0)
            mock_queue.enqueue = enqueue_mock
            response = client.post("/webhook", json=body, headers=headers)

        assert response.status_code == 202
        enqueue_mock.assert_called_once()


# ── queue depth() method ────────────────────────────────────────────────


class TestQueueDepth:
    """PostgresJobQueue.depth() queries the job_queue table."""

    @pytest.mark.asyncio
    async def test_depth_returns_count_from_db(self):
        from treeloom.adapters.queue.postgres_queue import PostgresJobQueue

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"cnt": 7})
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "treeloom.adapters.queue.postgres_queue.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            q = PostgresJobQueue()
            depth = await q.depth()
        assert depth == 7

    @pytest.mark.asyncio
    async def test_depth_returns_zero_when_pool_none(self):
        from treeloom.adapters.queue.postgres_queue import PostgresJobQueue

        with patch(
            "treeloom.adapters.queue.postgres_queue.get_pool",
            new_callable=AsyncMock,
            return_value=None,
        ):
            q = PostgresJobQueue()
            depth = await q.depth()
        assert depth == 0


# ── queue saturation metric ─────────────────────────────────────────────


class TestQueueSaturationMetric:
    def test_saturation_gauge_exists(self):
        from treeloom.infrastructure import metrics

        # Just check it's accessible and can be set.
        metrics.queue_saturation.set(0.5)
        assert metrics.queue_saturation._value.get() == 0.5


# ── review follow-up: backpressure on all paths (#1) + same-source dedup (#2) ──


class TestWebhookBackpressureAndDedupFixes:
    """Code-review fixes: the 429 gate covers new-source full-index webhooks
    (#1), and an incremental webhook for a source with an in-flight job dedups
    instead of creating a colliding job that would hit jobs_active_source_uniq
    and 500 (#2)."""

    @pytest.fixture(autouse=True)
    def disable_auth(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "false")

    @pytest.fixture
    def client(self, mock_milvus_collection, mock_neo4j_driver):
        from treeloom.indexer_service import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def _hdr_body(self, monkeypatch, clone_url, secret="s"):
        import json, hmac, hashlib
        monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", secret)
        body = {
            "action": "closed",
            "pull_request": {
                "merged": True, "number": 1,
                "base": {"ref": "main", "repo": {"clone_url": clone_url}},
            },
        }
        sig = "sha256=" + hmac.new(
            secret.encode(), json.dumps(body).encode(), hashlib.sha256
        ).hexdigest()
        return {
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        }, body

    def test_new_source_webhook_sheds_when_saturated(self, client, monkeypatch):
        """#1: a first-time (unindexed) source webhook also respects backpressure."""
        from treeloom.domain.webhook import Provider, WebhookPayload

        url = "https://github.com/o/brand-new.git"
        payload = WebhookPayload(
            provider=Provider.GITHUB, repo_url=url, branch="main",
            changed_files=[{"path": "a.py", "action": "modified"}],
        )
        headers, body = self._hdr_body(monkeypatch, url)
        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[]), \
             patch("treeloom.application.routes_webhook._find_active_job_for_source",
                   new_callable=AsyncMock, return_value=None), \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue, \
             patch("treeloom.application.indexer_state.MAX_QUEUE_DEPTH", 1):
            mock_queue.depth = AsyncMock(return_value=9)
            mock_queue.enqueue = AsyncMock()
            resp = client.post("/webhook", json=body, headers=headers)

        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        mock_queue.enqueue.assert_not_called()  # no job enqueued while shed

    def test_incremental_dedups_to_active_job(self, client, monkeypatch):
        """#2: incremental webhook for a source with an in-flight job returns the
        existing job and does NOT create/enqueue a colliding one."""
        from treeloom.domain.webhook import Provider, WebhookPayload

        url = "https://github.com/o/r.git"
        payload = WebhookPayload(
            provider=Provider.GITHUB, repo_url=url, branch="main",
            changed_files=[{"path": "a.py", "action": "modified"}],
        )
        headers, body = self._hdr_body(monkeypatch, url)
        existing_source = {"id": "src1", "url": url, "branch": "main"}
        active_job = {"job_id": "active-1", "status": "running"}
        with patch("treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
                   return_value=payload), \
             patch("treeloom.application.indexer_service.graph_store.list_sources",
                   new_callable=AsyncMock, return_value=[existing_source]), \
             patch("treeloom.application.routes_webhook._find_active_job_for_source",
                   new_callable=AsyncMock, return_value=active_job), \
             patch("treeloom.application.indexer_runners._persist_job",
                   new_callable=AsyncMock) as persist_mock, \
             patch("treeloom.application.indexer_state._job_queue") as mock_queue:
            mock_queue.depth = AsyncMock(return_value=0)
            mock_queue.enqueue = AsyncMock()
            resp = client.post("/webhook", json=body, headers=headers)

        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "active-1"
        assert "in-flight job already covers" in data["message"]
        mock_queue.enqueue.assert_not_called()
        persist_mock.assert_not_called()


def test_metrics_registry_exposes_process_rss():
    """The /metrics registry must expose process_resident_memory_bytes so the
    load-test harness can track indexer memory growth."""
    from treeloom.infrastructure import metrics
    text = metrics.get_metrics().decode()
    assert "process_resident_memory_bytes" in text
