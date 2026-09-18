"""— index jobs must finalize `failed`, not `done`, when every file
errored and produced zero chunks (otherwise a fully-broken run — e.g. an
unreachable embedding/vector backend — reads as success at a glance)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from treeloom.application import indexer_runners as svc
from treeloom.application import indexer_state as _state


# ── pure status resolver ────────────────────────────────────────────────────

class TestTerminalStatusResolver:
    def test_all_files_errored_zero_chunks_is_failed(self):
        assert svc._terminal_status_for_index(0, 3, 3) == "failed"

    def test_partial_success_with_chunks_stays_done(self):
        # some files errored but chunks landed → not a total failure
        assert svc._terminal_status_for_index(5, 3, 3) == "done"

    def test_some_but_not_all_errored_stays_done(self):
        # conservative: only flip when *every* attempted unit failed
        assert svc._terminal_status_for_index(0, 3, 1) == "done"

    def test_no_op_no_units_stays_done(self):
        assert svc._terminal_status_for_index(0, 0, 0) == "done"

    def test_zero_errors_stays_done(self):
        # empty files that yield no chunks but raise nothing
        assert svc._terminal_status_for_index(0, 3, 0) == "done"

    def test_error_overcount_still_failed(self):
        # batch-flush failures can push errors >= attempted; >= guards it
        assert svc._terminal_status_for_index(0, 2, 5) == "failed"


# ── incremental finalize ────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFinalizeIncrementalJob:
    async def test_all_changed_files_failed_marks_failed(self):
        job = {"job_id": "j1", "errors": 2, "files_removed": 0}
        with patch.object(svc, "_persist_job", new=AsyncMock()) as pj:
            await svc._finalize_incremental_job(job, 0, 2, "msg")
        assert job["status"] == "failed"
        assert "0 chunks" in job["error"]
        assert "/jobs/j1/errors" in job["error"]
        pj.assert_awaited_once()

    async def test_removal_only_delta_stays_done(self):
        # 3 files processed, all removals (successful no-ops), no errors
        job = {"job_id": "j2", "errors": 0, "files_removed": 3}
        with patch.object(svc, "_persist_job", new=AsyncMock()):
            await svc._finalize_incremental_job(job, 0, 3, "msg")
        assert job["status"] == "done"
        assert "error" not in job

    async def test_partial_success_stays_done(self):
        # 2 changed files, one errored, but a chunk landed
        job = {"job_id": "j3", "errors": 1, "files_removed": 0}
        with patch.object(svc, "_persist_job", new=AsyncMock()):
            await svc._finalize_incremental_job(job, 4, 2, "msg")
        assert job["status"] == "done"


# ── full-index finalize ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFinalizeJob:
    async def test_all_failed_marks_failed_and_skips_source_upsert(self):
        job = {
            "job_id": "j4", "source_id": "s4", "errors": 3,
            "source_path": "/x", "source_url": "", "source_branch": "",
        }
        with patch.object(svc, "_persist_job", new=AsyncMock()) as pj, \
             patch.object(svc, "_persist_job_sync") as pjs, \
             patch.object(svc.graph_store, "upsert_source", new=AsyncMock()) as up, \
             patch.object(_state._source_repo, "save", new=AsyncMock()) as save:
            await svc._finalize_job(job, 0, 3, "all failed")
        assert job["status"] == "failed"
        assert "0 chunks" in job["error"]
        # a fully-broken run must NOT be recorded as an indexed source
        up.assert_not_awaited()
        save.assert_not_awaited()
        pjs.assert_not_called()
        pj.assert_awaited_once()

    async def test_success_marks_done_and_upserts_source(self):
        job = {
            "job_id": "j5", "source_id": "s5", "errors": 0,
            "source_path": "/x", "source_url": "", "source_branch": "",
        }
        with patch.object(svc, "_persist_job", new=AsyncMock()), \
             patch.object(svc, "_persist_job_sync"), \
             patch.object(svc.graph_store, "upsert_source", new=AsyncMock()) as up, \
             patch.object(_state._source_repo, "save", new=AsyncMock()) as save:
            await svc._finalize_job(job, 7, 3, "ok")
        assert job["status"] == "done"
        up.assert_awaited_once()
        save.assert_awaited_once()
