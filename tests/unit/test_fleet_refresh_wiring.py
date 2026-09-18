"""The fleet auto-refresh tick must actually run, not merely import.

`_fleet_refresh_once` referenced four names the module never defined --
_all_source_dicts, select_sources_to_refresh, _find_active_job_for_source and
_enqueue_source_reindex -- left behind by the same indexer_service split that
broke repo indexing. Nothing caught it because the loop is off by default
(FLEET_AUTO_REFRESH_ENABLED=false), so the first NameError would have landed
on whoever turned it on in production.

Import-time linting is not enough here: the names are resolved inside the
function body, so only executing it proves the wiring.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from treeloom.application import lifecycle


class TestFleetRefreshTickExecutes:
    def test_returns_zero_without_a_queue_rather_than_raising(self):
        with patch.object(lifecycle._state, "_job_queue", None):
            assert asyncio.run(lifecycle._fleet_refresh_once()) == 0

    def test_full_tick_enqueues_a_stale_source(self):
        """Exercises every previously-undefined name in one pass."""
        src = {"id": "s1", "path": "/repo", "url": "https://example.com/r.git"}

        with patch.object(lifecycle._state, "_job_queue", object()), \
             patch.object(lifecycle._state, "_job_store", object()), \
             patch("treeloom.application.indexer_service._all_source_dicts",
                   new=AsyncMock(return_value=[src])), \
             patch("treeloom.application.indexer_runners.compute_source_staleness",
                   new=AsyncMock(return_value={"is_stale": True})), \
             patch("treeloom.application.routes_webhook._find_active_job_for_source",
                   new=AsyncMock(return_value=None)), \
             patch("treeloom.application.indexer_service._enqueue_source_reindex",
                   new=AsyncMock(return_value="job-1")) as enq:
            n = asyncio.run(lifecycle._fleet_refresh_once())

        assert n == 1, "the tick did not enqueue the stale source"
        enq.assert_awaited_once()

    def test_source_with_an_active_job_is_skipped(self):
        src = {"id": "s1", "path": "/repo", "url": "https://example.com/r.git"}
        with patch.object(lifecycle._state, "_job_queue", object()), \
             patch.object(lifecycle._state, "_job_store", object()), \
             patch("treeloom.application.indexer_service._all_source_dicts",
                   new=AsyncMock(return_value=[src])), \
             patch("treeloom.application.indexer_runners.compute_source_staleness",
                   new=AsyncMock(return_value={"is_stale": True})), \
             patch("treeloom.application.routes_webhook._find_active_job_for_source",
                   new=AsyncMock(return_value={"id": "running"})), \
             patch("treeloom.application.indexer_service._enqueue_source_reindex",
                   new=AsyncMock()) as enq:
            n = asyncio.run(lifecycle._fleet_refresh_once())
        assert n == 0
        enq.assert_not_awaited()
