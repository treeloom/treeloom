"""The LOW tier: probes, key rotation, unbounded listing, and scope narrowing.

Each is small, and each had the same character — a value was configured or
computed and then never actually consulted.
"""

import pytest


class TestHalfOpenProbeCap:
    """finding 53 CWE-362. half_open_max_attempts was configured (default 1) and
    reset in two places, but never incremented or checked. HALF_OPEN therefore
    admitted unlimited CONCURRENT probes: every caller arriving in the
    recovery window went through to a provider already known to be failing,
    which is the thundering herd the breaker exists to prevent.
    """

    def _breaker(self, **cfg):
        from treeloom.domain.circuit_breaker import CircuitBreaker, CircuitConfig

        return CircuitBreaker(CircuitConfig(**cfg))

    def _trip(self, b, n):
        for _ in range(n):
            b.record_failure()

    def test_half_open_admits_only_the_configured_number(self, mocker):
        import time

        from treeloom.domain.circuit_breaker import CircuitState

        b = self._breaker(failure_threshold=2, recovery_seconds=0.0,
                          half_open_max_attempts=1)
        self._trip(b, 2)
        assert b.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)
        time.sleep(0.01)

        assert b.is_open() is False, "the single probe is admitted"
        assert b.is_open() is True, "a concurrent second probe is refused"
        assert b.is_open() is True

    def test_a_larger_cap_admits_that_many(self):
        import time

        b = self._breaker(failure_threshold=2, recovery_seconds=0.0,
                          half_open_max_attempts=3)
        self._trip(b, 2)
        time.sleep(0.01)
        assert [b.is_open() for _ in range(4)] == [False, False, False, True]

    def test_a_successful_probe_closes_and_resets_the_budget(self):
        import time

        from treeloom.domain.circuit_breaker import CircuitState

        b = self._breaker(failure_threshold=2, recovery_seconds=0.0,
                          half_open_max_attempts=1)
        self._trip(b, 2)
        time.sleep(0.01)
        assert b.is_open() is False
        b.record_success()
        assert b.state == CircuitState.CLOSED
        assert b.is_open() is False, "closed circuits admit everything again"

    def test_a_failed_probe_reopens(self):
        import time

        from treeloom.domain.circuit_breaker import CircuitState

        b = self._breaker(failure_threshold=2, recovery_seconds=30.0,
                          half_open_max_attempts=1)
        self._trip(b, 2)
        b._last_failure_time -= 31  # elapse the recovery window
        assert b.is_open() is False
        b.record_failure()
        assert b.state == CircuitState.OPEN
        assert b.is_open() is True

    def test_closed_circuit_is_unaffected(self):
        b = self._breaker(failure_threshold=5)
        assert all(b.is_open() is False for _ in range(20))


class TestRerankClientKeyRotation:
    """finding 52 CWE-672. The key was baked into the singleton client's headers on
    first use, so rotating it had no effect until restart — including when it
    was rotated BECAUSE it leaked, which is the case that matters."""

    @pytest.fixture(autouse=True)
    def reset(self):
        from treeloom.adapters.rerank import cloud_reranker as cr

        cr._client = None
        cr._client_key = None
        yield
        cr._client = None
        cr._client_key = None

    def test_client_is_rebuilt_when_the_key_changes(self, monkeypatch):
        from treeloom.adapters.rerank import cloud_reranker as cr

        monkeypatch.setattr(cr, "_api_key", lambda: "key-one")
        first = cr._get_client()
        assert cr._get_client() is first, "stable while the key is unchanged"

        monkeypatch.setattr(cr, "_api_key", lambda: "key-two")
        second = cr._get_client()
        assert second is not first

    def test_the_new_client_carries_the_new_key(self, monkeypatch):
        from treeloom.adapters.rerank import cloud_reranker as cr

        monkeypatch.setattr(cr, "_api_key", lambda: "key-one")
        cr._get_client()
        monkeypatch.setattr(cr, "_api_key", lambda: "key-two")
        client = cr._get_client()
        assert "key-two" in str(client.headers.get("Authorization", "")) or \
               "key-two" in str(dict(client.headers))

    def test_a_missing_key_raises_rather_than_silently_reusing(self, monkeypatch):
        from treeloom.adapters.rerank import cloud_reranker as cr

        # Needs a cloud provider selected, or _provider_config raises first
        # for an unrelated reason (the default RERANKER_PROVIDER is `http`).
        monkeypatch.setattr(cr, "RERANKER_PROVIDER", "cohere")
        monkeypatch.setattr(cr, "_api_key", lambda: "")
        with pytest.raises(RuntimeError, match="needs an API key"):
            cr._get_client()

    def test_an_unset_key_does_not_discard_a_working_client(self, monkeypatch):
        """Rebuild only on a CHANGE to a real key. Treating "" as a change
        would tear down a working client whenever the env was read back empty."""
        from treeloom.adapters.rerank import cloud_reranker as cr

        monkeypatch.setattr(cr, "_api_key", lambda: "key-one")
        first = cr._get_client()
        monkeypatch.setattr(cr, "_api_key", lambda: "")
        assert cr._get_client() is first


class TestSourceListingIsBounded:
    """finding 55 CWE-400. SELECT * with no LIMIT on a request path."""

    def test_a_cap_exists_and_is_generous(self):
        from treeloom.adapters.sources.repository import SOURCE_LIST_MAX

        assert SOURCE_LIST_MAX >= 1000, (
            "the fleet docs target hundreds of sources; the cap must not "
            "truncate a real install"
        )

    def test_the_query_carries_a_limit(self):
        import inspect

        from treeloom.adapters.sources.repository import PostgreSourceRepository

        src = inspect.getsource(PostgreSourceRepository.list_all)
        assert "LIMIT" in src
