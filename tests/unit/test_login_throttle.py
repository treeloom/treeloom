"""Unit tests for login rate-limiting:

  * the pure limiter (`domain/authorization/login_throttle.py`)
  * the asyncpg-backed store (`adapters/postgresql/login_attempt_store.py`)
    against an in-memory fake pool — no live Postgres needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from treeloom.domain.authorization import login_throttle
from treeloom.adapters.postgresql import login_attempt_store as store_mod
from treeloom.adapters.postgresql.login_attempt_store import LoginAttemptStore


# ── Pure limiter ──────────────────────────────────────────────────────────


class TestIsLocked:
    def test_below_threshold_not_locked(self):
        assert login_throttle.is_locked(4, 5) is False

    def test_at_threshold_locked(self):
        # Exactly max_attempts failures already recorded → next attempt locked.
        assert login_throttle.is_locked(5, 5) is True

    def test_above_threshold_locked(self):
        assert login_throttle.is_locked(9, 5) is True

    def test_zero_failures_not_locked(self):
        assert login_throttle.is_locked(0, 5) is False

    def test_nonpositive_max_disables_throttle(self):
        # max_attempts <= 0 must never lock everyone out.
        assert login_throttle.is_locked(100, 0) is False
        assert login_throttle.is_locked(100, -3) is False


class TestRetryAfterSeconds:
    def test_backoff_grows_then_caps_at_window(self):
        window, max_attempts = 300, 5
        base = window // max_attempts  # 60
        # At threshold: no failures "over", so base wait.
        assert login_throttle.retry_after_seconds(5, window, max_attempts) == base
        # One over → doubles.
        assert login_throttle.retry_after_seconds(6, window, max_attempts) == base * 2
        # Two over → quadruples.
        assert login_throttle.retry_after_seconds(7, window, max_attempts) == base * 4
        # Far over → capped at the window, never longer.
        assert login_throttle.retry_after_seconds(99, window, max_attempts) == window

    def test_never_exceeds_window(self):
        for fc in range(5, 50):
            v = login_throttle.retry_after_seconds(fc, 120, 5)
            assert 1 <= v <= 120

    def test_minimum_one_second(self):
        # Tiny window // max could floor toward 0 — must clamp to >= 1.
        assert login_throttle.retry_after_seconds(1, 0, 5) == 1
        assert login_throttle.retry_after_seconds(5, 3, 5) >= 1

    def test_nonpositive_max_returns_window(self):
        assert login_throttle.retry_after_seconds(10, 300, 0) == 300


# ── Store against a fake asyncpg pool ─────────────────────────────────────


class _LockTimeout(Exception):
    """Stands in for asyncpg's lock_not_available error, which the store
    recognises by SQLSTATE rather than by type."""

    sqlstate = "55P03"


class _TxCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    """Minimal asyncpg connection double backed by an in-memory row list.

    Understands the queries the store issues: the advisory-lock preamble,
    INSERT, both failure-count SELECTs (per-(username, ip) and per-username),
    the pair DELETE and the retention DELETE.

    Set ``lock_contended`` to make the advisory lock raise as it would under
    `lock_timeout`, which is the only way to reach the contention branch
    without a real Postgres.
    """

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.lock_contended = False
        self.lock_timeout_set: str | None = None

    def transaction(self):
        return _TxCtx()

    async def execute(self, query: str, *args):
        q = " ".join(query.split())
        if q.startswith("SET LOCAL lock_timeout"):
            self.lock_timeout_set = q
        elif q.startswith("SELECT pg_advisory_xact_lock"):
            if self.lock_contended:
                raise _LockTimeout("canceling statement due to lock timeout")
        elif q.startswith("INSERT INTO login_attempts"):
            # record() passes succeeded explicitly; the throttle's conditional
            # INSERT writes a literal FALSE and passes only the two keys.
            if len(args) == 3:
                username, client_ip, succeeded = args
            else:
                (username, client_ip), succeeded = args, False
            self._rows.append(
                {
                    "username": username,
                    "client_ip": client_ip,
                    "succeeded": succeeded,
                    "attempted_at": datetime.now(timezone.utc),
                }
            )
        elif q.startswith("DELETE FROM login_attempts WHERE attempted_at"):
            (cutoff,) = args
            before = len(self._rows)
            self._rows[:] = [r for r in self._rows if r["attempted_at"] >= cutoff]
            return f"DELETE {before - len(self._rows)}"
        elif q.startswith("DELETE FROM login_attempts"):
            username, client_ip = args
            self._rows[:] = [
                r
                for r in self._rows
                if not (r["username"] == username and r["client_ip"] == client_ip)
            ]
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected execute: {q}")
        return "OK"

    async def fetchval(self, query: str, *args):
        if len(args) == 3:  # per-(username, client_ip)
            username, client_ip, cutoff = args
        else:  # per-username, across all IPs
            username, cutoff = args
            client_ip = None
        return sum(
            1
            for r in self._rows
            if r["username"] == username
            and (client_ip is None or r["client_ip"] == client_ip)
            and r["succeeded"] is False
            and r["attempted_at"] >= cutoff
        )


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, rows: list[dict]):
        self.conn = _FakeConn(rows)

    def acquire(self):
        return _AcquireCtx(self.conn)


@pytest.fixture
def rows():
    return []


@pytest.fixture
def patched_pool(monkeypatch, rows):
    pool = _FakePool(rows)

    async def _get_pool():
        return pool

    monkeypatch.setattr(store_mod, "get_pool", _get_pool)
    return pool


class TestAtomicRecordAndCount:
    """The path the login handler actually takes.

    The store's previous tests exercised `recent_failures`, a reader nothing
    called — so the method that decides every throttle outcome had no
    store-level coverage at all. These replace them.
    """

    WINDOW, MAX_IP, MAX_USER = 300, 5, 50

    async def _attempt(self, store, username="alice", ip="1.2.3.4"):
        return await store.atomic_record_and_count(
            username, ip, self.WINDOW, self.MAX_IP, self.MAX_USER
        )

    @pytest.mark.asyncio
    async def test_returns_prior_counts_not_post_insert_counts(self, patched_pool, rows):
        """The first attempt must report zero prior failures. Reporting 1
        would burn an allowance on the user's first typo — and the handler
        compares the returned value directly against the threshold."""
        store = LoginAttemptStore()
        assert await self._attempt(store) == (0, 0)
        assert await self._attempt(store) == (1, 1)
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_records_while_under_the_threshold(self, patched_pool, rows):
        store = LoginAttemptStore()
        for _ in range(self.MAX_IP):
            await self._attempt(store)
        assert len(rows) == self.MAX_IP

    @pytest.mark.asyncio
    async def test_stops_recording_once_locked_so_the_window_can_drain(
        self, patched_pool, rows
    ):
        """The regression this method was rewritten for. Recording a rejected
        attempt refreshes the sliding window, so an attacker sending one
        request per window keeps an account — the bootstrap admin included —
        locked out permanently."""
        store = LoginAttemptStore()
        for _ in range(self.MAX_IP):
            await self._attempt(store)
        assert len(rows) == self.MAX_IP

        for _ in range(20):
            per_ip, _ = await self._attempt(store)
            assert per_ip == self.MAX_IP
        assert len(rows) == self.MAX_IP, "a refused attempt must not be recorded"

    @pytest.mark.asyncio
    async def test_the_window_actually_drains(self, patched_pool, rows):
        """Age every recorded row past the window and the account is usable
        again — the self-healing property the conditional INSERT restores."""
        store = LoginAttemptStore()
        for _ in range(self.MAX_IP):
            await self._attempt(store)
        assert (await self._attempt(store))[0] == self.MAX_IP

        old = datetime.now(timezone.utc) - timedelta(seconds=self.WINDOW + 60)
        for r in rows:
            r["attempted_at"] = old

        assert await self._attempt(store) == (0, 0)

    @pytest.mark.asyncio
    async def test_per_username_ceiling_sees_every_source_ip(self, patched_pool):
        """The per-(username, ip) counter multiplies an attacker's allowance
        by the number of addresses they control; the per-username count is
        what bounds guesses per account under IP rotation (CWE-307)."""
        store = LoginAttemptStore()
        for i in range(4):
            per_ip, per_user = await self._attempt(store, ip=f"10.0.0.{i}")
            assert per_ip == 0, "each fresh IP starts clean"
            assert per_user == i, "the account-wide count keeps climbing"

    @pytest.mark.asyncio
    async def test_a_different_username_is_unaffected(self, patched_pool):
        store = LoginAttemptStore()
        for _ in range(self.MAX_IP):
            await self._attempt(store, username="alice")
        assert await self._attempt(store, username="bob") == (0, 0)

    @pytest.mark.asyncio
    async def test_successes_do_not_count_toward_the_limit(self, patched_pool):
        store = LoginAttemptStore()
        for _ in range(9):
            await store.record("alice", "1.2.3.4", succeeded=True)
        assert await self._attempt(store) == (0, 0)


class TestLockContention:
    """A bounded wait on the per-username lock, so a flood cannot hold one
    pool connection per waiter and starve the rest of the service."""

    @pytest.mark.asyncio
    async def test_lock_timeout_is_set_before_the_lock_is_taken(self, patched_pool):
        store = LoginAttemptStore()
        await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)
        assert patched_pool.conn.lock_timeout_set is not None
        assert "SET LOCAL" in patched_pool.conn.lock_timeout_set, (
            "must be transaction-scoped — a session-level timeout would leak "
            "onto whatever uses this pooled connection next"
        )
        assert str(store_mod.LOCK_WAIT_MS) in patched_pool.conn.lock_timeout_set

    @pytest.mark.asyncio
    async def test_contention_raises_the_dedicated_error(self, patched_pool):
        patched_pool.conn.lock_contended = True
        store = LoginAttemptStore()
        with pytest.raises(store_mod.LoginThrottleContention):
            await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)

    @pytest.mark.asyncio
    async def test_contended_attempt_records_nothing(self, patched_pool, rows):
        """It never reached the counts, so it must not leave a row that a
        later attempt would count."""
        patched_pool.conn.lock_contended = True
        store = LoginAttemptStore()
        with pytest.raises(store_mod.LoginThrottleContention):
            await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)
        assert rows == []

    @pytest.mark.asyncio
    async def test_other_database_errors_are_not_swallowed(self, patched_pool):
        """Only SQLSTATE 55P03 means contention. Anything else is a real
        failure and must not be reported as a throttle decision."""

        class _Boom(Exception):
            sqlstate = "08006"  # connection_failure

        async def _execute(query, *args):
            if "pg_advisory_xact_lock" in query:
                raise _Boom("connection died")
            return "OK"

        patched_pool.conn.execute = _execute
        store = LoginAttemptStore()
        with pytest.raises(_Boom):
            await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)


class TestPurgeExpired:
    """Rows fall out of the window but never out of the table: `clear()` only
    fires on a *successful* login, which a brute-force run never produces."""

    @pytest.mark.asyncio
    async def test_deletes_only_rows_past_retention(self, patched_pool, rows):
        now = datetime.now(timezone.utc)
        rows.extend(
            {
                "username": "u",
                "client_ip": "1.1.1.1",
                "succeeded": False,
                "attempted_at": now - timedelta(seconds=age),
            }
            for age in (10, 100, 5000, 9000)
        )
        store = LoginAttemptStore()
        deleted = await store.purge_expired(3600)
        assert deleted == 2
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_is_old_enough(self, patched_pool, rows):
        rows.append(
            {
                "username": "u",
                "client_ip": "1.1.1.1",
                "succeeded": False,
                "attempted_at": datetime.now(timezone.utc),
            }
        )
        store = LoginAttemptStore()
        assert await store.purge_expired(3600) == 0
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_a_purge_cannot_delete_a_row_still_being_counted(
        self, patched_pool, rows
    ):
        """Retention is deliberately several windows wide. If a purge could
        remove a row inside the counting window it would hand an attacker a
        free reset."""
        store = LoginAttemptStore()
        await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)
        assert await store.purge_expired(1200) == 0
        assert (await store.atomic_record_and_count(
            "alice", "1.2.3.4", 300, 5, 50
        ))[0] == 1


class TestLoginAttemptStore:
    @pytest.mark.asyncio
    async def test_clear_drops_only_matching_pair(self, patched_pool):
        store = LoginAttemptStore()
        await store.record("alice", "1.2.3.4", succeeded=False)
        await store.record("alice", "9.9.9.9", succeeded=False)
        await store.clear("alice", "1.2.3.4")
        counts = await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)
        assert counts == (0, 1), "the other IP's row survives and still counts"

    @pytest.mark.asyncio
    async def test_fail_loud_without_pool(self, monkeypatch):
        """A DB outage must not silently degrade to "0 failures", which would
        turn it into an open brute-force window."""

        async def _no_pool():
            return None

        monkeypatch.setattr(store_mod, "get_pool", _no_pool)
        store = LoginAttemptStore()
        with pytest.raises(RuntimeError):
            await store.atomic_record_and_count("alice", "1.2.3.4", 300, 5, 50)
        with pytest.raises(RuntimeError):
            await store.record("alice", "1.2.3.4", succeeded=False)
        with pytest.raises(RuntimeError):
            await store.clear("alice", "1.2.3.4")
        with pytest.raises(RuntimeError):
            await store.purge_expired(3600)
