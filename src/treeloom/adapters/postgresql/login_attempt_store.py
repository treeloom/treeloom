"""asyncpg-backed sliding-window store for login rate-limiting.

Records every login attempt keyed by ``(username, client_ip)`` and answers
"how many failures in the last N seconds?" so the `/auth/login` handler can
throttle brute-force attempts across all indexer workers (the counter lives
in Postgres, not process memory).

Schema lives in migration `019_login_attempts.sql`.

Fail-loud: every method raises ``RuntimeError`` if the Postgres pool is
unavailable, mirroring the summary-cache helpers. The login handler must be
able to trust the count — silently degrading to "0 failures" would turn a DB
outage into an open brute-force window.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from treeloom.adapters.postgresql.connection import get_pool
from treeloom.domain.authorization import login_throttle

_POOL_REQUIRED = (
    "PostgreSQL pool unavailable — login rate-limiting requires DATABASE_URL"
)

# How long an attempt waits for the per-username advisory lock before giving
# up. See LoginThrottleContention for why there is a bound at all.
LOCK_WAIT_MS = int(os.environ.get("TREELOOM_LOGIN_LOCK_WAIT_MS", "250"))

# Postgres SQLSTATE for a statement that hit `lock_timeout`.
_LOCK_NOT_AVAILABLE = "55P03"


class LoginThrottleContention(RuntimeError):
    """Raised when the per-username lock could not be taken in time.

    The lock serializes attempts for one username, and it is held for the
    duration of a pooled connection's transaction. Waiting on it without a
    bound means a flood against a single username parks one pool connection
    per in-flight request — the login path starving every other endpoint of
    database connections, which is a denial of service reached through the
    very mechanism added to stop one.

    So the wait is bounded, and exceeding it is itself the answer: more
    simultaneous attempts on one account than the lock can drain in
    LOCK_WAIT_MS is a flood, and the caller should refuse it exactly as it
    refuses an over-threshold count. Two people legitimately logging into a
    shared account at the same instant will normally acquire it in
    microseconds; if one does lose the race, they retry.
    """


class LoginAttemptStore:
    """CRUD over the `login_attempts` sliding-window table."""

    async def atomic_record_and_count(
        self,
        username: str,
        client_ip: str,
        window_seconds: int,
        max_attempts: int,
        max_per_username: int,
    ) -> tuple[int, int]:
        """Count prior failures in the window and, unless already locked,
        record this attempt. Returns ``(per_ip, per_username)`` PRIOR failure
        counts and whether a row was written: ``(per_ip, per_username)``.

        Both COUNTs and the conditional INSERT run in one transaction behind a
        per-username advisory lock, so concurrent attempts serialize and each
        sees every committed predecessor. That closes the check-then-record
        TOCTOU (CWE-362) where N concurrent requests all read the same
        pre-increment count and bypassed the threshold.

        The INSERT is CONDITIONAL, and that is load-bearing. An earlier version
        recorded unconditionally and let the caller refuse afterwards, which
        meant every *rejected* request also pushed a fresh row into the window.
        The sliding window then never drained: an attacker who knew a username
        could send one request per window forever and keep that account — the
        bootstrap admin included — locked out permanently. Writing only when
        the attempt will actually be evaluated restores self-healing, and the
        advisory lock still provides the serialization the TOCTOU fix needs.
        """
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(window_seconds, 0))
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Serialize concurrent attempts for this username; the lock is
                # held until this transaction commits, so the counts below see
                # every other in-flight attempt that has committed its INSERT.
                #
                # The wait is bounded so a flood cannot pin one pool connection
                # per in-flight request — see LoginThrottleContention. SET
                # LOCAL scopes the timeout to this transaction, so no other
                # user of this pooled connection inherits it.
                await conn.execute(f"SET LOCAL lock_timeout = '{LOCK_WAIT_MS}ms'")
                try:
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtext($1))", username
                    )
                except Exception as exc:
                    if getattr(exc, "sqlstate", None) == _LOCK_NOT_AVAILABLE:
                        raise LoginThrottleContention(username) from exc
                    raise
                per_ip = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM login_attempts
                     WHERE username = $1
                       AND client_ip = $2
                       AND succeeded = FALSE
                       AND attempted_at >= $3
                    """,
                    username,
                    client_ip,
                    cutoff,
                )
                per_username = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM login_attempts
                     WHERE username = $1
                       AND succeeded = FALSE
                       AND attempted_at >= $2
                    """,
                    username,
                    cutoff,
                )
                per_ip = int(per_ip or 0)
                per_username = int(per_username or 0)
                # Record ONLY when this attempt will actually be evaluated.
                # Recording while already locked would keep refreshing the
                # window and make the lockout permanent. The domain predicate
                # is reused so the store and the handler cannot disagree about
                # what "locked" means.
                if not (
                    login_throttle.is_locked(per_ip, max_attempts)
                    or login_throttle.is_locked(per_username, max_per_username)
                ):
                    await conn.execute(
                        """
                        INSERT INTO login_attempts (username, client_ip, succeeded)
                        VALUES ($1, $2, FALSE)
                        """,
                        username,
                        client_ip,
                    )
        return per_ip, per_username

    async def record(self, username: str, client_ip: str, succeeded: bool) -> None:
        """Append one attempt outcome for (username, client_ip)."""
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO login_attempts (username, client_ip, succeeded)
                VALUES ($1, $2, $3)
                """,
                username,
                client_ip,
                succeeded,
            )

    async def purge_expired(self, retention_seconds: int) -> int:
        """Delete attempts older than *retention_seconds*; return the count.

        Nothing reads a row once it falls out of the sliding window, so
        without this the table only ever grows: `clear()` fires on a
        *successful* login for one (username, client_ip) pair, which is
        precisely the traffic an attacker never generates. A spray across ten
        thousand usernames leaves ten thousand rows behind permanently, and
        the index that makes the window query fast grows with them.
        """
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=max(retention_seconds, 0)
        )
        async with pool.acquire() as conn:
            status = await conn.execute(
                "DELETE FROM login_attempts WHERE attempted_at < $1", cutoff
            )
        # asyncpg returns the command tag, e.g. "DELETE 42".
        try:
            return int(status.split()[-1])
        except (AttributeError, ValueError, IndexError):
            return 0

    async def clear(self, username: str, client_ip: str) -> None:
        """Drop all recorded attempts for (username, client_ip) — called on a
        successful login so prior typos don't carry forward."""
        pool = await get_pool()
        if pool is None:
            raise RuntimeError(_POOL_REQUIRED)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM login_attempts
                 WHERE username = $1 AND client_ip = $2
                """,
                username,
                client_ip,
            )
