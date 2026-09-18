"""Pure decision logic for login rate-limiting.

No I/O lives here — the DB-backed sliding-window counter
(`adapters/postgresql/login_attempt_store.py`) supplies the failure count
and the window, and the `/auth/login` handler wires the two together. Keeping
the policy pure makes the lockout threshold and backoff trivially
unit-testable.
"""

from __future__ import annotations


def is_locked(failure_count: int, max_attempts: int) -> bool:
    """Return True when this (username, client_ip) has reached the failure
    ceiling within the sliding window and should be refused with 429.

    ``failure_count`` is the number of *failed* attempts already recorded in
    the window (the current attempt is not yet counted). Lockout triggers at
    ``>= max_attempts`` so that exactly ``max_attempts`` failures are allowed
    before the next attempt is throttled.

    A non-positive ``max_attempts`` disables throttling entirely (never
    locked) rather than locking everyone out.
    """
    if max_attempts <= 0:
        return False
    return failure_count >= max_attempts


def retry_after_seconds(
    failure_count: int,
    window_seconds: int,
    max_attempts: int,
) -> int:
    """Seconds the caller should wait before retrying, for the ``Retry-After``
    header on a 429.

    Exponential backoff in the number of failures *over* the threshold,
    capped at the window length: a caller can never be told to wait longer
    than the sliding window itself (after which old failures age out and the
    count naturally drops below the threshold). The minimum is 1 second.

    base = window // max_attempts (a sane fraction of the window); each
    failure beyond ``max_attempts`` doubles the wait, clamped to
    ``window_seconds``.
    """
    if window_seconds <= 0:
        return 1
    if max_attempts <= 0:
        return window_seconds

    base = max(window_seconds // max_attempts, 1)
    over = max(failure_count - max_attempts, 0)
    wait = base * (2 ** over)
    return max(1, min(wait, window_seconds))
