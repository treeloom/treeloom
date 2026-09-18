"""Circuit breaker — three-state FSM for LLM API resilience.

Prevents cascading failures by short-circuiting LLM calls after
consecutive errors. States: CLOSED (normal) → OPEN (failing) →
HALF_OPEN (probing) → CLOSED.

Thread-safe: all state transitions protected by threading.Lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from treeloom.domain.audit import FailureMode


class CircuitState(Enum):
    """Three states of the circuit breaker."""

    CLOSED = auto()  # Normal operation — calls pass through
    OPEN = auto()  # Failing — calls rejected immediately
    HALF_OPEN = auto()  # Probing — one test call allowed


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker behavior.

    Defaults match the article's baseline: 5 failures before opening,
    30-second recovery window.
    """

    failure_threshold: int = 5  # Consecutive failures before OPEN
    recovery_seconds: float = 30.0  # Time before HALF_OPEN probe
    half_open_max_attempts: int = 1  # Probes before fully reopening


class CircuitBreaker:
    """Three-state FSM for LLM provider resilience.

    Usage:
        breaker = CircuitBreaker()
        result = breaker.call(lambda: llm_client.generate(...))

    When OPEN, calls return immediately with FAILURE_CIRCUIT_OPEN
    without making any LLM request.
    """

    def __init__(self, config: Optional[CircuitConfig] = None):
        self._config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_attempts: int = 0
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._current_state()

    def is_open(self) -> bool:
        """Check if the circuit is OPEN (calls should be rejected).

        This is the primary guard — callers check this before making
        any LLM request.

        In HALF_OPEN it admits at most `half_open_max_attempts` probes and
        then reports open again until one of them resolves. That cap was
        configured and reset but never counted or checked, so HALF_OPEN
        admitted unlimited CONCURRENT probes: every caller that arrived in the
        recovery window went through to a provider already known to be failing,
        which is the thundering herd the breaker exists to prevent (CWE-362).

        Counting here rather than in record_*() is deliberate — the probes are
        concurrent, so they must be counted when ADMITTED, not when they come
        back.
        """
        with self._lock:
            current = self._current_state()
            if current == CircuitState.OPEN:
                return True
            if current == CircuitState.HALF_OPEN:
                if self._half_open_attempts >= self._config.half_open_max_attempts:
                    return True
                self._half_open_attempts += 1
            return False

    def record_success(self) -> None:
        """Report a successful LLM call — resets failure count.

        In HALF_OPEN, a success transitions back to CLOSED.
        """
        with self._lock:
            self._failure_count = 0
            self._half_open_attempts = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Report a failed LLM call — increments failure count.

        After threshold consecutive failures, transitions to OPEN.
        In HALF_OPEN, a single failure reopens the circuit.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            current = self._current_state()
            if current == CircuitState.HALF_OPEN:
                # Probing failed — back to OPEN
                self._state = CircuitState.OPEN
                self._half_open_attempts = 0
            elif self._failure_count >= self._config.failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Force-reset to CLOSED (for testing/manual recovery)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_attempts = 0

    # ── internal ────────────────────────────────────────────────────

    def _current_state(self) -> CircuitState:
        """Determine current state, handling OPEN→HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._config.recovery_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_attempts = 0
                # Don't reset failure_count — we're still probing
        return self._state
