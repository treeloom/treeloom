"""Retry engine — failure-mode-aware retry with mutation hints.

Maps FailureMode → specific correction hints for the LLM.
Uses jittered exponential backoff to prevent thundering herds.
Injection failures are never retried (security bypass prevention).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from treeloom.domain.audit import FailureMode

# ── mutation hints: FailureMode → instruction for next prompt ──────

MUTATION_HINTS: dict[FailureMode, Optional[str]] = {
    FailureMode.SCHEMA_VIOLATION: (
        "Return ONLY a valid JSON object. Start with { and end with }. "
        "No markdown fencing. No preamble. No explanation."
    ),
    FailureMode.CONSTRAINT_VIOLATION: (
        "Re-read every numbered constraint. Each is a strict requirement, "
        "not a suggestion. Your response MUST satisfy ALL constraints."
    ),
    FailureMode.TOKEN_OVERFLOW: (
        "Your previous response was too long. Aim for half the length. "
        "Be concise and direct."
    ),
    FailureMode.TIMEOUT: (
        "Respond with a shorter, more direct answer. No conversational "
        "preamble. Get straight to the point."
    ),
    FailureMode.EMPTY_RESPONSE: (
        "You returned an empty response. Provide a complete answer as "
        "specified in the constraints."
    ),
    FailureMode.LLM_ERROR: (
        "The previous attempt failed due to a provider error. "
        "Please try again with a standard response."
    ),
    FailureMode.UNKNOWN: (
        "The previous attempt failed. Please try again with a valid response."
    ),
    # These are never retried:
    FailureMode.NONE: None,  # Not a failure
    FailureMode.CIRCUIT_OPEN: None,  # Circuit open — retry would be pointless
    FailureMode.PROMPT_INJECTION: None,  # Security event — hard stop
}


# ── retry configuration ────────────────────────────────────────────


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3  # Total attempts including first try
    base_delay_seconds: float = 1.0  # Base backoff before jitter
    max_delay_seconds: float = 30.0  # Cap on backoff
    jitter_factor: float = 0.5  # ±50% random jitter


# ── result of a retry decision ─────────────────────────────────────


@dataclass
class RetryDecision:
    """What the retry engine decided about a failed call."""

    should_retry: bool
    failure_mode: FailureMode
    attempt: int
    delay_seconds: float = 0.0
    mutation_hint: str = ""
    reason: str = ""

    @staticmethod
    def no_retry(
        failure_mode: FailureMode, attempt: int, reason: str
    ) -> "RetryDecision":
        return RetryDecision(
            should_retry=False,
            failure_mode=failure_mode,
            attempt=attempt,
            reason=reason,
        )

    @staticmethod
    def retry(
        failure_mode: FailureMode,
        attempt: int,
        hint: str,
        delay_seconds: float,
    ) -> "RetryDecision":
        return RetryDecision(
            should_retry=True,
            failure_mode=failure_mode,
            attempt=attempt,
            mutation_hint=hint,
            delay_seconds=delay_seconds,
            reason="retry with mutation hint",
        )


# ── the engine ─────────────────────────────────────────────────────


class RetryEngine:
    """Decides whether to retry and what hint to inject.

    Usage:
        engine = RetryEngine()
        decision = engine.evaluate(attempt=1, failure_mode=FailureMode.SCHEMA_VIOLATION)
        if decision.should_retry:
            time.sleep(decision.delay_seconds)
            # next_prompt += decision.mutation_hint
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self._config = config or RetryConfig()

    # ── public API ──────────────────────────────────────────────────

    def evaluate(
        self, attempt: int, failure_mode: FailureMode
    ) -> RetryDecision:
        """Decide whether this failure should be retried.

        Args:
            attempt: Current attempt number (1-based).
            failure_mode: Why the last call failed.

        Returns:
            RetryDecision with should_retry flag, delay, and mutation hint.
        """
        # Never retry injection attempts
        if failure_mode == FailureMode.PROMPT_INJECTION:
            return RetryDecision.no_retry(
                failure_mode, attempt, "prompt injection — security hard stop"
            )

        # Never retry when circuit is open
        if failure_mode == FailureMode.CIRCUIT_OPEN:
            return RetryDecision.no_retry(
                failure_mode, attempt, "circuit open — provider unavailable"
            )

        # No failure — no retry needed
        if failure_mode == FailureMode.NONE:
            return RetryDecision.no_retry(
                failure_mode, attempt, "no failure — success"
            )

        # Check attempt budget
        if attempt >= self._config.max_attempts:
            return RetryDecision.no_retry(
                failure_mode,
                attempt,
                f"exhausted max attempts ({self._config.max_attempts})",
            )

        # Get mutation hint
        hint = MUTATION_HINTS.get(failure_mode)
        if hint is None:
            # Fallback: generic hint for unhandled failure modes
            hint = (
                "Your previous response did not meet requirements. "
                "Please provide a valid response."
            )

        delay = self._backoff(attempt)

        return RetryDecision.retry(
            failure_mode=failure_mode,
            attempt=attempt,
            hint=hint,
            delay_seconds=delay,
        )

    def mutation_hint_for(self, failure_mode: FailureMode) -> Optional[str]:
        """Get the mutation hint for a failure mode (without evaluating retry)."""
        return MUTATION_HINTS.get(failure_mode)

    # ── internal ────────────────────────────────────────────────────

    def _backoff(self, attempt: int) -> float:
        """Jittered exponential backoff.

        delay = base_delay * 2^(attempt-1) ± jitter_factor%
        Capped at max_delay_seconds.
        """
        base = self._config.base_delay_seconds * (2 ** (attempt - 1))
        capped = min(base, self._config.max_delay_seconds)
        jitter = capped * self._config.jitter_factor * random.uniform(-1, 1)
        return max(0.0, capped + jitter)
