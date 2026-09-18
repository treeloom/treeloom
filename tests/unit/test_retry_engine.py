"""Tests for RetryEngine domain model."""

from treeloom.domain.audit import FailureMode
from treeloom.domain.retry_engine import RetryEngine, RetryConfig, RetryDecision


class TestRetryEngine:
    """Failure-mode-aware retry decisions."""

    def test_schema_violation_gets_retry_with_hint(self):
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.SCHEMA_VIOLATION
        )
        assert decision.should_retry is True
        assert "valid JSON" in decision.mutation_hint
        assert "Start with {" in decision.mutation_hint
        assert decision.delay_seconds >= 0

    def test_constraint_violation_gets_constraint_hint(self):
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=2, failure_mode=FailureMode.CONSTRAINT_VIOLATION
        )
        assert decision.should_retry is True
        assert "constraint" in decision.mutation_hint.lower()

    def test_timeout_gets_shorter_response_hint(self):
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.TIMEOUT
        )
        assert decision.should_retry is True
        assert "shorter" in decision.mutation_hint.lower()

    def test_token_overflow_gets_half_length_hint(self):
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=2, failure_mode=FailureMode.TOKEN_OVERFLOW
        )
        assert decision.should_retry is True
        assert "half" in decision.mutation_hint.lower()

    def test_empty_response_gets_complete_answer_hint(self):
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.EMPTY_RESPONSE
        )
        assert decision.should_retry is True
        assert "empty" in decision.mutation_hint.lower()

    def test_llm_error_gets_standard_hint(self):
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.LLM_ERROR
        )
        assert decision.should_retry is True

    def test_none_failure_mode_returns_no_retry(self):
        """Success — no retry needed."""
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.NONE
        )
        assert decision.should_retry is False
        assert "success" in decision.reason.lower()

    def test_prompt_injection_never_retried(self):
        """Security event — hard stop, no matter the attempt count."""
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.PROMPT_INJECTION
        )
        assert decision.should_retry is False
        assert "security" in decision.reason.lower()

    def test_circuit_open_never_retried(self):
        """Circuit open — retry would hit the same wall."""
        engine = RetryEngine()
        decision = engine.evaluate(
            attempt=1, failure_mode=FailureMode.CIRCUIT_OPEN
        )
        assert decision.should_retry is False
        assert "circuit" in decision.reason.lower()

    def test_exhausted_attempts_no_retry(self):
        """After max attempts, no more retries."""
        config = RetryConfig(max_attempts=3)
        engine = RetryEngine(config)

        decision = engine.evaluate(
            attempt=3, failure_mode=FailureMode.SCHEMA_VIOLATION
        )
        assert decision.should_retry is False
        assert "exhausted" in decision.reason.lower()

    def test_attempt_2_of_3_allows_retry(self):
        """Still within budget."""
        config = RetryConfig(max_attempts=3)
        engine = RetryEngine(config)

        decision = engine.evaluate(
            attempt=2, failure_mode=FailureMode.SCHEMA_VIOLATION
        )
        assert decision.should_retry is True

    def test_mutation_hint_for_returns_hint(self):
        engine = RetryEngine()
        hint = engine.mutation_hint_for(FailureMode.TIMEOUT)
        assert hint is not None
        assert "shorter" in hint.lower()

    def test_mutation_hint_for_none_returns_none(self):
        engine = RetryEngine()
        hint = engine.mutation_hint_for(FailureMode.NONE)
        assert hint is None

    def test_backoff_increases_with_attempts(self):
        """Later attempts should have longer backoff times."""
        engine = RetryEngine(RetryConfig(base_delay_seconds=1.0, jitter_factor=0.0))
        d1 = engine.evaluate(1, FailureMode.SCHEMA_VIOLATION)
        d2 = engine.evaluate(2, FailureMode.SCHEMA_VIOLATION)
        # Without jitter, delay should double
        assert d2.delay_seconds > d1.delay_seconds

    def test_backoff_capped_at_max(self):
        """Backoff should not exceed max_delay."""
        config = RetryConfig(
            max_attempts=5,  # need room for attempt 2
            base_delay_seconds=10.0,
            max_delay_seconds=15.0,
            jitter_factor=0.0,
        )
        engine = RetryEngine(config)
        # attempt 2: base * 2^1 = 20, capped at 15
        decision = engine.evaluate(2, FailureMode.SCHEMA_VIOLATION)
        assert decision.should_retry is True
        assert decision.delay_seconds == 15.0

    def test_all_failure_modes_have_hints(self):
        """Every retryable failure mode should have a hint."""
        engine = RetryEngine()
        non_retryable = {
            FailureMode.NONE,
            FailureMode.CIRCUIT_OPEN,
            FailureMode.PROMPT_INJECTION,
        }
        for fm in FailureMode:
            if fm in non_retryable:
                continue
            hint = engine.mutation_hint_for(fm)
            assert hint is not None, f"{fm} should have a mutation hint"
            assert len(hint) > 10, f"{fm} hint too short: {hint}"
