"""LLM Caller — wraps raw _chat() with the full control layer.

CircuitBreaker → RetryEngine → AuditLogger → ResponseValidator → FallbackRouter.

This is the adapter that wires the domain control-layer components into
the existing llm_adapter. Imported and used by generate_hyde() and
generate_summary() to gain resilience without changing their signatures.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Callable, Coroutine, Optional

from treeloom.domain.audit import AuditRecord, FailureMode, Operation
from treeloom.domain.circuit_breaker import CircuitBreaker, CircuitConfig
from treeloom.domain.fallback_router import FallbackRouter
from treeloom.domain.response_validator import ResponseSchema, ResponseValidator
from treeloom.domain.retry_engine import RetryConfig, RetryEngine
from treeloom.infrastructure.audit import JSONLAuditLogger

# ── singleton instances (module-level, shared across all callers) ──

from pathlib import Path

from treeloom.adapters.llm_api.llm_adapter import (
    LLM_MODEL,
    LLM_TIMEOUT,
    _chat,
    _HYDE_CACHE,
)

# Circuit breaker: 5 consecutive failures → OPEN, 30s recovery
_breaker = CircuitBreaker(
    CircuitConfig(failure_threshold=5, recovery_seconds=30.0)
)

# Retry engine: 3 total attempts, 1s base backoff, ±50% jitter
_retry = RetryEngine(
    RetryConfig(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=30.0)
)

# Audit logger: writes to ~/.treeloom/audit.jsonl
_audit = JSONLAuditLogger(
    Path("~/.treeloom/audit.jsonl"), rebuild=True
)

# Fallback router: strategies registered per operation type
_fallback = FallbackRouter()

# Default response schema for HyDE (code only, no markdown)
_HYDE_SCHEMA = ResponseSchema(
    min_length=5,
    max_length=8000,
    forbidden_phrases=[
        "I can't", "I don't know", "I cannot", "as an AI",
        "```json", "here's",
    ],
)

# Default response schema for summaries (concise sentence)
_SUMMARY_SCHEMA = ResponseSchema(
    min_length=5,
    max_length=500,
    forbidden_phrases=[
        "I can't", "I don't know", "as an AI",
        "here is", "this code",
    ],
)


# ── public API ─────────────────────────────────────────────────────


async def call_with_control_layer(
    messages: list[dict],
    max_tokens: int,
    operation: Operation,
    validator: Optional[ResponseValidator] = None,
    audit_id: str = "",
    timeout: Optional[float] = None,
) -> tuple[Optional[str], str]:
    """Call the LLM through the full control layer stack.

    Args:
        messages: Chat messages (system + user).
        max_tokens: Max tokens for the LLM response.
        operation: What operation this is (HYDE, SUMMARY, etc.).
        validator: Optional ResponseValidator for output checking.
        audit_id: Optional correlation ID (auto-generated if empty).
        timeout: Optional per-call timeout override.

    Returns:
        (response_text, strategy_name) where strategy_name is:
          - "simple" for first-try success
          - "prompt_mutation" for success after retry
          - "fallback:<name>" for fallback success
          - "error" if everything failed
    """
    input_hash = _hash_messages(messages)
    call_model = LLM_MODEL
    effective_timeout = timeout or LLM_TIMEOUT

    # ── Attempt loop ───────────────────────────────────────────────
    for attempt in range(1, _retry._config.max_attempts + 1):
        # 1. Circuit breaker check
        if _breaker.is_open():
            _audit.log(
                audit_id=audit_id,
                operation=operation,
                model=call_model,
                attempt=attempt,
                failure_mode=FailureMode.CIRCUIT_OPEN,
                input_hash=input_hash,
            )
            # Route to fallback
            result, strategy, _ = _fallback.execute(
                messages, FailureMode.CIRCUIT_OPEN, attempt
            )
            if result is not None:
                return result, f"fallback:{strategy}"
            return None, "error"

        # 2. Call the LLM
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                _chat(
                    messages,
                    max_tokens=max_tokens,
                    operation=str(getattr(operation, "value", operation) or ""),
                ),
                timeout=effective_timeout,
            )
            latency = (time.monotonic() - t0) * 1000
        except asyncio.TimeoutError:
            latency = effective_timeout * 1000
            raw = None

        # 3. Handle failure (timeout or None)
        if raw is None:
            fm = FailureMode.TIMEOUT if raw is None else FailureMode.LLM_ERROR
            _breaker.record_failure()
            _audit.log(
                audit_id=audit_id,
                operation=operation,
                model=call_model,
                attempt=attempt,
                failure_mode=fm,
                latency_ms=latency,
                passed=False,
                input_hash=input_hash,
            )
            decision = _retry.evaluate(attempt, fm)
            if decision.should_retry:
                _inject_mutation_hint(messages, decision.mutation_hint)
                await asyncio.sleep(decision.delay_seconds)
                continue
            # Retries exhausted — try fallback
            result, strategy, _ = _fallback.execute(messages, fm, attempt)
            if result is not None:
                return result, f"fallback:{strategy}"
            return None, "error"

        # 4. Validate response
        _breaker.record_success()
        if validator is not None:
            validation = validator.validate(raw)
            if not validation.passed:
                _audit.log(
                    audit_id=audit_id,
                    operation=operation,
                    model=call_model,
                    attempt=attempt,
                    failure_mode=validation.failure_mode,
                    latency_ms=latency,
                    passed=False,
                    input_hash=input_hash,
                )
                decision = _retry.evaluate(attempt, validation.failure_mode)
                if decision.should_retry:
                    _inject_mutation_hint(messages, decision.mutation_hint)
                    await asyncio.sleep(decision.delay_seconds)
                    continue
                result, strategy, _ = _fallback.execute(
                    messages, validation.failure_mode, attempt
                )
                if result is not None:
                    return result, f"fallback:{strategy}"
                return None, "error"
            raw = validation.cleaned_output

        # 5. Success
        _audit.log(
            audit_id=audit_id,
            operation=operation,
            model=call_model,
            attempt=attempt,
            failure_mode=FailureMode.NONE,
            latency_ms=latency,
            passed=True,
            input_hash=input_hash,
        )
        strategy = "simple" if attempt == 1 else "prompt_mutation"
        return raw, strategy

    # Should never reach here (max_attempts handled above)
    return None, "error"


def _inject_mutation_hint(messages: list[dict], hint: str) -> None:
    """Inject a retry mutation hint into the system message."""
    if not hint:
        return
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = f"{msg['content']}\n\n[Hint from previous attempt: {hint}]"
            return
    # No system message — inject as a new one at the start
    messages.insert(0, {"role": "system", "content": f"Hint: {hint}"})


def _hash_messages(messages: list[dict]) -> str:
    """Create a stable hash of the messages for deduplication."""
    raw = "|".join(
        f"{m['role']}:{m['content'][:200]}" for m in messages
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def register_fallback(name: str, strategy) -> None:
    """Register a fallback strategy for the global router.

    Called during application startup wiring. Strategies are tried
    in registration order.

    Example:
        register_fallback("cached_hyde", make_cached_hyde_strategy(cache))
        register_fallback("vector_only", make_vector_only_strategy(search_fn))
    """
    _fallback.register(name, strategy)


def get_audit_stats() -> dict:
    """Get current audit analytics (for /health endpoint)."""
    return _audit.stats()


def get_breaker_state() -> str:
    """Get current circuit breaker state (for /health endpoint)."""
    return _breaker.state.name


# Re-export singletons for testing
_circuit_breaker = _breaker
_retry_engine = _retry
_audit_logger = _audit
_fallback_router = _fallback
