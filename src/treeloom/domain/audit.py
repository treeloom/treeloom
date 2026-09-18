"""Audit logging domain model — protocols and data structures.

Following DDD: this module defines WHAT the audit system does (the contract)
without specifying HOW it persists or indexes. Infrastructure adapters implement
these protocols.
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol


class FailureMode(str, Enum):
    """Failure modes mapped from the control layer architecture.

    Each mode drives specific retry behavior and fallback strategy selection.
    """

    NONE = "none"                       # Call succeeded on first attempt
    SCHEMA_VIOLATION = "schema_violation"   # JSON structure invalid or missing keys
    CONSTRAINT_VIOLATION = "constraint_violation"  # Output violates hard rules
    TOKEN_OVERFLOW = "token_overflow"       # Response exceeded context budget
    TIMEOUT = "timeout"                     # LLM call timed out
    CIRCUIT_OPEN = "circuit_open"           # Circuit breaker rejected call
    LLM_ERROR = "llm_error"                 # Provider returned an error
    EMPTY_RESPONSE = "empty_response"       # Model returned nothing
    PROMPT_INJECTION = "prompt_injection"   # Blocked by InputGuard — never retried
    UNKNOWN = "unknown"                     # Catch-all for unclassified failures


class Operation(str, Enum):
    """LLM operations Treeloom performs."""

    HYDE = "hyde"                   # Hypothetical document embedding expansion
    CHUNK_SUMMARY = "chunk_summary"  # Per-chunk code summarization
    COMMUNITY_SUMMARY = "community_summary"  # Community analysis summarization
    BENCHMARK_QUERY = "benchmark_query"  # Benchmark mode session-model search


@dataclass(frozen=True)
class AuditRecord:
    """Immutable audit entry — one per LLM attempt.

    Fields match the article's JSONL schema with extensions for Treeloom's
    multi-operation context (operation, model, input_hash for deduplication).
    """

    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    attempt: int = 1
    operation: Operation = Operation.HYDE
    model: str = "unknown"
    failure_mode: FailureMode = FailureMode.NONE
    latency_ms: float = 0.0
    passed: bool = False
    input_hash: str = ""
    token_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to JSON-serializable dict (one JSONL line)."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "attempt": self.attempt,
            "operation": self.operation.value,
            "model": self.model,
            "failure_mode": self.failure_mode.value,
            "latency_ms": round(self.latency_ms, 1),
            "passed": self.passed,
            "input_hash": self.input_hash,
            "token_count": self.token_count,
        }


class AuditIndex(Protocol):
    """In-memory analytics index protocol.

    Rebuilt from the JSONL file on startup. Provides O(1) lookups for
    pass rates, failure distributions, and latency percentiles.
    """

    def ingest(self, record: AuditRecord) -> None:
        """Index a new record (called after write)."""
        ...

    def pass_rate(self) -> float:
        """Fraction of attempts that passed."""
        ...

    def failure_distribution(self) -> dict[FailureMode, int]:
        """Count of attempts per failure mode."""
        ...

    def latency_percentiles(self) -> dict[str, float]:
        """p50, p90, p99 latency in ms."""
        ...

    def total_attempts(self) -> int:
        """Total number of attempts indexed."""
        ...


class AuditLogWriter(Protocol):
    """Persistence protocol for audit records."""

    def write(self, record: AuditRecord) -> None:
        """Append one record to the log (thread-safe)."""
        ...

    def read_all(self) -> list[AuditRecord]:
        """Read all records from the log (used to rebuild index)."""
        ...
