"""Infrastructure adapter — JSONL file writer + in-memory analytics index.

Implements AuditLogWriter and AuditIndex protocols from domain/audit.py.
Thread-safe via threading.Lock. Index rebuilt from file on startup.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from treeloom.domain.audit import (
    AuditIndex,
    AuditLogWriter,
    AuditRecord,
    FailureMode,
    Operation,
)


class JSONLAuditLogger:
    """The orchestrator: owns a writer and an index, exposed as one object.

    Usage:
        logger = JSONLAuditLogger(Path("~/.treeloom/audit.jsonl"))
        logger.log(AuditRecord(operation=Operation.HYDE, passed=True, ...))

    Thread-safe: all write + index operations protected by a reentrant lock.
    """

    def __init__(self, path: Path, rebuild: bool = True):
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._writer = _JSONLFileWriter(self._path, self._lock)
        self._index = _InMemoryIndex()

        if rebuild and self._path.exists():
            self._rebuild_index()

    # ── public API ──────────────────────────────────────────────────

    def log(
        self,
        audit_id: str = "",
        operation: Operation = Operation.HYDE,
        model: str = "unknown",
        attempt: int = 1,
        failure_mode: FailureMode = FailureMode.NONE,
        latency_ms: float = 0.0,
        passed: bool = False,
        input_hash: str = "",
        token_count: int = 0,
    ) -> AuditRecord:
        """Log one LLM attempt. Returns the record for caller convenience.

        This is the primary entry point — callers don't construct AuditRecord
        directly.
        """
        record = AuditRecord(
            audit_id=audit_id,
            attempt=attempt,
            operation=operation,
            model=model,
            failure_mode=failure_mode,
            latency_ms=latency_ms,
            passed=passed,
            input_hash=input_hash,
            token_count=token_count,
        )
        with self._lock:
            self._writer.write(record)
            self._index.ingest(record)
        return record

    # ── analytics pass-through ─────────────────────────────────────
    # Delegates to the in-memory index. All reads are lock-free
    # (immutable data structures after ingest).

    def pass_rate(self) -> float:
        with self._lock:
            return self._index.pass_rate()

    def failure_distribution(self) -> dict[FailureMode, int]:
        with self._lock:
            return self._index.failure_distribution()

    def latency_percentiles(self) -> dict[str, float]:
        with self._lock:
            return self._index.latency_percentiles()

    def total_attempts(self) -> int:
        with self._lock:
            return self._index.total_attempts()

    def stats(self) -> dict:
        """Convenience: all analytics in one dict."""
        with self._lock:
            return {
                "total_attempts": self._index.total_attempts(),
                "pass_rate": self._index.pass_rate(),
                "failure_distribution": {
                    k.value: v
                    for k, v in self._index.failure_distribution().items()
                },
                "latency_p50_ms": self._index.latency_percentiles().get(
                    "p50", 0.0
                ),
                "latency_p90_ms": self._index.latency_percentiles().get(
                    "p90", 0.0
                ),
                "latency_p99_ms": self._index.latency_percentiles().get(
                    "p99", 0.0
                ),
            }

    # ── internal ────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Rebuild in-memory index from the JSONL file."""
        with self._lock:
            for record in self._writer.read_all():
                self._index.ingest(record)


# ══════════════════════════════════════════════════════════════════════
# Internal implementations (not part of public API)
# ══════════════════════════════════════════════════════════════════════


class _JSONLFileWriter:
    """Append-only JSONL writer. Thread-safe via shared lock."""

    def __init__(self, path: Path, lock: threading.RLock):
        self._path = path
        self._lock = lock

    def write(self, record: AuditRecord) -> None:
        """Append one JSON object as a line."""
        line = json.dumps(record.to_dict(), sort_keys=True)
        with open(self._path, "a") as f:
            f.write(line + "\n")

    def read_all(self) -> list[AuditRecord]:
        """Read all records back from the JSONL file."""
        records: list[AuditRecord] = []
        if not self._path.exists():
            return records
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(_dict_to_record(data))
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Corrupt line — skip and continue
                    pass
        return records


class _InMemoryIndex:
    """In-memory analytics index. Ingested records are immutable."""

    def __init__(self):
        self._latencies: list[float] = []
        self._failure_counts: dict[FailureMode, int] = {}
        self._passed: int = 0
        self._total: int = 0

    def ingest(self, record: AuditRecord) -> None:
        self._total += 1
        if record.passed:
            self._passed += 1
        self._latencies.append(record.latency_ms)
        fm = record.failure_mode
        if fm != FailureMode.NONE or not record.passed:
            self._failure_counts[fm] = self._failure_counts.get(fm, 0) + 1

    def pass_rate(self) -> float:
        if self._total == 0:
            return 0.0
        return self._passed / self._total

    def failure_distribution(self) -> dict[FailureMode, int]:
        return dict(self._failure_counts)

    def latency_percentiles(self) -> dict[str, float]:
        if not self._latencies:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        sorted_lat = sorted(self._latencies)
        return {
            "p50": _percentile(sorted_lat, 50),
            "p90": _percentile(sorted_lat, 90),
            "p99": _percentile(sorted_lat, 99),
        }

    def total_attempts(self) -> int:
        return self._total


# ── helpers ─────────────────────────────────────────────────────────


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Nearest-rank percentile (returns raw value, not interpolated)."""
    if not sorted_data:
        return 0.0
    idx = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * pct / 100)))
    return round(sorted_data[idx], 1)


def _dict_to_record(data: dict) -> AuditRecord:
    """Deserialize a dict back to AuditRecord (for index rebuild)."""
    return AuditRecord(
        audit_id=data.get("audit_id", ""),
        timestamp=data.get("timestamp", ""),
        attempt=data.get("attempt", 1),
        operation=Operation(data.get("operation", "hyde")),
        model=data.get("model", "unknown"),
        failure_mode=FailureMode(
            data.get("failure_mode", "none")
        ),
        latency_ms=data.get("latency_ms", 0.0),
        passed=data.get("passed", False),
        input_hash=data.get("input_hash", ""),
        token_count=data.get("token_count", 0),
    )
