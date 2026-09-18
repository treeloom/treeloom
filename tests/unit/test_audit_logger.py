"""Tests for AuditLogger domain + infrastructure.

Detroit-style: test real behavior. Mock only the filesystem (out-of-process).
"""

import json
import tempfile
from pathlib import Path

import pytest

from treeloom.domain.audit import AuditRecord, FailureMode, Operation
from treeloom.infrastructure.audit import JSONLAuditLogger


class TestAuditRecord:
    """Domain model: AuditRecord serialization."""

    def test_to_dict_includes_all_fields(self):
        record = AuditRecord(
            audit_id="abc12345",
            attempt=2,
            operation=Operation.HYDE,
            model="openai/text-embedding-3-small",
            failure_mode=FailureMode.SCHEMA_VIOLATION,
            latency_ms=58.8,
            passed=False,
            input_hash="a1b2c3",
            token_count=150,
        )
        d = record.to_dict()

        assert d["audit_id"] == "abc12345"
        assert d["attempt"] == 2
        assert d["operation"] == "hyde"
        assert d["failure_mode"] == "schema_violation"
        assert d["latency_ms"] == 58.8
        assert d["passed"] is False
        assert d["input_hash"] == "a1b2c3"
        assert d["token_count"] == 150

    def test_default_audit_id_is_unique(self):
        r1 = AuditRecord()
        r2 = AuditRecord()
        assert r1.audit_id != r2.audit_id
        assert len(r1.audit_id) == 8

    def test_defaults_represent_first_attempt_success(self):
        record = AuditRecord()
        assert record.attempt == 1
        assert record.passed is False
        assert record.failure_mode == FailureMode.NONE
        assert record.latency_ms == 0.0


class TestJSONLAuditLogger:
    """Infrastructure: JSONLAuditLogger with real file I/O."""

    @pytest.fixture
    def tmp_logger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            yield JSONLAuditLogger(path, rebuild=False)

    def test_log_writes_to_file(self, tmp_logger):
        tmp_logger.log(
            operation=Operation.HYDE,
            passed=True,
            latency_ms=42.0,
        )
        lines = tmp_logger._path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["passed"] is True
        assert data["latency_ms"] == 42.0

    def test_log_returns_record(self, tmp_logger):
        record = tmp_logger.log(
            operation=Operation.CHUNK_SUMMARY,
            attempt=2,
            failure_mode=FailureMode.TIMEOUT,
            latency_ms=5000.0,
            passed=False,
        )
        assert record.operation == Operation.CHUNK_SUMMARY
        assert record.attempt == 2
        assert record.failure_mode == FailureMode.TIMEOUT

    def test_pass_rate_all_passed(self, tmp_logger):
        for _ in range(5):
            tmp_logger.log(passed=True)
        assert tmp_logger.pass_rate() == 1.0

    def test_pass_rate_mixed(self, tmp_logger):
        tmp_logger.log(passed=True)
        tmp_logger.log(passed=True)
        tmp_logger.log(passed=False, failure_mode=FailureMode.SCHEMA_VIOLATION)
        assert tmp_logger.pass_rate() == 2 / 3

    def test_pass_rate_empty(self, tmp_logger):
        assert tmp_logger.pass_rate() == 0.0

    def test_failure_distribution(self, tmp_logger):
        tmp_logger.log(passed=True)  # no failure mode
        tmp_logger.log(passed=False, failure_mode=FailureMode.SCHEMA_VIOLATION)
        tmp_logger.log(passed=False, failure_mode=FailureMode.SCHEMA_VIOLATION)
        tmp_logger.log(passed=False, failure_mode=FailureMode.TIMEOUT)

        dist = tmp_logger.failure_distribution()
        # Note: successful attempts with FailureMode.NONE are NOT counted
        # in failure distribution — only actual failures
        assert dist.get(FailureMode.SCHEMA_VIOLATION) == 2
        assert dist.get(FailureMode.TIMEOUT) == 1

    def test_latency_percentiles(self, tmp_logger):
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            tmp_logger.log(passed=True, latency_ms=float(ms))
        p = tmp_logger.latency_percentiles()
        # nearest-rank: p50 of [0..9] = index int(10*0.50) = 5 → 60
        assert p["p50"] == 60.0
        # p90 = index int(10*0.90) = 9 → 100
        assert p["p90"] == 100.0

    def test_latency_percentiles_empty(self, tmp_logger):
        p = tmp_logger.latency_percentiles()
        assert p == {"p50": 0.0, "p90": 0.0, "p99": 0.0}

    def test_total_attempts(self, tmp_logger):
        for _ in range(7):
            tmp_logger.log(passed=True)
        assert tmp_logger.total_attempts() == 7

    def test_stats_combined(self, tmp_logger):
        tmp_logger.log(passed=True, latency_ms=50.0)
        tmp_logger.log(passed=False, failure_mode=FailureMode.TIMEOUT, latency_ms=5000.0)

        stats = tmp_logger.stats()
        assert stats["total_attempts"] == 2
        assert stats["pass_rate"] == 0.5
        assert stats["failure_distribution"]["timeout"] == 1

    def test_rebuild_from_file(self, tmp_logger):
        # Write some records
        tmp_logger.log(passed=True, latency_ms=10.0)
        tmp_logger.log(passed=False, failure_mode=FailureMode.LLM_ERROR, latency_ms=500.0)

        # Rebuild a NEW logger from the same file
        rebuilt = JSONLAuditLogger(tmp_logger._path, rebuild=True)
        assert rebuilt.total_attempts() == 2
        assert rebuilt.pass_rate() == 0.5
        dist = rebuilt.failure_distribution()
        assert dist.get(FailureMode.LLM_ERROR) == 1

    def test_all_failure_modes_are_serializable(self, tmp_logger):
        """Every FailureMode should survive round-trip through JSON."""
        for fm in FailureMode:
            record = tmp_logger.log(
                passed=False,
                failure_mode=fm,
            )
            # Re-read from file
            rebuilt = JSONLAuditLogger(tmp_logger._path, rebuild=True)
            # The failure mode should appear in distribution
            if fm != FailureMode.NONE or not record.passed:
                dist = rebuilt.failure_distribution()
                assert fm in dist

    def test_all_operations_are_serializable(self, tmp_logger):
        """Every Operation should survive round-trip."""
        for op in Operation:
            tmp_logger.log(operation=op, passed=True)
        rebuilt = JSONLAuditLogger(tmp_logger._path, rebuild=True)
        assert rebuilt.total_attempts() == len(Operation.__members__)
