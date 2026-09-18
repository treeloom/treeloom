"""Unit tests for preflight (scanner, model, recommender).

No live DB or filesystem dependencies — uses tempfile + Detroit-style fakes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from treeloom.preflight.coefficients import Coefficients, fallback_coefficients
from treeloom.preflight.model import estimate_total, estimate_file
from treeloom.preflight.recommender import build_report
from treeloom.preflight.scanner import FileStat, walk_repo


def _write(root: Path, rel: str, content: str = "x\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


@pytest.fixture
def tmp_repo():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_walk_repo_filters_to_supported_extensions(tmp_repo):
    _write(tmp_repo, "src/a.py", "print(1)\nprint(2)\n")
    _write(tmp_repo, "src/b.ts", "const x = 1;\n")
    _write(tmp_repo, "src/c.notreal", "garbage")
    stats, unsupported = walk_repo(str(tmp_repo), use_gitignore=False)
    rels = {s.rel_path for s in stats}
    assert "src/a.py" in rels
    assert "src/b.ts" in rels
    assert "src/c.notreal" not in rels
    assert unsupported == 1


def test_walk_repo_skips_node_modules_by_default(tmp_repo):
    _write(tmp_repo, "src/a.py", "x\n")
    _write(tmp_repo, "node_modules/lib/x.js", "x\n")
    stats, _ = walk_repo(str(tmp_repo), use_gitignore=False)
    rels = {s.rel_path for s in stats}
    assert "src/a.py" in rels
    assert all("node_modules" not in r for r in rels)


def test_walk_repo_honors_skip_patterns(tmp_repo):
    _write(tmp_repo, "src/a.js", "x\n")
    _write(tmp_repo, "src/a.min.js", "x\n")
    stats, _ = walk_repo(
        str(tmp_repo), use_gitignore=False, skip_patterns=("*.min.js",),
    )
    rels = {s.rel_path for s in stats}
    assert "src/a.js" in rels
    assert "src/a.min.js" not in rels


def test_walk_repo_collects_loc(tmp_repo):
    _write(tmp_repo, "src/a.py", "a\nb\nc\n")
    stats, _ = walk_repo(str(tmp_repo), use_gitignore=False)
    assert stats[0].loc == 3


def test_estimate_file_seconds_increase_with_summary_on():
    s = FileStat(path="x", rel_path="x.py", size_bytes=10_000,
                 loc=200, language="python", max_line_length=80)
    est = estimate_file(s, fallback_coefficients())
    assert est.seconds_summary_on > est.seconds_summary_off > 0


def test_estimate_total_concurrency_divides_wall_time():
    s = FileStat(path="x", rel_path="x.py", size_bytes=10_000,
                 loc=200, language="python", max_line_length=0)
    files = [s] * 16
    _, t8 = estimate_total(files, fallback_coefficients(), file_concurrency=8)
    _, t1 = estimate_total(files, fallback_coefficients(), file_concurrency=1)
    # concurrency=8 should be ~8x faster, modulo the constant warmup term
    assert t8.wall_seconds_summary_on < t1.wall_seconds_summary_on / 4


def test_build_report_red_when_summary_off_under_6h_and_on_over_24h():
    # Construct stats that will produce >24h summary_on, <6h summary_off
    # by inflating coefficients
    coefs = fallback_coefficients()
    coefs.global_["llm_s_per_chunk"] = 100.0  # absurdly slow LLM
    stats = [
        FileStat(path=f"x{i}", rel_path=f"x{i}.py", size_bytes=50_000,
                 loc=1000, language="python", max_line_length=0)
        for i in range(2000)
    ]
    per_file, total = estimate_total(stats, coefs, file_concurrency=8)
    report = build_report("/x", stats, 0, per_file, total)
    assert report.verdict == "red"
    assert any(w.code == "SUMMARY_VECTOR_BLOWS_BUDGET" for w in report.warnings)
    assert report.recommendations.env.get("USE_SUMMARY_VECTOR") == "0"


def test_build_report_green_for_small_repo():
    stats = [
        FileStat(path="a.py", rel_path="a.py", size_bytes=1000,
                 loc=30, language="python", max_line_length=0),
    ]
    per_file, total = estimate_total(stats, fallback_coefficients(), file_concurrency=8)
    report = build_report("/x", stats, 0, per_file, total)
    assert report.verdict in ("green", "yellow")  # tiny repo, definitely not red


def test_build_report_refuses_empty_dir():
    per_file, total = estimate_total([], fallback_coefficients(), file_concurrency=8)
    report = build_report("/empty", [], 0, per_file, total)
    assert report.verdict == "red"
    assert any(w.code == "NO_SUPPORTED_FILES" for w in report.warnings)


def test_build_report_skip_patterns_gated_by_feature_flag(monkeypatch):
    """When flag is off, recommendation contains no skip_patterns (only
    informational warning); when on, skip_patterns are populated."""
    stats = [FileStat(path="package-lock.json", rel_path="package-lock.json",
                      size_bytes=1_000_000, loc=10000,
                      language="json", max_line_length=0)]
    per_file, total = estimate_total(stats, fallback_coefficients(), file_concurrency=8)

    monkeypatch.setenv("TREELOOM_FEATURE_SKIP_PATTERNS", "0")
    r_off = build_report("/x", stats, 0, per_file, total, feature_skip_patterns=False)
    assert r_off.recommendations.skip_patterns == []
    # warning still surfaces the suggestion
    assert any(w.code == "NOISY_FILES_DETECTED" for w in r_off.warnings)

    r_on = build_report("/x", stats, 0, per_file, total, feature_skip_patterns=True)
    assert r_on.recommendations.skip_patterns  # non-empty


def test_build_report_includes_two_pass_strategy_when_summary_off_over_yellow():
    # Inflate per-file cost so even summary_off exceeds the 6h yellow threshold.
    coefs = fallback_coefficients()
    coefs.global_["graph_s_per_file_const"] = 20.0
    stats = [
        FileStat(path=f"x{i}", rel_path=f"x{i}.py", size_bytes=10_000,
                 loc=200, language="python", max_line_length=0)
        for i in range(20_000)
    ]
    per_file, total = estimate_total(stats, coefs, file_concurrency=8)
    assert total.wall_seconds_summary_off > 6 * 3600  # sanity check
    report = build_report("/x", stats, 0, per_file, total)
    assert any(w.code == "STRATEGY_TWO_PASS" for w in report.warnings)
    assert report.recommendations.env.get("strategy") == "two_pass"
