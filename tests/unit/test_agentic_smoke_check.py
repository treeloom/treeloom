"""Unit tests for the Phase-4 smoke-fixture checker (scripts/agentic_smoke_check.py).

The checker is a script, not a package module — load it by path. Its pure
`check_run(rows, summary, arm)` is tested with synthetic rows shaped exactly
like agentic_runner._arm_row output (hit_cap IS stored per-row — verified
against both current and incident-era artifact files).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_CHECKER = Path(__file__).resolve().parents[2] / "scripts" / "agentic_smoke_check.py"
_spec = importlib.util.spec_from_file_location("agentic_smoke_check", _CHECKER)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _row(qid="q1", *, hit_cap=False, recall1=1.0, correctness=4,
         retrieved=("/repo/src/Workspace.cs",),
         final="The logic lives in Workspace.cs under src/."):
    return {
        "id": qid,
        "arms": {
            "treeloom": {
                "hit_cap": hit_cap,
                "recall@1": recall1,
                "retrieved_files": list(retrieved),
                "final_answer": final,
                "judge": {"correctness": correctness},
            }
        },
    }


GOOD_SUMMARY = {"served_models": ["deepseek-v4-flash"], "agent_protocol": "native_tools"}


def test_good_run_passes():
    rows = [_row("q1"), _row("q2"), _row("q3")]
    assert smoke.check_run(rows, GOOD_SUMMARY) == []


def test_one_capped_row_of_three_still_passes():
    rows = [_row("q1"), _row("q2", hit_cap=True), _row("q3")]
    assert smoke.check_run(rows, GOOD_SUMMARY) == []


def test_two_capped_rows_fail_termination():
    rows = [_row("q1", hit_cap=True), _row("q2", hit_cap=True), _row("q3")]
    fails = smoke.check_run(rows, GOOD_SUMMARY)
    assert any("termination" in f for f in fails)


def test_ungrounded_row_fails_grounding():
    # recall@1 == 1.0 but the answer cites a hallucinated wrong-repo path —
    # the 2026-06-25 sonnet fingerprint.
    rows = [
        _row("q1"),
        _row("q2", final="See packages/twenty-server/src/net/TcpServer.cpp"),
        _row("q3"),
    ]
    fails = smoke.check_run(rows, GOOD_SUMMARY)
    assert any("grounding" in f and "q2" in f for f in fails)


def test_recall_miss_rows_are_exempt_from_grounding():
    rows = [_row("q1", recall1=0.0, final="no idea"), _row("q2"), _row("q3")]
    assert smoke.check_run(rows, GOOD_SUMMARY) == []


def test_low_mean_correctness_fails():
    rows = [_row("q1", correctness=1), _row("q2", correctness=1),
            _row("q3", correctness=5)]  # mean 2.33 < 3.0
    fails = smoke.check_run(rows, GOOD_SUMMARY)
    assert any("correctness" in f for f in fails)


def test_missing_served_models_fails():
    rows = [_row("q1"), _row("q2"), _row("q3")]
    fails = smoke.check_run(rows, {"agent_protocol": "native_tools"})
    assert any("served_models" in f for f in fails)


def test_react_protocol_fails():
    rows = [_row("q1"), _row("q2"), _row("q3")]
    summary = {"served_models": ["m"], "agent_protocol": "react"}
    fails = smoke.check_run(rows, summary)
    assert any("agent_protocol" in f for f in fails)


def test_empty_rows_fail():
    assert smoke.check_run([], GOOD_SUMMARY) == ["no rows found"]


def test_missing_arm_reported():
    rows = [{"id": "q1", "arms": {"grep": {}}}]
    fails = smoke.check_run(rows, GOOD_SUMMARY)
    assert any("missing arm" in f for f in fails)


def test_grounding_uses_basename_not_full_path():
    # The answer cites just the basename; retrieved_files are absolute paths.
    rows = [_row("q1", retrieved=("/x/y/z/FlagService.cs",),
                 final="Handled by FlagService.cs")] * 1
    rows += [_row("q2"), _row("q3")]
    assert smoke.check_run(rows, GOOD_SUMMARY) == []
