"""Unit tests for agent_metrics.aggregate() with significance reporting.

Detroit-style: no mocking of internals; all assertions on real outputs.
"""
from __future__ import annotations

import pytest

from treeloom.domain.benchmark.agent_metrics import aggregate
from treeloom.domain.benchmark.significance import sign_test, win_loss_tie


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_row(
    *,
    grep_correctness: float | None,
    grep_recall5: float | None,
    tl_correctness: float | None,
    tl_recall5: float | None,
    tokens_grep: int = 1000,
    tokens_tl: int = 500,
) -> dict:
    """Build a minimal per-query result row with grep and treeloom arms."""
    return {
        "arms": {
            "grep": {
                "total_tokens": tokens_grep,
                "prompt_tokens": tokens_grep,
                "completion_tokens": 0,
                "turns": 3,
                "tool_calls": 2,
                "latency_s": 1.0,
                "recall@1": 0.0,
                "recall@5": grep_recall5,
                "recall@10": 0.0,
                "mrr": 0.0,
                "judge": {"correctness": grep_correctness, "completeness": None},
            },
            "treeloom": {
                "total_tokens": tokens_tl,
                "prompt_tokens": tokens_tl,
                "completion_tokens": 0,
                "turns": 2,
                "tool_calls": 1,
                "latency_s": 0.5,
                "recall@1": 0.0,
                "recall@5": tl_recall5,
                "recall@10": 0.0,
                "mrr": 0.0,
                "judge": {"correctness": tl_correctness, "completeness": None},
            },
        }
    }


@pytest.fixture
def mixed_rows() -> list[dict]:
    """Five rows with a mix of treeloom wins, losses, and ties on correctness/recall@5."""
    return [
        # treeloom wins both
        _make_row(grep_correctness=0.5, tl_correctness=0.8, grep_recall5=0.0, tl_recall5=1.0),
        # treeloom wins both again
        _make_row(grep_correctness=0.4, tl_correctness=0.9, grep_recall5=0.0, tl_recall5=1.0),
        # tie on correctness, treeloom wins recall
        _make_row(grep_correctness=0.7, tl_correctness=0.7, grep_recall5=0.0, tl_recall5=1.0),
        # grep wins correctness, tie on recall
        _make_row(grep_correctness=0.9, tl_correctness=0.6, grep_recall5=1.0, tl_recall5=1.0),
        # grep wins both
        _make_row(grep_correctness=1.0, tl_correctness=0.3, grep_recall5=1.0, tl_recall5=0.0),
    ]


# ── win_loss_tie ──────────────────────────────────────────────────────────────

class TestWinLossTie:
    """Tests for the win_loss_tie helper."""

    def test_basic_counts(self, mixed_rows):
        """Verify win/loss/tie counts match manual inspection of fixture rows."""
        # correctness: rows 0,1=tl wins; row 2=tie; rows 3,4=grep wins
        w, l, t = win_loss_tie(mixed_rows, "treeloom", "grep", "correctness")
        assert w == 2
        assert l == 2
        assert t == 1

    def test_recall5_counts(self, mixed_rows):
        """recall@5: rows 0,1,2=tl wins; rows 3,4=tie/grep win."""
        w, l, t = win_loss_tie(mixed_rows, "treeloom", "grep", "recall@5")
        # row 0: tl 1.0 > grep 0.0 → win
        # row 1: tl 1.0 > grep 0.0 → win
        # row 2: tl 1.0 > grep 0.0 → win
        # row 3: tl 1.0 == grep 1.0 → tie
        # row 4: tl 0.0 < grep 1.0 → loss
        assert w == 3
        assert l == 1
        assert t == 1

    def test_missing_arm_skipped(self):
        """Rows missing one arm are excluded from counts."""
        rows = [
            {"arms": {"treeloom": {"recall@5": 1.0, "judge": {}}}},  # no grep
            {"arms": {"grep": {"recall@5": 0.0, "judge": {}}, "treeloom": {"recall@5": 1.0, "judge": {}}}},
        ]
        w, l, t = win_loss_tie(rows, "treeloom", "grep", "recall@5")
        assert w == 1
        assert l == 0
        assert t == 0

    def test_none_value_skipped(self):
        """Rows where either arm's metric is None are excluded."""
        rows = [
            {"arms": {"grep": {"judge": {"correctness": None}}, "treeloom": {"judge": {"correctness": 0.8}}}},
            {"arms": {"grep": {"judge": {"correctness": 0.5}}, "treeloom": {"judge": {"correctness": 0.9}}}},
        ]
        w, l, t = win_loss_tie(rows, "treeloom", "grep", "correctness")
        assert w == 1
        assert l == 0
        assert t == 0

    def test_judge_key_routing(self):
        """correctness and completeness must be read from arm['judge']."""
        rows = [
            {
                "arms": {
                    "a": {"correctness": 99, "judge": {"correctness": 0.2}},
                    "b": {"correctness": 99, "judge": {"correctness": 0.9}},
                }
            }
        ]
        # should use judge.correctness, so b wins over a
        w, l, t = win_loss_tie(rows, "a", "b", "correctness")
        assert w == 0
        assert l == 1
        assert t == 0

    def test_top_level_key_routing(self):
        """Non-judge keys (e.g. recall@5) are read from the arm top level."""
        rows = [
            {
                "arms": {
                    "a": {"recall@5": 1.0, "judge": {}},
                    "b": {"recall@5": 0.0, "judge": {}},
                }
            }
        ]
        w, l, t = win_loss_tie(rows, "a", "b", "recall@5")
        assert w == 1
        assert l == 0
        assert t == 0


# ── aggregate() — legacy comparison block ────────────────────────────────────

class TestAggregateComparisonBlock:
    """Tests for the legacy ``comparison`` block in aggregate()."""

    def test_correctness_win_rate_is_dict(self, mixed_rows):
        """correctness_win_rate_treeloom must be a dict with the right keys."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        cwr = result["comparison"]["correctness_win_rate_treeloom"]
        assert isinstance(cwr, dict)
        assert set(cwr.keys()) == {"win_rate", "wins", "losses", "ties", "p_value"}

    def test_correctness_win_rate_values(self, mixed_rows):
        """win_rate, wins, losses, ties must match manual counts."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        cwr = result["comparison"]["correctness_win_rate_treeloom"]
        # fixture: 2W 2L 1T (see TestWinLossTie.test_basic_counts)
        assert cwr["wins"] == 2
        assert cwr["losses"] == 2
        assert cwr["ties"] == 1
        assert cwr["win_rate"] == pytest.approx(2 / 5)

    def test_correctness_p_value_matches_sign_test(self, mixed_rows):
        """p_value must equal sign_test(wins, losses) for correctness."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        cwr = result["comparison"]["correctness_win_rate_treeloom"]
        assert cwr["p_value"] == pytest.approx(sign_test(cwr["wins"], cwr["losses"]))

    def test_recall5_win_rate_is_dict(self, mixed_rows):
        """recall@5_win_rate_treeloom must be a dict with the right keys."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        rwr = result["comparison"]["recall@5_win_rate_treeloom"]
        assert isinstance(rwr, dict)
        assert set(rwr.keys()) == {"win_rate", "wins", "losses", "ties", "p_value"}

    def test_recall5_p_value_matches_sign_test(self, mixed_rows):
        """p_value must equal sign_test(wins, losses) for recall@5."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        rwr = result["comparison"]["recall@5_win_rate_treeloom"]
        assert rwr["p_value"] == pytest.approx(sign_test(rwr["wins"], rwr["losses"]))

    def test_token_fields_unchanged(self, mixed_rows):
        """token_ratio and tokens_saved_pct must remain plain floats."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        comp = result["comparison"]
        assert isinstance(comp["token_ratio_treeloom_over_grep"], float)
        assert isinstance(comp["tokens_saved_pct"], float)


# ── aggregate() — generic comparisons block ──────────────────────────────────

class TestAggregateComparisonsBlock:
    """Tests for the generic ``comparisons`` block in aggregate()."""

    def test_treeloom_vs_grep_recall5_is_dict(self, mixed_rows):
        """recall@5_win_rate in comparisons must be a dict with the right keys."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        rwr = result["comparisons"]["treeloom_vs_grep"]["recall@5_win_rate"]
        assert isinstance(rwr, dict)
        assert set(rwr.keys()) == {"win_rate", "wins", "losses", "ties", "p_value"}

    def test_treeloom_vs_grep_recall5_p_value(self, mixed_rows):
        """p_value for recall@5 in comparisons must equal sign_test(wins, losses)."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        rwr = result["comparisons"]["treeloom_vs_grep"]["recall@5_win_rate"]
        assert rwr["p_value"] == pytest.approx(sign_test(rwr["wins"], rwr["losses"]))

    def test_treeloom_vs_grep_correctness_p_value(self, mixed_rows):
        """p_value for correctness in comparisons must equal sign_test(wins, losses)."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        cwr = result["comparisons"]["treeloom_vs_grep"]["correctness_win_rate"]
        assert cwr["p_value"] == pytest.approx(sign_test(cwr["wins"], cwr["losses"]))

    def test_comparison_and_comparisons_agree(self, mixed_rows):
        """Legacy comparison block and generic comparisons block must agree on counts."""
        result = aggregate(mixed_rows, ["grep", "treeloom"], repo="x", model="m")
        leg = result["comparison"]["correctness_win_rate_treeloom"]
        gen = result["comparisons"]["treeloom_vs_grep"]["correctness_win_rate"]
        assert leg["wins"] == gen["wins"]
        assert leg["losses"] == gen["losses"]
        assert leg["ties"] == gen["ties"]
        assert leg["p_value"] == pytest.approx(gen["p_value"])

    def test_no_grep_arm_skips_legacy_block(self):
        """Without 'grep' in arms, the legacy comparison block must be absent."""
        rows = [
            _make_row(
                grep_correctness=0.5, tl_correctness=0.8,
                grep_recall5=0.0, tl_recall5=1.0,
            )
        ]
        # Override: only treeloom and 'other'
        rows[0]["arms"]["other"] = rows[0]["arms"].pop("grep")
        result = aggregate(rows, ["other", "treeloom"], repo="x", model="m")
        assert "comparison" not in result
        assert "treeloom_vs_other" in result["comparisons"]

    def test_zero_zero_p_value_is_one(self):
        """When both arms always tie (wins==losses==0), p_value must be 1.0."""
        rows = [
            _make_row(
                grep_correctness=0.7, tl_correctness=0.7,
                grep_recall5=1.0, tl_recall5=1.0,
            )
        ]
        result = aggregate(rows, ["grep", "treeloom"], repo="x", model="m")
        cwr = result["comparison"]["correctness_win_rate_treeloom"]
        assert cwr["wins"] == 0
        assert cwr["losses"] == 0
        assert cwr["p_value"] == 1.0


# ── zero-match guard ──────────────────────────────────────────────────────

from treeloom.domain.benchmark.agent_metrics import detect_zero_match_arms  # noqa: E402


def _arm_with_reads(*, reads: list[str], recall5: float, correctness: float = 0.5) -> dict:
    return {
        "total_tokens": 500, "prompt_tokens": 500, "completion_tokens": 0,
        "turns": 2, "tool_calls": 1, "latency_s": 0.5,
        "recall@1": recall5, "recall@5": recall5, "recall@10": recall5, "mrr": 0.0,
        "retrieved_files": reads,
        "judge": {"correctness": correctness, "completeness": None},
    }


class TestZeroMatchGuard:
    """an arm that reads files but never matches a ground-truth file across
    the whole run is a path/GT normalization bug, not a bad retriever."""

    def test_guard_fires_when_reads_but_zero_matches(self):
        rows = [
            {"arms": {"grep": _arm_with_reads(reads=["a.py", "b.py"], recall5=0.0)}},
            {"arms": {"grep": _arm_with_reads(reads=["c.py"], recall5=0.0)}},
        ]
        with pytest.warns(UserWarning, match="ZERO ground-truth"):
            suspects = detect_zero_match_arms(rows, ["grep"])
        assert len(suspects) == 1
        assert suspects[0]["arm"] == "grep"
        assert suspects[0]["total_reads"] == 3
        assert suspects[0]["total_matches"] == 0

    def test_guard_silent_when_any_match(self):
        rows = [
            {"arms": {"grep": _arm_with_reads(reads=["a.py"], recall5=0.0)}},
            {"arms": {"grep": _arm_with_reads(reads=["b.py"], recall5=1.0)}},
        ]
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning would raise
            suspects = detect_zero_match_arms(rows, ["grep"])
        assert suspects == []

    def test_guard_silent_when_no_reads(self):
        rows = [{"arms": {"grep": _arm_with_reads(reads=[], recall5=0.0)}}]
        suspects = detect_zero_match_arms(rows, ["grep"])
        assert suspects == []

    def test_aggregate_surfaces_warning_in_summary(self):
        rows = [
            {"arms": {
                "grep": _arm_with_reads(reads=["a.py"], recall5=0.0),
                "treeloom": _arm_with_reads(reads=["b.py"], recall5=0.0),
            }},
        ]
        with pytest.warns(UserWarning):
            result = aggregate(rows, ["grep", "treeloom"], repo="three.js", model="m")
        assert "zero_match_warning" in result
        flagged = {s["arm"] for s in result["zero_match_warning"]}
        assert flagged == {"grep", "treeloom"}
