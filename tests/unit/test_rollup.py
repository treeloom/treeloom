"""Unit tests for benchmark rollup — Detroit-style, no mocks, no I/O."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from treeloom.domain.benchmark.rollup import (
    format_comparison_table,
    format_rollup_table,
    multi_repo_rollup,
)
from treeloom.domain.benchmark.significance import sign_test


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_summary(
    repo: str,
    n_queries: int,
    grep_tokens: float,
    tl_tokens: float,
    grep_recall5: float,
    tl_recall5: float,
    grep_corr: float,
    tl_corr: float,
    corr_wins: int,
    corr_losses: int,
    corr_ties: int,
    rec5_wins: int,
    rec5_losses: int,
    rec5_ties: int,
) -> dict:
    """Build a fake agentic summary dict in the Task-A dict win_rate format."""
    c_total = corr_wins + corr_losses + corr_ties
    r_total = rec5_wins + rec5_losses + rec5_ties
    tokens_saved_pct = (grep_tokens - tl_tokens) / grep_tokens * 100 if grep_tokens else 0.0
    token_ratio = tl_tokens / grep_tokens if grep_tokens else 0.0
    return {
        "repo": repo,
        "model": "test-model",
        "n_queries": n_queries,
        "arms": ["grep", "treeloom"],
        "grep": {
            "mean_total_tokens": grep_tokens,
            "mean_recall@5": grep_recall5,
            "mean_correctness": grep_corr,
        },
        "treeloom": {
            "mean_total_tokens": tl_tokens,
            "mean_recall@5": tl_recall5,
            "mean_correctness": tl_corr,
        },
        "comparison": {
            "token_ratio_treeloom_over_grep": round(token_ratio, 4),
            "tokens_saved_pct": round(tokens_saved_pct, 2),
            "correctness_win_rate_treeloom": {
                "win_rate": round(corr_wins / c_total, 4) if c_total else 0.0,
                "wins": corr_wins,
                "losses": corr_losses,
                "ties": corr_ties,
                "p_value": sign_test(corr_wins, corr_losses),
            },
            "recall@5_win_rate_treeloom": {
                "win_rate": round(rec5_wins / r_total, 4) if r_total else 0.0,
                "wins": rec5_wins,
                "losses": rec5_losses,
                "ties": rec5_ties,
                "p_value": sign_test(rec5_wins, rec5_losses),
            },
        },
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def summary_a():
    return _make_summary(
        repo="repo-a",
        n_queries=10,
        grep_tokens=20000.0,
        tl_tokens=18000.0,
        grep_recall5=0.8,
        tl_recall5=0.9,
        grep_corr=3.5,
        tl_corr=4.0,
        corr_wins=6,
        corr_losses=2,
        corr_ties=2,
        rec5_wins=5,
        rec5_losses=3,
        rec5_ties=2,
    )


@pytest.fixture
def summary_b():
    return _make_summary(
        repo="repo-b",
        n_queries=20,
        grep_tokens=15000.0,
        tl_tokens=12000.0,
        grep_recall5=0.7,
        tl_recall5=0.75,
        grep_corr=3.0,
        tl_corr=3.2,
        corr_wins=10,
        corr_losses=5,
        corr_ties=5,
        rec5_wins=8,
        rec5_losses=7,
        rec5_ties=5,
    )


# ── multi_repo_rollup ────────────────────────────────────────────────────────


class TestMultiRepoRollup:
    def test_pooled_treeloom_tokens_weighted_average(self, summary_a, summary_b):
        """Pooled treeloom tokens = n_queries-weighted average."""
        rollup = multi_repo_rollup([summary_a, summary_b])
        pooled = rollup["pooled"]

        # n_a=10, tl_a=18000; n_b=20, tl_b=12000; total=30
        expected = (18000.0 * 10 + 12000.0 * 20) / 30
        assert pooled["treeloom_mean_total_tokens"] == round(expected, 4)

    def test_pooled_grep_tokens_weighted_average(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        pooled = rollup["pooled"]
        expected = (20000.0 * 10 + 15000.0 * 20) / 30
        assert pooled["grep_mean_total_tokens"] == round(expected, 4)

    def test_pooled_correctness_p_value_equals_sign_test_of_summed_counts(
        self, summary_a, summary_b
    ):
        """Pooled correctness p_value must equal sign_test(sum_wins, sum_losses)."""
        rollup = multi_repo_rollup([summary_a, summary_b])
        pooled = rollup["pooled"]

        # a: 6 wins, 2 losses; b: 10 wins, 5 losses
        total_wins = 6 + 10
        total_losses = 2 + 5
        expected_p = sign_test(total_wins, total_losses)
        assert pooled["correctness"]["p_value"] == round(expected_p, 4)

    def test_pooled_recall5_p_value_equals_sign_test_of_summed_counts(
        self, summary_a, summary_b
    ):
        rollup = multi_repo_rollup([summary_a, summary_b])
        pooled = rollup["pooled"]
        # a: 5 wins, 3 losses; b: 8 wins, 7 losses
        total_wins = 5 + 8
        total_losses = 3 + 7
        expected_p = sign_test(total_wins, total_losses)
        assert pooled["recall@5"]["p_value"] == round(expected_p, 4)

    def test_pooled_win_loss_tie_counts_are_summed(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        pooled = rollup["pooled"]
        assert pooled["correctness"]["wins"] == 6 + 10
        assert pooled["correctness"]["losses"] == 2 + 5
        assert pooled["correctness"]["ties"] == 2 + 5

    def test_n_repos_and_total_queries(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        assert rollup["pooled"]["n_repos"] == 2
        assert rollup["pooled"]["total_queries"] == 30

    def test_repos_list_length_matches_input(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        assert len(rollup["repos"]) == 2

    def test_repo_row_names_preserved(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        names = [r["repo"] for r in rollup["repos"]]
        assert "repo-a" in names
        assert "repo-b" in names

    def test_tokens_saved_pct_recomputed_from_pooled_means(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        pooled = rollup["pooled"]
        grep_tok = pooled["grep_mean_total_tokens"]
        tl_tok = pooled["treeloom_mean_total_tokens"]
        expected_saved = round((grep_tok - tl_tok) / grep_tok * 100, 2)
        assert pooled["tokens_saved_pct"] == expected_saved

    def test_single_summary_rollup(self, summary_a):
        """Single-summary rollup degenerates cleanly."""
        rollup = multi_repo_rollup([summary_a])
        pooled = rollup["pooled"]
        assert pooled["n_repos"] == 1
        assert pooled["total_queries"] == 10
        assert pooled["treeloom_mean_total_tokens"] == 18000.0

    def test_empty_summaries(self):
        rollup = multi_repo_rollup([])
        assert rollup["repos"] == []
        assert rollup["pooled"]["n_repos"] == 0
        assert rollup["pooled"]["total_queries"] == 0

    def test_summary_without_comparison_block_included(self):
        """Summaries lacking a comparison block are included but contribute 0 counts."""
        s = {
            "repo": "no-comp",
            "n_queries": 5,
            "arms": ["treeloom"],
            "treeloom": {
                "mean_total_tokens": 10000.0,
                "mean_recall@5": 0.6,
                "mean_correctness": 3.0,
            },
        }
        rollup = multi_repo_rollup([s])
        assert len(rollup["repos"]) == 1
        assert rollup["repos"][0]["repo"] == "no-comp"
        # No wins/losses from a missing comparison block
        assert rollup["pooled"]["correctness"]["wins"] == 0
        assert rollup["pooled"]["correctness"]["p_value"] == 1.0


# ── format_comparison_table ──────────────────────────────────────────────────


class TestFormatComparisonTable:
    def test_table_contains_pipe_characters(self, summary_a):
        table = format_comparison_table(summary_a)
        assert "|" in table

    def test_table_contains_arm_names(self, summary_a):
        table = format_comparison_table(summary_a)
        assert "grep" in table
        assert "treeloom" in table

    def test_table_contains_p_value(self, summary_a):
        table = format_comparison_table(summary_a)
        # p-value column should contain a numeric value
        assert "0." in table or "1.0" in table

    def test_table_contains_recall5_row(self, summary_a):
        table = format_comparison_table(summary_a)
        assert "recall@5" in table

    def test_table_contains_correctness_row(self, summary_a):
        table = format_comparison_table(summary_a)
        assert "correctness" in table

    def test_table_contains_tokens_row(self, summary_a):
        table = format_comparison_table(summary_a)
        assert "tokens" in table.lower()

    def test_no_comparison_block_returns_note(self):
        s = {"repo": "x", "n_queries": 5, "arms": ["treeloom"]}
        result = format_comparison_table(s)
        assert "grep-vs-treeloom" in result.lower() or "comparison" in result.lower()
        assert "|" not in result  # Not a table — just a note

    def test_legacy_flat_win_rate_handled_gracefully(self):
        """Old-format summaries with float win_rate values don't crash."""
        s = {
            "repo": "legacy",
            "n_queries": 10,
            "arms": ["grep", "treeloom"],
            "grep": {"mean_total_tokens": 10000.0, "mean_recall@5": 0.8, "mean_correctness": 3.5},
            "treeloom": {"mean_total_tokens": 8000.0, "mean_recall@5": 0.85, "mean_correctness": 3.8},
            "comparison": {
                "token_ratio_treeloom_over_grep": 0.8,
                "tokens_saved_pct": 20.0,
                "correctness_win_rate_treeloom": 0.6,  # legacy flat float
                "recall@5_win_rate_treeloom": 0.55,
            },
        }
        table = format_comparison_table(s)
        assert "|" in table
        assert "grep" in table


# ── format_rollup_table ──────────────────────────────────────────────────────


class TestFormatRollupTable:
    def test_contains_pooled_row(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        table = format_rollup_table(rollup)
        assert "Pooled" in table

    def test_contains_one_row_per_repo(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        table = format_rollup_table(rollup)
        assert "repo-a" in table
        assert "repo-b" in table

    def test_contains_pipe_characters(self, summary_a, summary_b):
        rollup = multi_repo_rollup([summary_a, summary_b])
        table = format_rollup_table(rollup)
        assert "|" in table

    def test_single_repo_table_has_pooled_row(self, summary_a):
        rollup = multi_repo_rollup([summary_a])
        table = format_rollup_table(rollup)
        assert "Pooled" in table
        assert "repo-a" in table

    def test_empty_rollup_still_has_pooled_row(self):
        rollup = multi_repo_rollup([])
        table = format_rollup_table(rollup)
        assert "Pooled" in table
