"""Tests for the two-sided binomial sign test."""

import pytest

from treeloom.domain.benchmark.significance import sign_test


def test_minilm_significant():
    """MiniLM 8W/23L should be significant (p ≈ 0.011)."""
    assert sign_test(8, 23) == pytest.approx(0.011, abs=0.002)


def test_jina_v2_tie():
    """jina-v2 15W/14L is effectively a tie (p == 1.0)."""
    assert sign_test(15, 14) == pytest.approx(1.0)


def test_zero_zero():
    """No observations: p == 1.0 (no evidence either way)."""
    assert sign_test(0, 0) == 1.0


def test_symmetry():
    """sign_test(wins, losses) == sign_test(losses, wins)."""
    assert sign_test(7, 19) == sign_test(19, 7)


def test_monotonicity():
    """A more extreme split yields a smaller p-value."""
    assert sign_test(5, 25) < sign_test(12, 18)
