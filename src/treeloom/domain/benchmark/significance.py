"""Benchmark domain: two-sided binomial sign test + per-query win/loss counting — pure, no I/O.

Two-sided binomial sign test for comparing per-query win/loss splits.

Usage
-----
Call sign_test(wins, losses) where wins and losses are counts of queries where
one arm was better than the other.  Ties are excluded before calling this
function (they carry no directional information).

The sign test asks: if there were truly no difference, what is the probability
of seeing a split this extreme or more extreme?  Under the null hypothesis,
each resolved query is equally likely to go either way (p=0.5), so the number
of wins follows Binomial(n, 0.5) where n = wins + losses.

Interpreting the result
-----------------------
- p < 0.05: the split is statistically significant — reject the null of no
  difference at the 5% level.
- p >= 0.05 ("ns", not significant): the split is *consistent with no
  difference* at this sample size.  This does NOT prove the two arms are
  equal — it means the evidence is insufficient at n to distinguish a real
  effect from noise.  The point estimates (mean scores, win rates) remain the
  best directional guess; significance tests only speak to how confidently we
  can rule out chance.

At n=100 with ~45 ties excluded (n_resolved ~55), the test has low power for
small true differences.  A split of 28W/18L (p=0.18) could easily arise by
chance even if one arm is genuinely a few points better.
"""
from __future__ import annotations

from math import comb


def sign_test(wins: int, losses: int) -> float:
    """Return the two-sided binomial sign-test p-value.

    Parameters
    ----------
    wins:
        Number of queries where the arm of interest scored higher.
    losses:
        Number of queries where the other arm scored higher.

    Returns
    -------
    float
        Two-sided p-value in [0, 1].  Returns 1.0 when wins == losses == 0.

    Notes
    -----
    Ties must be excluded by the caller before passing counts here.
    The test statistic is k = max(wins, losses); we compute the probability
    of seeing k or more wins (or losses) under Binomial(n=wins+losses, p=0.5)
    and double it for the two-sided test, capped at 1.0.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    two_n = 1 << n  # 2**n as integer for exact arithmetic
    tail = sum(comb(n, i) for i in range(k, n + 1))
    p = 2 * tail / two_n
    return min(p, 1.0)


def _field(arm: dict, key: str):
    """Resolve a metric value from an arm result dict.

    Keys ``correctness`` and ``completeness`` live under the arm's ``judge``
    sub-dict; everything else is top-level.  Mirrors the rule in
    ``agent_metrics._field`` without creating a circular import.
    """
    if key in ("correctness", "completeness"):
        return arm.get("judge", {}).get(key)
    return arm.get(key)


def win_loss_tie(
    rows: list[dict],
    chal: str,
    base: str,
    key: str,
) -> tuple[int, int, int]:
    """Count per-query wins, losses, and ties for ``chal`` vs ``base`` on ``key``.

    Parameters
    ----------
    rows:
        List of per-query result dicts, each shaped
        ``{"arms": {armname: {...}}}``.
    chal:
        Challenger arm name.
    base:
        Baseline arm name.
    key:
        Metric key to compare.  ``correctness``/``completeness`` are read from
        each arm's ``judge`` sub-dict; all other keys are top-level on the arm.

    Returns
    -------
    tuple[int, int, int]
        ``(wins, losses, ties)`` where a *win* means ``chal`` strictly beat
        ``base``, a *loss* means ``base`` strictly beat ``chal``, and a *tie*
        means both values were equal and non-None.  Rows where either arm is
        absent or the metric value is ``None`` are skipped entirely.
    """
    wins = losses = ties = 0
    for r in rows:
        arms = r.get("arms", {})
        if chal not in arms or base not in arms:
            continue
        vc = _field(arms[chal], key)
        vb = _field(arms[base], key)
        if vc is None or vb is None:
            continue
        if vc > vb:
            wins += 1
        elif vc < vb:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties
