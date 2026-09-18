"""
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

# The canonical implementation now lives in the domain module so it is
# importable as pure library code without depending on scripts/ being on the
# path.  Re-export it here so the CLI and any existing importers keep working.
from treeloom.domain.benchmark.significance import sign_test  # noqa: F401


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python significance.py <wins> <losses>", file=sys.stderr)
        sys.exit(1)
    wins = int(sys.argv[1])
    losses = int(sys.argv[2])
    print(sign_test(wins, losses))
