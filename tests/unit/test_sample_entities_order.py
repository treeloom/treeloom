"""A partial generation run must be a representative sample.

sample_entities() drew a uniform SET, then sorted it back into file_path
order, so generation swept the tree alphabetically. Any prefix of a run was
biased: on featbit v5.4.9 the first 64% of a symbol-free run held ZERO
TypeScript queries, because all 1,018 front-end entities sorted after the C#
back-end. With generation now checkpointed, stopping early is a real option,
so every prefix has to stand on its own.
"""

import random

import pytest

from treeloom.application.benchmark.enhanced_query_gen import sample_entities


def _pool():
    """Path-ordered like list_entities: 700 'back-end', then 300 'front-end'."""
    return ([{"id": f"b{i}", "file_path": f"/r/back-end/{i:04d}.cs"} for i in range(700)]
            + [{"id": f"f{i}", "file_path": f"/r/front-end/{i:04d}.ts"} for i in range(300)])


def _old(entities, k, seed):
    rng = random.Random(seed)
    return [entities[i] for i in sorted(rng.sample(range(len(entities)), k))]


class TestSampleSetIsUnchanged:
    def test_full_run_covers_the_same_entities_as_before(self):
        """Reproducibility of the SET: only the order may change."""
        pool = _pool()
        assert ({e["id"] for e in sample_entities(pool, 400, 0)}
                == {e["id"] for e in _old(pool, 400, 0)})

    def test_deterministic_for_a_seed(self):
        pool = _pool()
        assert sample_entities(pool, 400, 7) == sample_entities(pool, 400, 7)

    def test_different_seeds_differ(self):
        pool = _pool()
        assert sample_entities(pool, 400, 1) != sample_entities(pool, 400, 2)


class TestEveryPrefixIsRepresentative:
    @pytest.mark.parametrize("pct", [10, 25, 50, 64])
    def test_front_end_appears_in_every_prefix(self, pct):
        s = sample_entities(_pool(), 500, 0)
        n = len(s) * pct // 100
        share = sum(1 for e in s[:n] if "front-end" in e["file_path"]) / n
        assert 0.18 <= share <= 0.42, (
            f"first {pct}% has {share:.0%} front-end vs ~30% overall -- "
            "the prefix is clustered, as the old path-sorted order was")

    def test_old_order_really_was_biased(self):
        """Pin the failure this fixes, so the test above means something."""
        s = _old(_pool(), 500, 0)
        n = len(s) * 64 // 100
        assert sum(1 for e in s[:n] if "front-end" in e["file_path"]) == 0


class TestSmallPool:
    def test_pool_smaller_than_k_is_complete_but_shuffled(self):
        """This branch returned the pool in path order -- the same bias."""
        pool = _pool()[:50]
        s = sample_entities(pool, 500, 0)
        assert {e["id"] for e in s} == {e["id"] for e in pool}
        assert [e["id"] for e in s] != [e["id"] for e in pool]
