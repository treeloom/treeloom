"""Hand-computed tests for the nDCG@k / Average-Precision ranking metrics."""
import math

import pytest

from treeloom.domain.benchmark.metrics import average_precision, ndcg_at_k


class TestNDCG:
    def test_rank1_hit_single_relevant_is_perfect(self):
        assert ndcg_at_k(["a", "b", "c"], ["a"], 10) == 1.0

    def test_relevant_at_rank2_is_discounted(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        assert ndcg_at_k(["x", "a"], ["a"], 10) == pytest.approx(1 / math.log2(3))

    def test_no_relevant_retrieved_is_zero(self):
        assert ndcg_at_k(["x", "y"], ["a"], 10) == 0.0

    def test_no_ground_truth_is_zero(self):
        assert ndcg_at_k(["a", "b"], [], 10) == 0.0

    def test_two_relevant_partial_ranking(self):
        # retrieved a(1), x(2), b(3); relevant {a,b}
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309
        dcg = 1.0 + 0.5
        idcg = 1.0 + 1 / math.log2(3)
        assert ndcg_at_k(["a", "x", "b"], ["a", "b"], 10) == pytest.approx(dcg / idcg)

    def test_k_truncates_pool(self):
        # relevant item is at rank 2 but k=1 -> not seen -> 0
        assert ndcg_at_k(["x", "a"], ["a"], 1) == 0.0

    def test_ideal_ranking_scores_one(self):
        assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 10) == 1.0


class TestAveragePrecision:
    def test_rank1_hit_single_relevant(self):
        assert average_precision(["a", "b"], ["a"], 10) == 1.0

    def test_relevant_at_rank2(self):
        assert average_precision(["x", "a"], ["a"], 10) == pytest.approx(0.5)

    def test_two_relevant_ranks_1_and_3(self):
        # precision@1 = 1/1, precision@3 = 2/3; AP = (1 + 2/3)/2
        assert average_precision(["a", "x", "b"], ["a", "b"], 10) == pytest.approx(
            (1.0 + 2 / 3) / 2)

    def test_no_ground_truth_is_zero(self):
        assert average_precision(["a", "b"], [], 10) == 0.0

    def test_no_hit_is_zero(self):
        assert average_precision(["x", "y"], ["a"], 10) == 0.0

    def test_k_truncates_pool(self):
        assert average_precision(["x", "a"], ["a"], 1) == 0.0
