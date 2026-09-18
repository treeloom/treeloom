"""Equivalent ground-truth files: finding EITHER counts as finding the answer.

A C# partial class split across Foo.cs and Foo.Log.cs is one class. The
schema's `additional_relevant_files` cannot say that: recall is FRACTIONAL
(hits / len(relevant)), so a sibling listed there means "find BOTH" -- it
halves the score of an agent that found the labelled file and caps recall@1
at 0.5. `relevant_file_alternatives` canonicalises the retrieved side
instead, so the denominator is untouched.

The end-to-end tests drive the REAL agentic scoring path with only the LLM
stubbed, because the failure this guards against was two scorers (`run` and
`agentic`) quietly disagreeing about the same query.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from treeloom.application.benchmark import agentic_runner as ar
from treeloom.application.benchmark.agent_loop import AgentRunResult
from treeloom.application.benchmark.judge import JudgeScore
from treeloom.domain.benchmark.agent_metrics import TokenAccount
from treeloom.domain.benchmark.metrics import canonicalize_retrieved, recall_at_k

LOG, MAIN, OTHER = "/r/X.Log.cs", "/r/X.cs", "/r/Y.cs"
ALT = {LOG: [MAIN]}


class TestCanonicalize:
    def test_alternative_scores_like_the_labelled_file(self):
        assert recall_at_k(canonicalize_retrieved([MAIN], ALT), [LOG], 1) == 1.0

    def test_labelled_file_is_not_penalised(self):
        assert recall_at_k(canonicalize_retrieved([LOG], ALT), [LOG], 1) == 1.0

    def test_both_count_once_and_take_one_rank_slot(self):
        assert canonicalize_retrieved([MAIN, LOG, OTHER], ALT) == [LOG, OTHER]

    def test_unrelated_file_still_misses(self):
        assert recall_at_k(canonicalize_retrieved([OTHER], ALT), [LOG], 1) == 0.0

    @pytest.mark.parametrize("alts", [None, {}])
    def test_no_alternatives_is_a_no_op(self, alts):
        assert canonicalize_retrieved([MAIN, OTHER], alts) == [MAIN, OTHER]

    def test_why_additional_relevant_files_was_the_wrong_tool(self):
        """Pin the failure mode, so the tests above mean something."""
        assert recall_at_k([LOG], [LOG, MAIN], 1) == 0.5
        assert recall_at_k([MAIN], [LOG, MAIN], 1) == 0.5


def _run(retrieved, q_extra):
    """Drive run_one_query with the agent and judge stubbed out."""
    rr = AgentRunResult(final_answer="it does X", turns=2, tool_calls=1,
                        token_account=TokenAccount(), retrieved_files=retrieved)
    q = {"id": "q1", "query": "what does X do?", "relevant_files": [LOG], **q_extra}
    with patch.object(ar, "_run_arm", new=AsyncMock(return_value=rr)), \
         patch.object(ar, "judge_answer",
                      new=AsyncMock(return_value=JudgeScore(4, 4, "ok", parse_ok=True))):
        row = asyncio.run(ar.run_one_query(
            q, {"gold_answer": "it does X"}, repo="/r", search_url="http://x",
            agent_llm=object(), judge_llm=object(), arms=["grep"], max_turns=3))
    return row["arms"]["grep"]


class TestAgenticScoringEndToEnd:
    def test_main_file_gets_full_credit_with_alternatives(self):
        arm = _run([MAIN], {"relevant_file_alternatives": ALT})
        assert arm["recall@1"] == 1.0 and arm["mrr"] == 1.0

    def test_main_file_scored_zero_without_them(self):
        """The situation before this change."""
        assert _run([MAIN], {})["recall@1"] == 0.0

    def test_additional_relevant_files_now_count_in_agentic(self):
        """Consistency with runner.py, which always unioned them."""
        arm = _run([OTHER], {"additional_relevant_files": [OTHER]})
        assert arm["recall@5"] > 0
