"""Guard for truncated symbol-free benchmark queries.

`_behavioral_query` capped completions at 120 tokens, which cut detailed
behavioral descriptions off mid-word. "Given a rule identifier and two
versions of" was generated and KEPT -- an unanswerable row that poisons a
query set exactly like a non-discriminating entity name does. The only
symptom is a quietly depressed score for every arm, which reads as a real
result rather than a data defect.
"""

import pytest


class TestSymbolFreeQueriesAreNotTruncated:
    """A token-budget cutoff produced unanswerable benchmark rows.

    `_behavioral_query` capped completions at 120 tokens, which cut detailed
    behavioral descriptions mid-word -- "Given a rule identifier and two
    versions of" was generated and kept. That poisons a query set exactly like
    a non-discriminating entity name does, and the only symptom is a quietly
    lower score for every arm.
    """

    def test_truncated_outputs_are_rejected(self):
        from treeloom.application.benchmark.llm_query_gen import _looks_truncated
        for bad in ["Given a rule identifier and two versions of",
                    "...drops the invalid ones, dedu",
                    "under 128 characters, and match al",
                    "", "   "]:
            assert _looks_truncated(bad) is True, bad

    def test_complete_questions_are_kept(self):
        from treeloom.application.benchmark.llm_query_gen import _looks_truncated
        for good in ["How does the class validate its fields?",
                     "It stores an injected reference to a view container.",
                     "Which action receives a batch of payloads over HTTP POST!"]:
            assert _looks_truncated(good) is False, good

    def test_budget_is_large_enough_for_a_detailed_query(self):
        """The observed truncations were ~220 chars; 120 tokens was too tight."""
        import inspect
        from treeloom.application.benchmark import llm_query_gen
        src = inspect.getsource(llm_query_gen._behavioral_query)
        assert "max_tokens=120" not in src, "the 120-token cap truncated queries"
