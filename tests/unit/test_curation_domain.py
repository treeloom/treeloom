"""Detroit-style tests for domain/benchmark/curation.py (pure, no I/O)."""
from treeloom.domain.benchmark.curation import apply_verdicts


def _row(qid, query="what does the widget do"):
    return {"id": qid, "query": query, "relevant_files": ["/repo/a.ts"],
            "strategy": "llm_symbol_free", "difficulty": "hard"}


class TestApplyVerdicts:
    def test_approve_keeps_row_and_stamps_curation(self):
        rows = [_row("q1")]
        v2, stats = apply_verdicts(
            rows,
            {"q1": {"verdict": "approve", "reviewer": "rodger",
                    "reviewed_at": "2026-07-06T12:00:00Z"}},
            dataset="set-curation",
        )
        assert stats == {"approved": 1}
        assert v2[0]["query"] == "what does the widget do"
        assert v2[0]["curation"] == {
            "dataset": "set-curation", "reviewer": "rodger",
            "reviewed_at": "2026-07-06T12:00:00Z", "status": "approved",
        }

    def test_edit_replaces_query_and_preserves_original(self):
        v2, stats = apply_verdicts(
            [_row("q1")],
            {"q1": {"verdict": "edit", "corrected_query": "how are auth policies loaded",
                    "review_reason": "sense drift"}},
        )
        assert stats == {"edited": 1}
        assert v2[0]["query"] == "how are auth policies loaded"
        assert v2[0]["curation"]["original_query"] == "what does the widget do"
        assert v2[0]["curation"]["reason"] == "sense drift"

    def test_edit_without_correction_kept_and_counted(self):
        v2, stats = apply_verdicts([_row("q1")], {"q1": {"verdict": "edit"}})
        assert stats == {"edit_missing_correction": 1}
        assert v2[0]["query"] == "what does the widget do"
        assert v2[0]["curation"]["status"] == "edit_missing_correction"

    def test_reject_drops_row(self):
        v2, stats = apply_verdicts(
            [_row("q1"), _row("q2")], {"q1": {"verdict": "reject"}}
        )
        assert stats == {"rejected_dropped": 1, "unreviewed": 1}
        assert [r["id"] for r in v2] == ["q2"]

    def test_unreviewed_inherits_flagged_prior_status(self):
        v2, stats = apply_verdicts(
            [_row("q1"), _row("q2")],
            {"q1": {"prior_status": "flagged"}},
        )
        assert stats == {"flagged": 1, "unreviewed": 1}
        by_id = {r["id"]: r for r in v2}
        assert by_id["q1"]["curation"]["status"] == "flagged"
        assert by_id["q2"]["curation"]["status"] == "unreviewed"

    def test_input_rows_not_mutated(self):
        rows = [_row("q1")]
        apply_verdicts(rows, {"q1": {"verdict": "edit", "corrected_query": "x y z"}})
        assert "curation" not in rows[0]
        assert rows[0]["query"] == "what does the widget do"


class TestCurationSummary:
    def test_counts_statuses_and_uncurated(self):
        from treeloom.domain.benchmark.curation import curation_summary

        rows = [
            {"id": "a", "curation": {"status": "approved"}},
            {"id": "b", "curation": {"status": "flagged"}},
            {"id": "c", "curation": {"status": "flagged"}},
            {"id": "d"},
            {"id": "e", "curation": {}},
        ]
        assert curation_summary(rows) == {
            "approved": 1, "flagged": 2, "uncurated": 2,
        }

    def test_empty(self):
        from treeloom.domain.benchmark.curation import curation_summary

        assert curation_summary([]) == {}
