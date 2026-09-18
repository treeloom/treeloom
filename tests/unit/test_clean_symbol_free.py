"""`clean` silently emptied every symbol-free query set.

Two contracts that cannot both hold:
  * --llm-symbol-free guarantees a query does NOT contain its entity name;
  * is_answerable() requires the query DOES mention its entity name.

So clean rejected 100% of a symbol-free set and reported "wrote 0 rows" with
exit 0. The benchmark-refresh runbook pipes the symbol-free set straight
through clean, so the documented workflow produced an empty file that looked
like a normal run.
"""

import json
import subprocess
import sys

import pytest


def _row(i, query, name="UpdateWorkspaceValidator", files=("/r/a.cs",), strat="llm_symbol_free"):
    return {"id": f"q{i}", "query": query, "strategy": strat,
            "relevant_files": list(files), "repo": "r",
            "entity": {"id": f"e{i}", "name": name, "type": "Class"}}


def _clean(tmp_path, rows, *flags):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = subprocess.run(
        [sys.executable, "-m", "treeloom.benchmark", "clean",
         "--in", str(inp), "--out", str(out), *flags],
        capture_output=True, text=True)
    written = ([json.loads(l) for l in out.read_text().splitlines() if l.strip()]
               if out.exists() else None)
    return p, written


SYMFREE = [_row(i, f"How does the component validate workspace input number {i}?")
           for i in range(10)]


class TestSymbolFreeFlag:
    def test_symbol_free_rows_survive_with_the_flag(self, tmp_path):
        p, rows = _clean(tmp_path, SYMFREE, "--symbol-free")
        assert p.returncode == 0, p.stderr
        assert rows is not None and len(rows) == 10

    def test_dedup_still_applies(self, tmp_path):
        dupes = SYMFREE + [_row(99, SYMFREE[0]["query"])]
        _, rows = _clean(tmp_path, dupes, "--symbol-free")
        assert len(rows) == 10

    def test_limit_is_a_seeded_sample_not_a_head_slice(self, tmp_path):
        """A head slice of a path-ordered set is exactly the bias to avoid."""
        _, a = _clean(tmp_path, SYMFREE, "--symbol-free", "--limit", "4", "--seed", "1")
        assert [r["id"] for r in a] != [f"q{i}" for i in range(4)]

    def test_rows_without_ground_truth_are_still_dropped(self, tmp_path):
        bad = SYMFREE + [_row(50, "What does it do?", files=())]
        _, rows = _clean(tmp_path, bad, "--symbol-free")
        assert "q50" not in {r["id"] for r in rows}


class TestNeverSilentlyEmpty:
    def test_symbol_free_set_without_flag_fails_loudly(self, tmp_path):
        """The old behaviour: exit 0, empty file. Now: non-zero, no file, a hint."""
        p, rows = _clean(tmp_path, SYMFREE)
        assert p.returncode != 0
        assert rows is None, "an empty benchmark set was written"
        assert "--symbol-free" in p.stderr, "the error must say how to fix it"

    def test_symbol_bearing_rows_still_filter_normally(self, tmp_path):
        """The name rules still protect symbol-BEARING sets -- unchanged."""
        bearing = [_row(i, f"What does UpdateWorkspaceValidator do in case {i}?",
                        strat="base") for i in range(5)]
        p, rows = _clean(tmp_path, bearing)
        assert p.returncode == 0 and len(rows) == 5
