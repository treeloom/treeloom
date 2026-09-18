"""Concurrent agentic runs must never share a results file.

Names are `<UTC second>_<repo><suffix>`, so runs on the same repo starting in
the same second collided, and rows were opened with "w": the second run
truncated the first's file, then both wrote into it. Two parallel Cell B runs
in the 2026-09 refresh started one second apart.
"""

from concurrent.futures import ThreadPoolExecutor

from treeloom.application.benchmark.agentic_runner import _claim_result_paths


class TestClaimResultPaths:
    def test_first_claim_keeps_the_plain_name(self, tmp_path):
        rows, summary = _claim_result_paths(tmp_path, "2026-09-21T150023_featbit")
        assert rows.name == "2026-09-21T150023_featbit.jsonl"
        assert summary.name == "2026-09-21T150023_featbit_summary.json"
        assert rows.exists(), "the name must be claimed on disk, not just computed"

    def test_collision_gets_a_disambiguated_stem(self, tmp_path):
        a, _ = _claim_result_paths(tmp_path, "t_repo")
        b, sb = _claim_result_paths(tmp_path, "t_repo")
        c, _ = _claim_result_paths(tmp_path, "t_repo")
        assert [a.name, b.name, c.name] == ["t_repo.jsonl", "t_repo-2.jsonl", "t_repo-3.jsonl"]
        assert sb.name == "t_repo-2_summary.json"

    def test_rows_and_summary_still_pair_by_suffix_swap(self, tmp_path):
        """scripts/agentic_smoke*.{sh,py} derive one from the other."""
        _claim_result_paths(tmp_path, "t_repo")
        rows, summary = _claim_result_paths(tmp_path, "t_repo")
        assert str(rows).removesuffix(".jsonl") + "_summary.json" == str(summary)

    def test_existing_rows_are_never_touched(self, tmp_path):
        prior = tmp_path / "t_repo.jsonl"
        prior.write_text('{"id": "paid-for-row"}\n')
        rows, _ = _claim_result_paths(tmp_path, "t_repo")
        assert rows != prior
        assert prior.read_text() == '{"id": "paid-for-row"}\n'

    def test_simultaneous_claims_are_all_unique(self, tmp_path):
        """The actual race: O_EXCL lets exactly one claimant win each name."""
        with ThreadPoolExecutor(max_workers=32) as pool:
            got = list(pool.map(lambda _: _claim_result_paths(tmp_path, "t_repo")[0],
                                range(64)))
        assert len({p.name for p in got}) == 64
