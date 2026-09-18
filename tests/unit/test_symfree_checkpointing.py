"""Symbol-free generation must survive a kill.

Generation is sequential and each entity costs an LLM round trip, so a few
thousand entities is ~an hour of PAID calls. The rows were accumulated in
memory and written only at the very end, so a process-level kill at entity
3,400 discarded the entire hour and every dollar of it.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from treeloom.application.benchmark import llm_query_gen as g


def _row(eid, q="How does it validate the payload?"):
    return {"id": f"r-{eid}", "query": q, "relevant_files": ["/f.py"],
            "repo": "r", "entity": {"id": eid, "name": "N", "type": "Class"}}


class TestLoadCheckpoint:
    def test_missing_file_is_a_clean_start(self):
        rows, done = g._load_checkpoint("/nonexistent/path.partial")
        assert rows == [] and done == set()

    def test_none_path_is_a_clean_start(self):
        assert g._load_checkpoint(None) == ([], set())

    def test_reads_back_rows_and_entity_ids(self, tmp_path):
        p = tmp_path / "c.partial"
        p.write_text("\n".join(json.dumps(_row(f"e{i}")) for i in range(3)) + "\n")
        rows, done = g._load_checkpoint(str(p))
        assert len(rows) == 3
        assert done == {"e0", "e1", "e2"}

    def test_torn_final_line_is_skipped_not_fatal(self, tmp_path):
        """The exact state a kill mid-write leaves behind.

        A resume that dies on the wreckage of the crash it exists to recover
        from is worse than no checkpoint at all.
        """
        p = tmp_path / "c.partial"
        p.write_text(json.dumps(_row("e0")) + "\n" + '{"id": "r-e1", "que')
        rows, done = g._load_checkpoint(str(p))
        assert len(rows) == 1 and done == {"e0"}

    def test_blank_lines_are_ignored(self, tmp_path):
        p = tmp_path / "c.partial"
        p.write_text(json.dumps(_row("e0")) + "\n\n\n")
        assert len(g._load_checkpoint(str(p))[0]) == 1


class TestGenerationStreamsAndResumes:
    def _entities(self, n):
        return [{"id": f"e{i}", "name": f"Name{i}", "type": "Class",
                 "file_path": "/f.py", "signature": ""} for i in range(n)]

    def _run(self, tmp_path, entities, done_ids=(), answer="It validates the payload."):
        ckpt = tmp_path / "out.jsonl.partial"
        if done_ids:
            ckpt.write_text("\n".join(json.dumps(_row(e)) for e in done_ids) + "\n")
        with patch.object(g, "_read_source", return_value="class Name0: pass"), \
             patch.object(g, "_behavioral_query", new=AsyncMock(return_value=answer)), \
             patch("treeloom.graph_store.list_entities",
                   new=AsyncMock(return_value=entities)):
            rows = asyncio.run(g.generate_symbol_free_queries(
                repo="r", llm=object(), max_entities=len(entities),
                checkpoint_path=str(ckpt)))
        return rows, ckpt

    def test_rows_are_on_disk_before_the_run_ends(self, tmp_path):
        rows, ckpt = self._run(tmp_path, self._entities(3))
        assert len(rows) == 3
        written = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
        assert len(written) == 3, "rows were not streamed to the checkpoint"

    def test_resume_skips_already_done_entities(self, tmp_path):
        """The LLM must not be re-paid for work the checkpoint already holds."""
        with patch.object(g, "_read_source", return_value="class X: pass"), \
             patch.object(g, "_behavioral_query",
                          new=AsyncMock(return_value="It does a thing.")) as bq, \
             patch("treeloom.graph_store.list_entities",
                   new=AsyncMock(return_value=self._entities(5))):
            ckpt = tmp_path / "out.jsonl.partial"
            ckpt.write_text("\n".join(json.dumps(_row(f"e{i}")) for i in range(3)) + "\n")
            rows = asyncio.run(g.generate_symbol_free_queries(
                repo="r", llm=object(), max_entities=5, checkpoint_path=str(ckpt)))
        assert len(rows) == 5, "resumed rows plus new ones"
        assert bq.await_count == 2, f"re-paid for done entities ({bq.await_count} calls)"

    def test_no_checkpoint_path_keeps_old_behaviour(self, tmp_path):
        with patch.object(g, "_read_source", return_value="class X: pass"), \
             patch.object(g, "_behavioral_query", new=AsyncMock(return_value="It works.")), \
             patch("treeloom.graph_store.list_entities",
                   new=AsyncMock(return_value=self._entities(2))):
            rows = asyncio.run(g.generate_symbol_free_queries(
                repo="r", llm=object(), max_entities=2))
        assert len(rows) == 2
