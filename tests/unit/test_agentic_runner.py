"""Unit tests for agentic_runner path normalization and the Phase-1
harness-rebuild row shaping (--debug-transcripts, internal-reranker-3).

Detroit-style: pure-function assertions, no services; the agent-loop test
scripts the out-of-process LLM boundary with a fake, never mocks internals.

The bug: ground-truth `relevant_files` are container-style paths
(`/data/repos/<repo>/...`) while the agent's `retrieved_files` are absolute
paths under the on-disk `--repo` checkout. The old `_norm` realpath'd both sides
but never relativized, so they never string-matched off-container and recall
read 0.0 for a working retriever. `_rel` reduces both to repo-relative form.
"""
from __future__ import annotations

import os

import pytest

from treeloom.adapters.benchmark.tool_base import Tool
from treeloom.application.benchmark.agent_loop import AgentRunResult, run_agent
from treeloom.application.benchmark.agentic_runner import _arm_row, _rel
from treeloom.domain.benchmark.agent_metrics import TokenAccount
from treeloom.domain.benchmark.metrics import recall_at_k


def test_rel_reduces_container_gt_and_ondisk_read_to_same_string():
    repo_root = "/home/user/source/mrdoob_three.js"
    # Ground truth — container-style absolute path.
    gt = ["/data/repos/mrdoob_three.js/src/Foo.js"]
    # Retrieved — on-disk path under the actual repo root.
    read = [os.path.join(repo_root, "src/Foo.js")]

    relevant = _rel(gt, repo_root)
    retrieved = _rel(read, repo_root)

    assert relevant == ["src/Foo.js"]
    assert retrieved == ["src/Foo.js"]
    # And therefore recall is credited.
    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_rel_nested_path_components():
    repo_root = "/srv/checkout/google_guava"
    gt = ["/data/repos/google_guava/guava/src/com/google/common/Base.java"]
    read = [os.path.join(repo_root, "guava/src/com/google/common/Base.java")]
    assert _rel(gt, repo_root) == _rel(read, repo_root)
    assert _rel(gt, repo_root) == ["guava/src/com/google/common/Base.java"]


def test_rel_dedupes_preserving_order():
    repo_root = "/repo/x"
    paths = [
        "/data/repos/x/a.py",
        os.path.join(repo_root, "a.py"),  # same repo-relative key as above
        "/data/repos/x/b.py",
    ]
    assert _rel(paths, repo_root) == ["a.py", "b.py"]


def test_rel_non_matching_path_left_relative_not_crashing():
    # A path with neither the repo root prefix nor the repo-name component is
    # returned normalized (leading slash stripped) rather than raising.
    repo_root = "/repo/myproj"
    assert _rel(["/totally/unrelated/file.py"], repo_root) == ["totally/unrelated/file.py"]


def test_rel_empty_repo_root_is_safe():
    assert _rel(["/data/repos/x/a.py"], "") == ["data/repos/x/a.py"]


def test_rel_trailing_slash_on_repo_root():
    repo_root = "/home/user/source/three.js/"
    read = ["/home/user/source/three.js/src/Foo.js"]
    gt = ["/data/repos/three.js/src/Foo.js"]
    assert _rel(read, repo_root) == ["src/Foo.js"]
    assert _rel(gt, repo_root) == ["src/Foo.js"]


# ── Phase 1 (internal-reranker-3): --debug-transcripts row shaping ──────────

def _result_with_transcript() -> AgentRunResult:
    return AgentRunResult(
        final_answer="It is validated in src/foo.py",
        turns=2,
        tool_calls=1,
        token_account=TokenAccount(),
        retrieved_files=["src/foo.py"],
        transcript=[{"turn": 0, "assistant": "THOUGHT...\nACTION: ...",
                     "parsed": "action", "tool": "search_code",
                     "args": {"query": "x"},
                     "observation": "OBSERVATION-PAYLOAD"}],
    )


def test_arm_row_omits_transcript_by_default():
    row = _arm_row(_result_with_transcript(), ["src/foo.py"], ["src/foo.py"],
                   {"correctness": 5, "completeness": 5})
    assert "transcript" not in row
    # The rest of the row shape is intact.
    assert row["final_answer"] == "It is validated in src/foo.py"
    assert row["recall@1"] == 1.0
    assert row["judge"]["correctness"] == 5


def test_arm_row_includes_transcript_when_debug_flag_on():
    rr = _result_with_transcript()
    row = _arm_row(rr, ["src/foo.py"], ["src/foo.py"], {"correctness": 5},
                   debug_transcripts=True)
    assert row["transcript"] is rr.transcript
    assert row["transcript"][0]["observation"] == "OBSERVATION-PAYLOAD"


class _ScriptedLLM:
    """Fakes only the out-of-process LLM boundary: canned assistant replies."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    async def chat(self, messages, **kw):
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return self.replies.pop(0), usage, None


@pytest.mark.asyncio
async def test_run_agent_transcript_records_tool_observations():
    """The transcript --debug-transcripts persists must INCLUDE the observation
    text the loop actually fed back (the Jun-25 incident was undecidable
    post-hoc precisely because observations were never recorded)."""
    async def fake_search(**kwargs):
        return {"hits": "the-observation-body from src/foo.py"}

    tool = Tool(name="search_code", description="search", run=fake_search)
    llm = _ScriptedLLM([
        'THOUGHT: look\nACTION: search_code({"query": "x"})',
        "THOUGHT: done\nFINAL_ANSWER: found it in src/foo.py",
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm, retrieved_files=[])
    assert res.final_answer == "found it in src/foo.py"
    obs = [e["observation"] for e in res.transcript if "observation" in e]
    assert len(obs) == 1
    assert "the-observation-body from src/foo.py" in obs[0]
    # And the same observation went back to the model as a user turn — the
    # transcript mirrors what was actually sent.
    assert res.transcript[0]["tool"] == "search_code"
