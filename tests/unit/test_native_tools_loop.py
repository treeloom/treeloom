"""Phase 2 (internal-reranker-3): the native tool_calls agent loop.

Detroit-style: the only fake is the out-of-process LLM boundary (a scripted
chat() emitting canned (content, tool_calls) replies and recording exactly
what the loop sent). Tools are real Tool objects with real async callables.

The Anthropic tool_use/tool_result block translation is tested at the
body-builder level in tests/unit/test_model_route.py — here we pin the
loop-side protocol: schemas passed on every call, OpenAI/DeepSeek
conversation shapes (assistant messages carrying their tool_calls, per-call
`{"role":"tool","tool_call_id",...}` results), tool-less termination, the
forced-final cap fallback, defensive argument decoding, obs_char_limit, and
token accounting over the new message shapes.
"""
from __future__ import annotations

import copy

import pytest

from treeloom.adapters.benchmark.tool_base import Tool
from treeloom.application.benchmark.agent_loop import run_agent
from treeloom.application.benchmark.agentic_runner import _effective_native_tools
from treeloom.domain.benchmark.agent_metrics import categorize_prompt

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _tc(call_id: str, name: str, arguments: str) -> dict:
    """One OpenAI-shaped tool call."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


class _ScriptedNativeLLM:
    """Fakes only the out-of-process LLM boundary.

    `replies` is a list of (content, tool_calls) or (content, tool_calls,
    usage) tuples returned in order. Records a deep copy of every call's
    messages plus the tools kwarg so tests can assert the exact conversation
    the loop sent."""

    def __init__(self, replies: list[tuple]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def chat(self, messages, *, tools=None, max_tokens=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        reply = self.replies.pop(0)
        content, tcs = reply[0], reply[1]
        usage = reply[2] if len(reply) > 2 else dict(_USAGE)
        return content, usage, tcs


# ── (a) two rounds of tool_calls, then a tool-less final message ────────────

@pytest.mark.asyncio
async def test_native_loop_two_tool_rounds_then_final():
    order: list[tuple] = []

    async def search(**kw):
        order.append(("search_code", kw))
        return {"hits": "found src/foo.py"}

    async def read_file(**kw):
        order.append(("read_file", kw))
        return {"content": "def foo(): ..."}

    tools = [
        Tool(name="search_code", description="semantic search", run=search,
             parameters={"type": "object",
                         "properties": {"query": {"type": "string"}}}),
        Tool(name="read_file", description="read a file", run=read_file),
    ]
    llm = _ScriptedNativeLLM([
        ("let me search", [_tc("c1", "search_code", '{"query": "foo"}')]),
        (None, [_tc("c2", "read_file", '{"path": "src/foo.py"}')]),
        ("The answer is in src/foo.py: foo() does X.", None),
    ])

    res = await run_agent(query="where is foo?", tools=tools, llm=llm,
                          retrieved_files=[], native_tools=True)

    # Termination on the tool-less message — its text is the final answer.
    assert res.final_answer == "The answer is in src/foo.py: foo() does X."
    assert res.turns == 3
    assert res.tool_calls == 2
    assert res.hit_cap is False and res.error == ""

    # Execution order follows the emitted tool_calls, args json-decoded.
    assert order == [("search_code", {"query": "foo"}),
                     ("read_file", {"path": "src/foo.py"})]

    # Tool schemas built once via openai_schema() and passed on EVERY call.
    expected_schemas = [t.openai_schema() for t in tools]
    assert all(c["tools"] == expected_schemas for c in llm.calls)

    # Conversation shape as seen by the FINAL call (OpenAI/DeepSeek dialect):
    # assistant turns carry their tool_calls; each result is a role:"tool"
    # message bound by tool_call_id.
    msgs = llm.calls[2]["messages"]
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "tool", "assistant", "tool"]
    assert msgs[2]["content"] == "let me search"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "search_code"
    assert msgs[3]["tool_call_id"] == "c1"
    assert "found src/foo.py" in msgs[3]["content"]
    assert msgs[4]["tool_calls"][0]["id"] == "c2"
    assert msgs[5]["tool_call_id"] == "c2"

    # Token accounting: three usage blocks summed, tool results bucketed as
    # tool_observation, and no fallback (usage was always present).
    assert res.token_account.total_tokens == 3 * _USAGE["total_tokens"]
    assert res.token_account.used_fallback is False
    assert res.token_account.categories["tool_observation"] > 0
    assert res.token_account.categories["agent_output"] > 0

    # Transcript: same flat shape as Phase 1 — tool, args and the observation
    # text actually fed back, plus the terminal "final" entry.
    assert res.transcript[0]["parsed"] == "action"
    assert res.transcript[0]["tool"] == "search_code"
    assert res.transcript[0]["args"] == {"query": "foo"}
    assert "found src/foo.py" in res.transcript[0]["observation"]
    assert res.transcript[1]["tool"] == "read_file"
    assert res.transcript[-1]["parsed"] == "final"


# ── (b) malformed tool arguments → error observation, loop continues ────────

@pytest.mark.asyncio
async def test_native_loop_malformed_arguments_feed_error_and_continue():
    ran: list[dict] = []

    async def search(**kw):
        ran.append(kw)
        return {"hits": "ok"}

    tool = Tool(name="search_code", description="s", run=search)
    llm = _ScriptedNativeLLM([
        ("", [_tc("c1", "search_code", '{"query": ')]),  # broken JSON
        ("recovered final answer", None),
    ])

    res = await run_agent(query="q", tools=[tool], llm=llm,
                          retrieved_files=[], native_tools=True)

    # The tool never ran and the run didn't crash — the loop continued to a
    # normal final answer.
    assert ran == []
    assert res.tool_calls == 0
    assert res.final_answer == "recovered final answer"
    assert res.error == ""

    # The parse failure was fed back as THAT call's tool result.
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert "could not parse tool arguments" in tool_msgs[0]["content"]
    assert "could not parse tool arguments" in res.transcript[0]["observation_error"]


@pytest.mark.asyncio
async def test_native_loop_non_object_arguments_rejected():
    async def search(**kw):
        return {"hits": "ok"}

    tool = Tool(name="search_code", description="s", run=search)
    llm = _ScriptedNativeLLM([
        ("", [_tc("c1", "search_code", '["not", "an", "object"]')]),
        ("done", None),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm,
                          retrieved_files=[], native_tools=True)
    assert res.tool_calls == 0
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "must be a JSON object" in tool_msg["content"]


@pytest.mark.asyncio
async def test_native_loop_unknown_tool_feeds_error_and_continues():
    tool = Tool(name="search_code", description="s",
                run=lambda **kw: (_ for _ in ()).throw(AssertionError))
    llm = _ScriptedNativeLLM([
        ("", [_tc("c1", "no_such_tool", '{"x": 1}')]),
        ("final", None),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm,
                          retrieved_files=[], native_tools=True)
    assert res.tool_calls == 0
    assert res.final_answer == "final"
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "unknown tool" in tool_msg["content"]


@pytest.mark.asyncio
async def test_native_loop_tool_exception_becomes_error_observation():
    async def boom(**kw):
        raise RuntimeError("kaput")

    tool = Tool(name="search_code", description="s", run=boom)
    llm = _ScriptedNativeLLM([
        ("", [_tc("c1", "search_code", '{"q": "x"}')]),
        ("final", None),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm,
                          retrieved_files=[], native_tools=True)
    assert res.tool_calls == 1  # attempted, same semantics as ReAct
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "tool raised: kaput" in tool_msg["content"]


# ── (c) cap behavior: never emits tool-less → forced final ──────────────────

@pytest.mark.asyncio
async def test_native_loop_cap_forces_final():
    async def search(**kw):
        return {"hits": "x"}

    tool = Tool(name="search_code", description="s", run=search)
    llm = _ScriptedNativeLLM([
        ("", [_tc("c1", "search_code", '{"q": "a"}')]),
        ("", [_tc("c2", "search_code", '{"q": "b"}')]),
        ("forced final text", None),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm, retrieved_files=[],
                          max_turns=2, native_tools=True)
    assert res.hit_cap is True
    assert res.turns == 2
    assert res.tool_calls == 2
    assert res.final_answer == "forced final text"
    assert res.transcript[-1]["forced"] is True
    # The forced call STILL carries the tool schemas — Anthropic rejects
    # tool_use/tool_result history without a tools param.
    assert llm.calls[-1]["tools"] is not None
    last_user = [m for m in llm.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert "do NOT call any more tools" in last_user["content"]


@pytest.mark.asyncio
async def test_native_loop_forced_final_still_tool_calling_falls_back():
    # If even the forced call emits tool_calls with empty text, the loop uses
    # the last non-empty assistant text rather than an empty answer.
    async def search(**kw):
        return {"hits": "x"}

    tool = Tool(name="search_code", description="s", run=search)
    llm = _ScriptedNativeLLM([
        ("interim thought", [_tc("c1", "search_code", '{"q": "a"}')]),
        ("", [_tc("c2", "search_code", '{"q": "b"}')]),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm, retrieved_files=[],
                          max_turns=1, native_tools=True)
    assert res.hit_cap is True
    assert res.final_answer == "interim thought"


# ── (e) obs_char_limit applied to each tool result ───────────────────────────

@pytest.mark.asyncio
async def test_native_loop_obs_char_limit_truncates_tool_results():
    async def search(**kw):
        return "X" * 10_000  # string passthrough (production markdown path)

    tool = Tool(name="search_code", description="s", run=search)
    llm = _ScriptedNativeLLM([
        ("", [_tc("c1", "search_code", '{"q": "a"}')]),
        ("final", None),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm, retrieved_files=[],
                          obs_char_limit=100, native_tools=True)
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) < 200
    assert "truncated" in tool_msg["content"]
    assert "truncated" in res.transcript[0]["observation"]


# ── token accounting details over the native message shapes ─────────────────

def test_categorize_prompt_native_shapes():
    msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "where is foo?"},
        {"role": "assistant", "content": None,  # all-tool-call turn
         "tool_calls": [_tc("c1", "search_code", '{"query": "foo bar baz"}')]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "## src/foo.py:1-10\ndef foo(): ..."},
    ]
    cats = categorize_prompt(msgs)
    assert cats["system"] > 0
    assert cats["query"] > 0
    # role:"tool" results count as tool_observation.
    assert cats["tool_observation"] > 0
    # The assistant's tool-call JSON counts as agent_output even with None content.
    assert cats["agent_output"] > 0


def test_categorize_prompt_does_not_crash_on_block_content():
    # Defensive: non-string content (e.g. block lists) is stringified.
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    cats = categorize_prompt(msgs)
    assert cats["query"] > 0


@pytest.mark.asyncio
async def test_native_loop_fallback_counts_tool_call_json():
    # A server omitting `usage` falls back to tiktoken — the tool-call JSON
    # must be included in the completion estimate.
    async def search(**kw):
        return {"hits": "x"}

    tool = Tool(name="search_code", description="s", run=search)
    llm = _ScriptedNativeLLM([
        (None, [_tc("c1", "search_code", '{"query": "some longish query"}')], None),
        ("final", None, None),
    ])
    res = await run_agent(query="q", tools=[tool], llm=llm,
                          retrieved_files=[], native_tools=True)
    assert res.token_account.used_fallback is True
    # None content + tool_calls still produced completion tokens (the JSON).
    assert res.token_account.completion_tokens > 0


# ── protocol resolution (auto-default vs explicit override) ─────────────────

class _FakeRoutedLLM:
    def __init__(self, supports: bool):
        self.supports_native_tools = supports


def test_effective_native_tools_defaults_to_vendor_axis():
    assert _effective_native_tools(None, _FakeRoutedLLM(True)) is True
    assert _effective_native_tools(None, _FakeRoutedLLM(False)) is False


def test_effective_native_tools_explicit_override_wins():
    assert _effective_native_tools(False, _FakeRoutedLLM(True)) is False
    assert _effective_native_tools(True, _FakeRoutedLLM(False)) is True
