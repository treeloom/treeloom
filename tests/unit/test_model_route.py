"""cross-model routing + request-shape (no network).

Also home to the Phase-1 harness-rebuild instrumentation tests
(internal-reranker-3): multi-turn Anthropic message-assembly replays, HTTP
error-body surfacing, served-model recording, and the agent-arm token-budget
source pin. Mocks only the httpx boundary — never internal treeloom classes.
"""
import ast
import inspect

import httpx
import pytest

from treeloom.adapters.benchmark.agent_llm import AgentLLM
from treeloom.adapters.benchmark.llm_client import resolve_model_route


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    # A stray NOMCP override must NOT hijack per-model routing.
    monkeypatch.delenv("NOMCP_LLM_URL", raising=False)
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    monkeypatch.delenv("REASONING_MAX_COMPLETION_TOKENS", raising=False)


@pytest.mark.parametrize(
    "model,vendor,reasoning,temp,token_param,host",
    [
        # Opus 4.8: reasoning-class (no temp) but Anthropic still wants max_tokens.
        ("claude-opus-4-8", "anthropic", True, False, "max_tokens", "api.anthropic.com"),
        # Sonnet 4.6: not reasoning, but we omit temperature for ALL claude.
        ("claude-sonnet-4-6", "anthropic", False, False, "max_tokens", "api.anthropic.com"),
        # OpenAI reasoning: no temp AND max_completion_tokens.
        ("gpt-5.5", "openai", True, False, "max_completion_tokens", "api.openai.com"),
        ("o3-mini", "openai", True, False, "max_completion_tokens", "api.openai.com"),
        ("gpt-4o", "openai", False, True, "max_tokens", "api.openai.com"),
        ("deepseek-chat", "deepseek", False, True, "max_tokens", "api.deepseek.com"),
        ("deepseek-reasoner", "deepseek", True, False, "max_tokens", "api.deepseek.com"),
        ("qwen2.5-coder:7b", "local", False, True, "max_tokens", "localhost"),
    ],
)
def test_route_vendor_and_shape(model, vendor, reasoning, temp, token_param, host):
    r = resolve_model_route(model)
    assert r is not None
    assert r["vendor"] == vendor
    assert r["reasoning"] is reasoning
    assert r["supports_temperature"] is temp
    assert r["token_param"] == token_param
    assert host in r["url"]
    assert r["model"] == model


def test_route_unknown_model_is_none():
    assert resolve_model_route("mistral-large") is None
    assert resolve_model_route(None) is None


def test_route_keys_resolved_per_vendor():
    assert resolve_model_route("claude-opus-4-8")["key"] == "sk-ant-test"
    assert resolve_model_route("gpt-5.5")["key"] == "sk-oai-test"
    assert resolve_model_route("deepseek-chat")["key"] == "sk-ds-test"


def test_agentllm_routes_by_model_name():
    llm = AgentLLM(model="claude-opus-4-8")
    assert "api.anthropic.com" in llm.url
    assert llm.key == "sk-ant-test"
    assert llm.reasoning is True
    assert llm.token_param == "max_tokens"


def test_opus_body_no_temperature_but_max_tokens():
    """Opus 4.8: reasoning floor + NO temperature, yet `max_tokens` (not the
    OpenAI `max_completion_tokens`) — the exact combo smoke-testing exposed."""
    llm = AgentLLM(model="claude-opus-4-8")
    body = llm._build_body([{"role": "user", "content": "hi"}], None, 400)
    assert "temperature" not in body
    assert "max_completion_tokens" not in body
    assert body["max_tokens"] == 16000  # floored (reasoning-class)


def test_sonnet_body_no_temperature_no_floor():
    llm = AgentLLM(model="claude-sonnet-4-6")
    body = llm._build_body([{"role": "user", "content": "hi"}], None, 400)
    assert "temperature" not in body       # omitted for all claude
    assert body["max_tokens"] == 400       # not reasoning → no floor


def test_reasoning_body_uses_max_completion_tokens_no_temperature():
    llm = AgentLLM(model="gpt-5.5")
    body = llm._build_body([{"role": "user", "content": "hi"}], None, 400)
    assert "temperature" not in body          # reasoning models reject it
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 16000   # floored above the 400 ask


def test_reasoning_floor_overridable(monkeypatch):
    monkeypatch.setenv("REASONING_MAX_COMPLETION_TOKENS", "2000")
    monkeypatch.setenv("REASONING_EFFORT", "low")
    llm = AgentLLM(model="gpt-5.5")
    body = llm._build_body([{"role": "user", "content": "hi"}], None, 400)
    assert body["max_completion_tokens"] == 2000
    assert body["reasoning_effort"] == "low"


def test_standard_body_keeps_temperature_and_max_tokens():
    llm = AgentLLM(model="deepseek-chat", temperature=0.0)
    body = llm._build_body([{"role": "user", "content": "hi"}], None, 400)
    assert body["max_tokens"] == 400
    assert body["temperature"] == 0.0
    assert "max_completion_tokens" not in body


# ── Anthropic native adapter ────────────────────────────────────────

def test_claude_uses_native_by_default_and_compat_opt_out(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_USE_COMPAT", raising=False)
    assert AgentLLM(model="claude-opus-4-8").native_anthropic is True
    monkeypatch.setenv("ANTHROPIC_USE_COMPAT", "1")
    assert AgentLLM(model="claude-opus-4-8").native_anthropic is False
    # non-anthropic never goes native
    assert AgentLLM(model="deepseek-chat").native_anthropic is False


def test_anthropic_body_hoists_system_with_cache_control():
    llm = AgentLLM(model="claude-sonnet-4-6")
    msgs = [{"role": "system", "content": "you are X"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}]
    body = llm._anthropic_body(msgs, None, 400)
    # system hoisted out of messages into a cached top-level block
    assert body["system"][0]["text"] == "you are X"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert "temperature" not in body          # omitted for all claude
    assert body["max_tokens"] == 400          # sonnet not reasoning → no floor


def test_anthropic_body_floors_budget_for_opus_reasoning():
    body = AgentLLM(model="claude-opus-4-8")._anthropic_body(
        [{"role": "user", "content": "q"}], None, 400)
    assert body["max_tokens"] == 16000        # opus reasoning → floored


def test_normalize_anthropic_usage_folds_cache_into_prompt():
    u = {"input_tokens": 100, "output_tokens": 40,
         "cache_read_input_tokens": 900, "cache_creation_input_tokens": 50}
    n = AgentLLM._normalize_anthropic_usage(u)
    assert n["prompt_tokens"] == 100 + 900 + 50   # so uncached = input recovers
    assert n["completion_tokens"] == 40
    assert n["cache_read_input_tokens"] == 900
    assert n["cache_creation_input_tokens"] == 50


def test_anthropic_convo_merges_consecutive_drops_empty_and_starts_user():
    # Native Messages requires alternation + non-empty + leading user.
    msgs = [
        {"role": "system", "content": "sys"},          # excluded from convo
        {"role": "assistant", "content": "leading asst"},  # trimmed (not user)
        {"role": "user", "content": "OBSERVATION: x"},
        {"role": "user", "content": "You have used 3 turns"},  # consecutive user
        {"role": "assistant", "content": ""},            # empty -> dropped
        {"role": "assistant", "content": "THOUGHT"},
    ]
    convo = AgentLLM._anthropic_convo(msgs)
    assert [m["role"] for m in convo] == ["user", "assistant"]   # alternates, starts user
    assert convo[0]["content"] == "OBSERVATION: x\n\nYou have used 3 turns"  # merged
    assert convo[1]["content"] == "THOUGHT"


def test_anthropic_body_uses_normalized_convo():
    llm = AgentLLM(model="claude-sonnet-4-6")
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    body = llm._anthropic_body(msgs, None, 400)
    assert len(body["messages"]) == 1
    # last message carries the cache_control breakpoint (blocks form)
    blk = body["messages"][0]["content"]
    assert blk == [{"type": "text", "text": "a\n\nb",
                    "cache_control": {"type": "ephemeral"}}]


def test_anthropic_body_caches_last_message_not_just_system():
    # The transcript (last message), not the tiny system prompt, is the real
    # re-sent cost — the breakpoint must be on the last conversation message.
    llm = AgentLLM(model="claude-opus-4-8")
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "thought"},
            {"role": "user", "content": "OBSERVATION: big payload"}]
    body = llm._anthropic_body(msgs, None, 400)
    last = body["messages"][-1]["content"]
    assert isinstance(last, list) and last[0]["cache_control"] == {"type": "ephemeral"}
    # earlier messages stay plain strings (single breakpoint on the prefix end)
    assert isinstance(body["messages"][0]["content"], str)


def test_extract_anthropic_text_and_tool_use():
    data = {"content": [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
    ]}
    text, tcs = AgentLLM._extract_anthropic(data)
    assert text == "hello world"
    assert tcs[0]["function"]["name"] == "search"
    assert '"q": "x"' in tcs[0]["function"]["arguments"]


# ── Phase 1 (internal-reranker-3): multi-turn replay of the message assembly ─
#
# Characterization: every OBSERVATION the ReAct loop feeds back must survive
# into the wire body `_anthropic_body` builds. The Jun-25 sonnet failure was
# undecidable post-hoc partly because this transform was only ever replayed on
# the happy path — these pin the edge cases too (empty assistant turn,
# forced-final consecutive-user merge, repeated body builds).

OBS_ONE = "OBSERVATION: ## src/services/segment.py:10-42 (0.91)\ndef update_segment(): ..."
OBS_TWO = "OBSERVATION: ## src/api/workspace.cs:5-30 (0.88)\nclass WorkspaceController ..."


def _body_text(body: dict) -> str:
    """Flatten body['messages'] content (plain strings or cache_control block
    lists) into one searchable string."""
    parts = []
    for m in body["messages"]:
        c = m["content"]
        if isinstance(c, list):
            parts.extend(b.get("text", "") for b in c)
        else:
            parts.append(c)
    return "\n".join(parts)


def _replay_messages() -> list[dict]:
    """The exact shape run_agent builds: [system, query, (assistant ACTION,
    user OBSERVATION) x 2]."""
    return [
        {"role": "system", "content": "you are a code-search agent"},
        {"role": "user", "content": "where is workspace update validated?"},
        {"role": "assistant",
         "content": 'THOUGHT: search\nACTION: search_code({"query": "workspace"})'},
        {"role": "user", "content": OBS_ONE},
        {"role": "assistant",
         "content": 'THOUGHT: narrow\nACTION: search_code({"query": "segment"})'},
        {"role": "user", "content": OBS_TWO},
    ]


def test_anthropic_body_multi_turn_replay_keeps_every_observation():
    llm = AgentLLM(model="claude-sonnet-4-6")
    body = llm._anthropic_body(_replay_messages(), None, 512)
    text = _body_text(body)
    assert "src/services/segment.py:10-42" in text
    assert "src/api/workspace.cs:5-30" in text
    # Happy path already alternates: nothing merged, nothing dropped.
    assert [m["role"] for m in body["messages"]] == [
        "user", "assistant", "user", "assistant", "user"]


def test_anthropic_body_empty_assistant_between_observations_keeps_both():
    # An empty assistant turn between two observations makes the user turns
    # consecutive after the drop — the merge must retain BOTH observations.
    msgs = _replay_messages()
    msgs[4] = {"role": "assistant", "content": ""}  # empty → dropped
    body = AgentLLM(model="claude-sonnet-4-6")._anthropic_body(msgs, None, 512)
    text = _body_text(body)
    assert "src/services/segment.py:10-42" in text
    assert "src/api/workspace.cs:5-30" in text
    # The two observations merged into ONE user turn (alternation preserved).
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]


def test_anthropic_body_forced_final_reminder_merge_keeps_observation():
    # Turn cap: run_agent appends a second consecutive user message ("Output
    # FINAL_ANSWER now"). The merge must retain the observation AND the reminder.
    reminder = "You have used all steps. Output FINAL_ANSWER now."
    msgs = _replay_messages() + [{"role": "user", "content": reminder}]
    body = AgentLLM(model="claude-sonnet-4-6")._anthropic_body(msgs, None, 512)
    last = body["messages"][-1]
    assert last["role"] == "user"
    merged = "".join(b.get("text", "") for b in last["content"])
    assert "src/api/workspace.cs:5-30" in merged
    assert reminder in merged
    # Earlier observation untouched.
    assert "src/services/segment.py:10-42" in _body_text(body)


def test_anthropic_body_is_idempotent_across_calls():
    # A previous chat() call turns the LAST convo message into block form for
    # cache_control. That mutation must not leak into the caller's `messages`
    # list — the next turn's body build must still see every observation.
    llm = AgentLLM(model="claude-sonnet-4-6")
    msgs = _replay_messages()
    body1 = llm._anthropic_body(msgs, None, 512)
    body2 = llm._anthropic_body(msgs, None, 512)
    for body in (body1, body2):
        text = _body_text(body)
        assert "src/services/segment.py:10-42" in text
        assert "src/api/workspace.cs:5-30" in text
    assert body1 == body2
    # The input transcript itself stays plain strings (unmutated).
    assert all(isinstance(m["content"], str) for m in msgs)


# ── Phase 2 (internal-reranker-3): native tool_calls ─────────────────────────
#
# Route axis + the Anthropic tool_use/tool_result block translation, pinned at
# the body-builder level (the loop-side protocol lives in
# test_native_tools_loop.py).

@pytest.mark.parametrize(
    "model,supports",
    [
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-8", True),
        ("gpt-4o", True),
        ("gpt-5.5", True),
        ("deepseek-chat", True),
        ("deepseek-reasoner", True),
        ("qwen2.5-coder:7b", False),   # local llama.cpp/ollama → ReAct
    ],
)
def test_route_supports_native_tools_axis(model, supports):
    assert resolve_model_route(model)["supports_native_tools"] is supports


def test_agentllm_supports_native_tools_attribute():
    assert AgentLLM(model="deepseek-chat").supports_native_tools is True
    assert AgentLLM(model="qwen2.5-coder:7b").supports_native_tools is False
    # Tier fallback (no --model) carries no vendor info → ReAct default.
    assert AgentLLM().supports_native_tools is False


def _native_replay_messages() -> list[dict]:
    """The exact shape the native loop builds: [system, user,
    assistant+tool_calls, tool result, assistant with TWO tool_calls (no
    text), two tool results]."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "where is workspace update validated?"},
        {"role": "assistant", "content": "let me search",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "search_code",
                                      "arguments": '{"query": "workspace"}'}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "## src/services/segment.py:10-42\ndef update_segment(): ..."},
        {"role": "assistant", "content": None,
         "tool_calls": [
             {"id": "c2", "type": "function",
              "function": {"name": "read_file",
                           "arguments": '{"path": "src/foo.py"}'}},
             {"id": "c3", "type": "function",
              "function": {"name": "read_file",
                           "arguments": '{"path": "src/bar.py"}'}},
         ]},
        {"role": "tool", "tool_call_id": "c2", "content": "def foo(): ..."},
        {"role": "tool", "tool_call_id": "c3", "content": "def bar(): ..."},
    ]


def test_anthropic_convo_translates_tool_calls_to_blocks():
    convo = AgentLLM._anthropic_convo(_native_replay_messages())
    assert [m["role"] for m in convo] == [
        "user", "assistant", "user", "assistant", "user"]
    # assistant with text + one tool call → [text block, tool_use block]
    a1 = convo[1]["content"]
    assert a1[0] == {"type": "text", "text": "let me search"}
    assert a1[1]["type"] == "tool_use"
    assert a1[1]["id"] == "c1"
    assert a1[1]["name"] == "search_code"
    assert a1[1]["input"] == {"query": "workspace"}   # arguments json-decoded
    # role:"tool" result → user message with a tool_result block
    u1 = convo[2]["content"]
    assert u1 == [{"type": "tool_result", "tool_use_id": "c1",
                   "content": "## src/services/segment.py:10-42\n"
                              "def update_segment(): ..."}]
    # None content → no text block, just the two tool_use blocks
    a2 = convo[3]["content"]
    assert [b["type"] for b in a2] == ["tool_use", "tool_use"]
    # BOTH results merged into ONE user turn (Anthropic requires every
    # tool_result for a multi-tool_use turn in the next user message).
    u2 = convo[4]["content"]
    assert [b["tool_use_id"] for b in u2] == ["c2", "c3"]
    assert all(b["type"] == "tool_result" for b in u2)


def test_anthropic_convo_unparseable_tool_arguments_become_empty_input():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "search_code",
                                      "arguments": '{"broken": '}}]},
    ]
    convo = AgentLLM._anthropic_convo(msgs)
    assert convo[1]["content"][0]["input"] == {}


def test_anthropic_convo_forced_final_reminder_after_tool_results():
    # The cap reminder (a plain user string) lands AFTER tool results — it must
    # merge into the same user turn as a text block, preserving alternation.
    reminder = "You have used all steps. Reply with your final answer now."
    msgs = _native_replay_messages() + [{"role": "user", "content": reminder}]
    convo = AgentLLM._anthropic_convo(msgs)
    assert [m["role"] for m in convo] == [
        "user", "assistant", "user", "assistant", "user"]
    last = convo[-1]["content"]
    assert last[0]["type"] == "tool_result"
    assert last[-1] == {"type": "text", "text": reminder}


def test_anthropic_body_native_tools_translation_and_cache_control():
    llm = AgentLLM(model="claude-sonnet-4-6")
    tools = [{"type": "function",
              "function": {"name": "search_code", "description": "semantic search",
                           "parameters": {"type": "object",
                                          "properties": {"query": {"type": "string"}}}}}]
    body = llm._anthropic_body(_native_replay_messages(), tools, 512)
    # OpenAI schemas → Anthropic {name, description, input_schema}.
    assert body["tools"] == [{
        "name": "search_code", "description": "semantic search",
        "input_schema": {"type": "object",
                         "properties": {"query": {"type": "string"}}}}]
    # cache_control breakpoint on the FINAL block of the LAST (block-form)
    # message — same whole-prefix caching semantics as the string path.
    last_blocks = body["messages"][-1]["content"]
    assert last_blocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in b for b in last_blocks[:-1])
    # System block still hoisted + cached.
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_body_native_replay_is_idempotent_and_non_mutating():
    llm = AgentLLM(model="claude-sonnet-4-6")
    msgs = _native_replay_messages()
    body1 = llm._anthropic_body(msgs, None, 512)
    body2 = llm._anthropic_body(msgs, None, 512)
    assert body1 == body2
    # The caller's transcript is untouched (no block-form leakage).
    assert msgs == _native_replay_messages()


# ── Phase 1: HTTP error bodies + served-model recording (mock httpx only) ───

def _fake_post(payload: dict, status: int = 200):
    async def post(self, url, json=None):
        return httpx.Response(status, json=payload,
                              request=httpx.Request("POST", url))
    return post


def test_raise_for_status_appends_response_body():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(
        400,
        json={"error": {"type": "invalid_request_error",
                        "message": "Your credit balance is too low"}},
        request=req)
    with pytest.raises(httpx.HTTPStatusError) as ei:
        AgentLLM._raise_for_status_with_body(resp)
    # agent_loop stringifies the exception into the row's `error` field, so
    # the body must be in str(e) — a bare 400 is not self-identifying.
    assert "credit balance is too low" in str(ei.value)
    assert "400" in str(ei.value)


def test_raise_for_status_truncates_body_to_500_chars():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, json={"error": {"message": "x" * 2000}}, request=req)
    with pytest.raises(httpx.HTTPStatusError) as ei:
        AgentLLM._raise_for_status_with_body(resp)
    assert len(str(ei.value)) < 800


def test_raise_for_status_noop_on_2xx():
    resp = httpx.Response(200, json={"ok": True},
                          request=httpx.Request("POST", "http://x"))
    AgentLLM._raise_for_status_with_body(resp)  # must not raise


@pytest.mark.asyncio
async def test_chat_http_error_message_contains_body(monkeypatch):
    llm = AgentLLM(model="deepseek-chat")
    monkeypatch.setattr(
        httpx.AsyncClient, "post",
        _fake_post({"error": {"message": "Insufficient Balance"}}, status=400))
    with pytest.raises(httpx.HTTPStatusError) as ei:
        await llm.chat([{"role": "user", "content": "hi"}])
    assert "Insufficient Balance" in str(ei.value)


@pytest.mark.asyncio
async def test_chat_records_served_model_openai_compat(monkeypatch):
    llm = AgentLLM(model="deepseek-chat")
    assert llm.served_models == set()
    payload = {"model": "deepseek-v4-flash",  # alias echo / remap fingerprint
               "choices": [{"message": {"content": "ok"}}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                         "total_tokens": 4}}
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post(payload))
    content, usage, _ = await llm.chat([{"role": "user", "content": "hi"}])
    assert content == "ok"
    assert usage["total_tokens"] == 4
    assert llm.served_models == {"deepseek-v4-flash"}
    # Accumulates a SET across calls — no duplicates.
    await llm.chat([{"role": "user", "content": "hi again"}])
    assert llm.served_models == {"deepseek-v4-flash"}


@pytest.mark.asyncio
async def test_chat_records_served_model_anthropic_native(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_USE_COMPAT", raising=False)
    llm = AgentLLM(model="claude-sonnet-4-6")
    assert llm.native_anthropic is True
    payload = {"model": "claude-sonnet-4-6-20260401",  # dated id, not the alias
               "content": [{"type": "text", "text": "ok"}],
               "usage": {"input_tokens": 3, "output_tokens": 1}}
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post(payload))
    content, _, _ = await llm.chat([{"role": "user", "content": "hi"}])
    assert content == "ok"
    assert llm.served_models == {"claude-sonnet-4-6-20260401"}


# ── Phase 1: agent-arm token budget right-sized (512 → 2048) ────────────────

def test_agentic_runner_token_budgets_source_pinned():
    """AST source pin (idiom of test_graph_scoring_default.py): the literal
    max_tokens each AgentLLM in run_agentic is constructed with. The Jun-25
    sonnet probes truncated every completion at exactly 512 (completion_tokens
    = turns x 512) — the measured agent gets 2048; judge/gold stay 256/512."""
    from treeloom.application.benchmark import agentic_runner

    tree = ast.parse(inspect.getsource(agentic_runner))
    budgets: dict[str, int] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "AgentLLM"):
            for kw in node.value.keywords:
                if kw.arg == "max_tokens" and isinstance(kw.value, ast.Constant):
                    budgets[node.targets[0].id] = kw.value.value
    assert budgets.get("agent_llm") == 2048
    assert budgets.get("judge_llm") == 256
    assert budgets.get("gold_llm") == 512


def test_reasoning_floor_still_applies_over_2048():
    # The 2048 agent budget must not defeat the reasoning floor.
    llm = AgentLLM(model="gpt-5.5", max_tokens=2048)
    body = llm._build_body([{"role": "user", "content": "hi"}], None, None)
    assert body["max_completion_tokens"] == 16000
    a_body = AgentLLM(model="claude-opus-4-8", max_tokens=2048)._anthropic_body(
        [{"role": "user", "content": "hi"}], None, None)
    assert a_body["max_tokens"] == 16000
