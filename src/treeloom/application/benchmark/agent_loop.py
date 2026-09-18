"""Benchmark application: the multi-turn ReAct agent harness.

Arm-agnostic — the caller injects the tool list (grep/glob/read for the grep
arm, search_code for the treeloom arm). The loop accumulates real token usage,
turn/tool-call counts, and a transcript, and stops on FINAL_ANSWER or a forced
final answer at the turn cap. Only the tools differ between arms; everything
else (prompt skeleton, caps, temperature) is identical for a fair comparison.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from treeloom.adapters.benchmark.agent_llm import AgentLLM
from treeloom.adapters.benchmark.tool_base import Tool
from treeloom.domain.benchmark.agent_metrics import TokenAccount, categorize_prompt
from treeloom.domain.benchmark.agent_protocol import parse_action

_SYSTEM_TEMPLATE = """You are a code-search agent. Answer the user's question about a codebase by \
investigating it with the tools provided. Work in a loop, ONE step per turn:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>({{<json args>}})

After each ACTION you receive an OBSERVATION. Keep going until you can answer. \
When confident, output exactly:

THOUGHT: <final reasoning>
FINAL_ANSWER: <concise answer naming the specific file(s) and function(s)/class(es) \
that satisfy the request, and explaining what they do>

Rules:
- Emit exactly ONE ACTION per turn, OR a FINAL_ANSWER. Never both.
- ACTION args MUST be a single-line JSON object, e.g. ACTION: grep({{"pattern": "Foo"}})
- Only reference files you have actually observed; never invent paths.

Available tools:
{tools}
"""


_NATIVE_SYSTEM_TEMPLATE = """You are a code-search agent. Answer the user's question about a codebase by \
investigating it with the tools provided (native tool calls). Call tools to gather evidence — one or \
more calls per turn. When you can answer, reply WITHOUT calling any tool: that plain-text reply is \
your final answer. Name the specific file(s) and function(s)/class(es) that satisfy the request and \
explain what they do.

Rules:
- Only reference files you have actually observed; never invent paths.
"""


@dataclass
class AgentRunResult:
    final_answer: str
    turns: int
    tool_calls: int
    token_account: TokenAccount
    retrieved_files: list[str]
    transcript: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    hit_cap: bool = False
    error: str = ""


def build_system_prompt(tools: list[Tool]) -> str:
    return _SYSTEM_TEMPLATE.format(tools="\n".join(t.describe() for t in tools))


def build_native_system_prompt() -> str:
    """System prompt for the native tool_calls loop. No ReAct sentinels and no
    textual tool list — the tool names/descriptions/parameters travel in the
    request's `tools` schemas instead."""
    return _NATIVE_SYSTEM_TEMPLATE


def _truncate(obj: dict | str, limit: int) -> str:
    # A tool may return a pre-rendered string (e.g. the treeloom arm's
    # production markdown serialization). Pass strings through raw — JSON-encoding
    # them would escape every newline/quote and re-inflate the token count,
    # defeating the whole point of measuring the real serialization.
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    if len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text)} chars]"
    return text


def _prune(messages: list[dict], window: int | None) -> list[dict]:
    """Return the messages actually sent to the LLM.

    With `window` set, keep [system, query] + the last `window` messages and
    replace the middle with a one-line elision marker, so prompt size grows
    with the window rather than the whole transcript (the dominant agent-loop
    cost). `window=None` sends the full transcript (default, measurable).
    """
    if window is None or len(messages) <= 2 + window:
        return messages
    elided = len(messages) - 2 - window
    marker = {"role": "user",
              "content": f"OBSERVATION: [{elided} earlier step(s) elided to save context]"}
    return messages[:2] + [marker] + messages[-window:]


async def run_agent(
    *,
    query: str,
    tools: list[Tool],
    llm: AgentLLM,
    retrieved_files: list[str],
    max_turns: int = 12,
    obs_char_limit: int = 4000,
    context_window: int | None = None,
    system_extra: str = "",
    native_tools: bool = False,
) -> AgentRunResult:
    """Run the agent loop for one query/arm. `retrieved_files` is the per-arm
    log mutated by the tools (read_file paths or search_code chunk paths).

    `system_extra` is appended to the system prompt and therefore re-sent
    every turn — used for ambient-context arms (e.g. the aider repo map),
    whose per-turn context cost is the thing being measured.

    `native_tools=True` switches from the ReAct text protocol to native
    structured tool_calls (the model-agnostic path; internal-reranker-3
    Phase 2). ReAct stays byte-identical when the flag is off."""
    if native_tools:
        return await _run_agent_native(
            query=query, tools=tools, llm=llm, retrieved_files=retrieved_files,
            max_turns=max_turns, obs_char_limit=obs_char_limit,
            context_window=context_window, system_extra=system_extra,
        )
    tool_map = {t.name: t for t in tools}
    system = build_system_prompt(tools)
    if system_extra:
        system = f"{system}\n{system_extra}"
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    acct = TokenAccount()
    transcript: list[dict] = []
    tool_calls = 0
    last_answer = ""
    t0 = time.monotonic()

    for turn in range(max_turns):
        sent = _prune(messages, context_window)
        prompt_text = "\n".join(m["content"] for m in sent)
        prompt_cats = categorize_prompt(sent)
        try:
            content, usage, _ = await llm.chat(sent)
        except Exception as e:
            # One bad LLM call (e.g. context overflow on a small model) ends
            # this arm gracefully with the best answer so far — never aborts
            # the whole benchmark.
            return AgentRunResult(
                final_answer=last_answer,
                turns=turn,
                tool_calls=tool_calls,
                token_account=acct,
                retrieved_files=retrieved_files,
                transcript=transcript,
                latency_s=round(time.monotonic() - t0, 3),
                error=f"llm call failed at turn {turn}: {e}",
            )
        acct.add(usage, prompt_text=prompt_text, completion_text=content,
                 prompt_categories=prompt_cats)
        action = parse_action(content)
        last_answer = content.strip()
        transcript.append({"turn": turn, "assistant": content, "parsed": action.kind})

        if action.kind == "final":
            return AgentRunResult(
                final_answer=action.final_answer,
                turns=turn + 1,
                tool_calls=tool_calls,
                token_account=acct,
                retrieved_files=retrieved_files,
                transcript=transcript,
                latency_s=round(time.monotonic() - t0, 3),
            )

        messages.append({"role": "assistant", "content": content})

        if action.kind == "none" or action.tool not in tool_map:
            hint = action.error or f"unknown tool {action.tool!r}"
            obs = (
                f"OBSERVATION: Could not run that step ({hint}). Emit exactly one "
                f"ACTION: tool({{...}}) line or a FINAL_ANSWER."
            )
            messages.append({"role": "user", "content": obs})
            transcript[-1]["observation_error"] = hint
            transcript[-1]["observation"] = obs
            continue

        tool_calls += 1
        try:
            result = await tool_map[action.tool].run(**action.args)
        except Exception as e:  # tool must never crash the loop
            result = {"error": f"tool raised: {e}"}
        obs_text = _truncate(result, obs_char_limit)
        messages.append({"role": "user", "content": f"OBSERVATION: {obs_text}"})
        transcript[-1]["tool"] = action.tool
        transcript[-1]["args"] = action.args
        # Keep the observation text the loop actually fed back in the
        # transcript, so --debug-transcripts rows are self-diagnosing (the
        # 2026-06-25 incident was undecidable post-hoc without it).
        transcript[-1]["observation"] = obs_text

    # Hit the turn cap — force one final answer.
    messages.append(
        {"role": "user", "content": "You have used all steps. Output FINAL_ANSWER now."}
    )
    sent = _prune(messages, context_window)
    prompt_text = "\n".join(m["content"] for m in sent)
    prompt_cats = categorize_prompt(sent)
    err = ""
    try:
        content, usage, _ = await llm.chat(sent)
        acct.add(usage, prompt_text=prompt_text, completion_text=content,
                 prompt_categories=prompt_cats)
        forced = parse_action(content)
        answer = forced.final_answer if forced.kind == "final" else content.strip()
        transcript.append({"turn": max_turns, "assistant": content, "forced": True})
    except Exception as e:
        answer = last_answer
        err = f"forced final call failed: {e}"
    return AgentRunResult(
        final_answer=answer,
        turns=max_turns,
        tool_calls=tool_calls,
        token_account=acct,
        retrieved_files=retrieved_files,
        transcript=transcript,
        latency_s=round(time.monotonic() - t0, 3),
        hit_cap=True,
        error=err,
    )


# ── Native tool_calls loop (internal-reranker-3 Phase 2) ─────────────────────

def _msg_text(m: dict) -> str:
    """Countable text of one message for the tiktoken fallback: content (which
    may be None on all-tool-call assistant turns) plus any tool_calls JSON, so
    the fallback estimate covers the same bytes the vendor bills for."""
    c = m.get("content") or ""
    if not isinstance(c, str):
        c = json.dumps(c, default=str)
    tcs = m.get("tool_calls")
    return c + (json.dumps(tcs, default=str) if tcs else "")


def _decode_tool_args(raw) -> tuple[dict | None, str]:
    """Defensively json-decode a tool call's `arguments`. Returns (args, error);
    exactly one side is meaningful — a malformed/non-object payload yields an
    error string to feed back as the observation instead of crashing the loop."""
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (ValueError, TypeError) as e:
        return None, f"could not parse tool arguments as JSON: {e}"
    if not isinstance(args, dict):
        return None, f"tool arguments must be a JSON object, got {type(args).__name__}"
    return args, ""


async def _run_agent_native(
    *,
    query: str,
    tools: list[Tool],
    llm: AgentLLM,
    retrieved_files: list[str],
    max_turns: int,
    obs_char_limit: int,
    context_window: int | None,
    system_extra: str,
) -> AgentRunResult:
    """The native tool_calls execution branch of `run_agent`.

    Protocol: pass `tools=` schemas on every call; a response WITH tool_calls
    appends the assistant message (tool_calls included), executes each call and
    appends per-call `{"role":"tool","tool_call_id",...}` results; a TOOL-LESS
    assistant message terminates — its text is the final answer. The Anthropic
    native path translates these shapes into tool_use/tool_result blocks in
    `AgentLLM._anthropic_convo`.

    `context_window` pruning is deliberately NOT applied here: `_prune` cuts
    mid-transcript, which can orphan `role:"tool"` results from their assistant
    `tool_calls` turn — an invalid conversation both OpenAI and Anthropic 400.
    The flag was a ReAct-era ablation; combine it with --no-native-tools.
    """
    tool_map = {t.name: t for t in tools}
    schemas = [t.openai_schema() for t in tools]
    system = build_native_system_prompt()
    if system_extra:
        system = f"{system}\n{system_extra}"
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    acct = TokenAccount()
    transcript: list[dict] = []
    tool_calls = 0
    last_answer = ""
    t0 = time.monotonic()

    for turn in range(max_turns):
        prompt_text = "\n".join(_msg_text(m) for m in messages)
        prompt_cats = categorize_prompt(messages)
        try:
            content, usage, tcs = await llm.chat(messages, tools=schemas)
        except Exception as e:
            return AgentRunResult(
                final_answer=last_answer,
                turns=turn,
                tool_calls=tool_calls,
                token_account=acct,
                retrieved_files=retrieved_files,
                transcript=transcript,
                latency_s=round(time.monotonic() - t0, 3),
                error=f"llm call failed at turn {turn}: {e}",
            )
        # Tool-call JSON is billed completion output — include it in the
        # tiktoken fallback text so `used_fallback` totals stay comparable.
        completion_text = (content or "") + (
            json.dumps(tcs, default=str) if tcs else "")
        acct.add(usage, prompt_text=prompt_text, completion_text=completion_text,
                 prompt_categories=prompt_cats)
        if content and content.strip():
            last_answer = content.strip()

        if not tcs:
            # A tool-less assistant message IS the final answer.
            transcript.append({"turn": turn, "assistant": content or "",
                               "parsed": "final"})
            return AgentRunResult(
                final_answer=(content or "").strip(),
                turns=turn + 1,
                tool_calls=tool_calls,
                token_account=acct,
                retrieved_files=retrieved_files,
                transcript=transcript,
                latency_s=round(time.monotonic() - t0, 3),
            )

        # Assistant turn WITH its tool_calls — required context for the
        # role:"tool" results that follow (and for the Anthropic tool_use
        # block translation).
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": tcs})
        entry = {"turn": turn, "assistant": content or "", "parsed": "action"}
        transcript.append(entry)
        for i, tc in enumerate(tcs):
            rec = entry if i == 0 else {"turn": turn}
            if i > 0:
                transcript.append(rec)
            fn = tc.get("function") or {}
            name = fn.get("name")
            args, arg_err = _decode_tool_args(fn.get("arguments"))
            rec["tool"] = name
            rec["args"] = args if args is not None else fn.get("arguments")
            if arg_err:
                result: dict | str = {"error": arg_err}
                rec["observation_error"] = arg_err
            elif name not in tool_map:
                err_msg = f"unknown tool {name!r}"
                result = {"error": err_msg}
                rec["observation_error"] = err_msg
            else:
                tool_calls += 1
                try:
                    result = await tool_map[name].run(**args)
                except Exception as e:  # tool must never crash the loop
                    result = {"error": f"tool raised: {e}"}
            obs_text = _truncate(result, obs_char_limit)
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": obs_text})
            rec["observation"] = obs_text

    # Hit the turn cap — force one final answer. `tools=` must still be sent:
    # Anthropic rejects tool_use/tool_result history without a tools param.
    messages.append(
        {"role": "user",
         "content": "You have used all steps. Reply with your final answer now, "
                    "in plain text — do NOT call any more tools."}
    )
    prompt_text = "\n".join(_msg_text(m) for m in messages)
    prompt_cats = categorize_prompt(messages)
    err = ""
    try:
        content, usage, tcs = await llm.chat(messages, tools=schemas)
        completion_text = (content or "") + (
            json.dumps(tcs, default=str) if tcs else "")
        acct.add(usage, prompt_text=prompt_text, completion_text=completion_text,
                 prompt_categories=prompt_cats)
        answer = (content or "").strip() or last_answer
        transcript.append({"turn": max_turns, "assistant": content or "",
                           "forced": True})
    except Exception as e:
        answer = last_answer
        err = f"forced final call failed: {e}"
    return AgentRunResult(
        final_answer=answer,
        turns=max_turns,
        tool_calls=tool_calls,
        token_account=acct,
        retrieved_files=retrieved_files,
        transcript=transcript,
        latency_s=round(time.monotonic() - t0, 3),
        hit_cap=True,
        error=err,
    )
