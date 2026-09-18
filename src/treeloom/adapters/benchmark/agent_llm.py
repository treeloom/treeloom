"""Benchmark adapter: OpenAI-compatible chat client that exposes token usage.

Unlike `treeloom.adapters.llm_api.llm_adapter._chat` (which returns only the
content string), the agentic benchmark needs the real `usage` block from every
turn to sum true token cost. This client returns `(content, usage|None)` and is
configured via `resolve_llm_config()` (BENCHMARK_QUALITY tiers + NOMCP_LLM_*
overrides), with an optional `--model` override.
"""
from __future__ import annotations

import json
import os

import httpx

from treeloom.adapters.benchmark.llm_client import (
    resolve_llm_config,
    resolve_model_route,
)


class AgentLLM:
    """Thin async chat wrapper returning (content, usage) for token accounting.

    Config precedence: explicit ``config=`` > NOMCP_LLM_URL hard override >
    per-model vendor route (``resolve_model_route``, the cross-model path)
    > BENCHMARK_QUALITY tier. The model NAME selects the vendor endpoint/key, so
    ``AgentLLM(model="claude-opus-4-8")`` and ``AgentLLM(model="gpt-5.5")`` reach
    the right API without any env juggling.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        config: dict | None = None,
    ):
        if config is not None:
            cfg = config
        elif model and not os.environ.get("NOMCP_LLM_URL"):
            cfg = resolve_model_route(model) or resolve_llm_config()
        else:
            cfg = resolve_llm_config()
        self.url = cfg["url"].rstrip("/")
        self.key = cfg["key"]
        self.model = model or cfg["model"]
        self.timeout = cfg["timeout"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Three independent request-shape axes (see resolve_model_route):
        #   reasoning            → floor the token budget (hidden thinking tokens)
        #   supports_temperature → whether to send `temperature` at all
        #   token_param          → "max_tokens" vs "max_completion_tokens"
        self.reasoning = bool(cfg.get("reasoning", False))
        self.supports_temperature = bool(cfg.get("supports_temperature", True))
        self.token_param = cfg.get("token_param", "max_tokens")
        self.vendor = cfg.get("vendor", "")
        # Whether the vendor reliably implements native tool_calls. The tier
        # fallback (`resolve_llm_config`) carries no vendor info, so absent ⇒
        # False: an un-routed model defaults to the ReAct text protocol.
        self.supports_native_tools = bool(cfg.get("supports_native_tools", False))
        # Use Anthropic's NATIVE Messages API (not the OpenAI-compat shim) so
        # prompt caching actually fires — the compat path won't cache, so
        # Claude cost would read as uncached/upper-bound. Set ANTHROPIC_USE_COMPAT=1
        # to fall back to the chat/completions shim.
        self.native_anthropic = (
            self.vendor == "anthropic"
            and os.environ.get("ANTHROPIC_USE_COMPAT") != "1"
        )
        # Every distinct `model` id the vendor reports actually serving —
        # surfaced into _summary.json as `served_models` so floating-alias
        # remaps (the Jun-25 deepseek-chat V3→v4-flash incident) leave a
        # recorded fingerprint instead of an unrecorded environment difference.
        self.served_models: set[str] = set()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    @staticmethod
    def _raise_for_status_with_body(resp: httpx.Response) -> None:
        """`raise_for_status`, but with the response body in the message.

        The agent loop stringifies exceptions into the row's `error` field, so
        the body must live in the exception MESSAGE — a bare 400 makes a
        credit-balance error and a message-shape error indistinguishable in
        rows (the 2026-06-25 sonnet "0 turns" incident). Type is preserved
        (still `httpx.HTTPStatusError`) so callers' handling is unchanged.
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise httpx.HTTPStatusError(
                f"{e}; response body: {resp.text[:500]}",
                request=e.request, response=e.response,
            ) from None

    def _record_served_model(self, data: dict) -> None:
        served = data.get("model")
        if served:
            self.served_models.add(str(served))

    def _budget(self, max_tokens: int | None) -> int:
        budget = max_tokens or self.max_tokens
        if self.reasoning:
            # Hidden thinking tokens otherwise starve the visible answer.
            budget = max(budget, int(
                os.environ.get("REASONING_MAX_COMPLETION_TOKENS", "16000")))
        return budget

    def _build_body(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int | None,
    ) -> dict:
        """Build the /chat/completions body, branching on the request shape.

        Standard models get `max_tokens` + `temperature`. Reasoning models get
        `max_completion_tokens` (floored so hidden reasoning can't starve the
        visible answer) and NO `temperature` (they reject a non-default value).
        Pure + side-effect-free so it can be unit-tested without a network call.
        """
        budget = self._budget(max_tokens)
        body: dict = {"model": self.model, "messages": messages,
                      self.token_param: budget}
        if self.supports_temperature:
            body["temperature"] = self.temperature
        if self.reasoning and self.vendor == "openai":
            effort = os.environ.get("REASONING_EFFORT")
            if effort:
                body["reasoning_effort"] = effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    # ── Anthropic native Messages API (cache_control → real caching) ──

    def _anthropic_headers(self) -> dict:
        return {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}

    @staticmethod
    def _to_anthropic_tool(t: dict) -> dict:
        fn = t.get("function", t)
        return {"name": fn.get("name", ""), "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object",
                                                          "properties": {}}}

    @staticmethod
    def _tool_use_blocks(m: dict) -> list[dict]:
        """OpenAI assistant-with-tool_calls message → Anthropic content blocks:
        [{type:text},...] (if any text) + one {type:tool_use} per call."""
        blocks: list[dict] = []
        text = (m.get("content") or "").strip()
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            try:
                inp = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except (ValueError, TypeError):
                inp = {}
            if not isinstance(inp, dict):
                inp = {}
            blocks.append({"type": "tool_use", "id": tc.get("id"),
                           "name": fn.get("name"), "input": inp})
        return blocks

    @staticmethod
    def _anthropic_convo(messages: list[dict]) -> list[dict]:
        """Coerce an OpenAI-style transcript into a valid Anthropic message list.

        Native Messages is STRICT where the OpenAI-compat shim is lenient: it
        rejects empty content, requires user/assistant to STRICTLY ALTERNATE, and
        must START with a user turn. The ReAct loop violates this (e.g. an
        OBSERVATION + a turn-budget reminder both arrive as `user`), which 400s
        the native API. So: drop empties, MERGE consecutive same-role turns, and
        trim any leading non-user turn.

        Native-tools shapes translate to content BLOCKS: an assistant message
        with `tool_calls` → [{type:text},{type:tool_use,...}] blocks; a
        `role:"tool"` result → a user message with a {type:tool_result} block
        (consecutive results merge into ONE user turn, which is exactly what
        Anthropic requires after a multi-tool_use assistant turn). Pure-text
        messages keep plain-string content; blocks are built fresh per call so
        repeated body builds never mutate the caller's transcript.
        """
        out: list[dict] = []

        def _append(role: str, item) -> None:
            """Append or merge into the last turn. `item` is str | block list;
            merging mixed forms promotes the string side to a text block."""
            if out and out[-1]["role"] == role:
                prev = out[-1]["content"]
                if isinstance(prev, str) and isinstance(item, str):
                    out[-1]["content"] = prev + "\n\n" + item
                else:
                    pb = [{"type": "text", "text": prev}] if isinstance(prev, str) else list(prev)
                    nb = [{"type": "text", "text": item}] if isinstance(item, str) else item
                    out[-1]["content"] = pb + nb
            else:
                out.append({"role": role, "content": item})

        for m in messages:
            role = m.get("role")
            if role == "tool":
                _append("user", [{"type": "tool_result",
                                  "tool_use_id": m.get("tool_call_id"),
                                  "content": m.get("content") or ""}])
                continue
            if role not in ("user", "assistant"):
                continue
            if role == "assistant" and m.get("tool_calls"):
                blocks = AgentLLM._tool_use_blocks(m)
                if blocks:
                    _append("assistant", blocks)
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            _append(role, content)
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    def _anthropic_body(
        self, messages: list[dict], tools: list[dict] | None,
        max_tokens: int | None,
    ) -> dict:
        """Native Messages body with a `cache_control` breakpoint on the LAST
        message — NOT just the system block.

        The agent's system prompt is small (~220 tokens, below Anthropic's
        1024-token cache minimum), so caching it does nothing. The expensive
        re-sent content is the growing TRANSCRIPT (the ~16k-token search
        observations re-fed every turn). Anthropic caches everything up to and
        including a breakpoint, so putting it on the last message caches the
        whole conversation-so-far; the next turn (this list + a new pair) reads
        that prefix from cache. The system block keeps a breakpoint too (free;
        engages once the conversation crosses 1024 tokens)."""
        sys_parts = [m.get("content", "") for m in messages
                     if m.get("role") == "system" and m.get("content")]
        convo = self._anthropic_convo(messages)
        if convo:
            last = convo[-1]
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"],
                                    "cache_control": {"type": "ephemeral"}}]
            else:
                # Already block-form (tool_use / tool_result from the native
                # loop): breakpoint goes on the FINAL block — Anthropic caches
                # everything up to and including it, same semantics as above.
                blocks = list(last["content"])
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                last["content"] = blocks
        body: dict = {"model": self.model, "max_tokens": self._budget(max_tokens),
                      "messages": convo}
        if sys_parts:
            body["system"] = [{"type": "text", "text": "\n\n".join(sys_parts),
                               "cache_control": {"type": "ephemeral"}}]
        if self.supports_temperature:
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = [self._to_anthropic_tool(t) for t in tools]
        return body

    @staticmethod
    def _normalize_anthropic_usage(u: dict) -> dict:
        """Map native usage → the OpenAI-shaped dict TokenAccount expects.
        prompt_tokens includes cache read+creation so cost_model's
        `uncached = prompt - cached - cache_write` recovers input_tokens."""
        inp = int(u.get("input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        cr = int(u.get("cache_read_input_tokens") or 0)
        cw = int(u.get("cache_creation_input_tokens") or 0)
        return {"prompt_tokens": inp + cr + cw, "completion_tokens": out,
                "total_tokens": inp + cr + cw + out,
                "cache_read_input_tokens": cr,
                "cache_creation_input_tokens": cw}

    @staticmethod
    def _extract_anthropic(data: dict) -> tuple[str, list[dict] | None]:
        blocks = data.get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tus = [b for b in blocks if b.get("type") == "tool_use"]
        tool_calls = [
            {"id": b.get("id"), "type": "function",
             "function": {"name": b.get("name"),
                          "arguments": json.dumps(b.get("input") or {})}}
            for b in tus
        ] or None
        return text, tool_calls

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict | None, list[dict] | None]:
        """One chat completion.

        Returns (content, usage, tool_calls). `usage` is the raw OpenAI usage
        dict (or None if the server omits it). `tool_calls` is populated only
        when `tools` are passed and the model emits native tool calls.
        """
        if self.native_anthropic:
            body = self._anthropic_body(messages, tools, max_tokens)
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=self._anthropic_headers()
            ) as client:
                resp = await client.post(f"{self.url}/messages", json=body)
                self._raise_for_status_with_body(resp)
                data = resp.json()
            self._record_served_model(data)
            content, tool_calls = self._extract_anthropic(data)
            usage = self._normalize_anthropic_usage(data.get("usage") or {})
            return content, usage, tool_calls

        body = self._build_body(messages, tools, max_tokens)
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers()) as client:
            resp = await client.post(f"{self.url}/chat/completions", json=body)
            self._raise_for_status_with_body(resp)
            data = resp.json()

        self._record_served_model(data)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""
        usage = data.get("usage")
        tool_calls = message.get("tool_calls")
        return content, usage, tool_calls
