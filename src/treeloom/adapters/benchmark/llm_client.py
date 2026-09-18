"""Benchmark adapter: LLM client for session model search pattern generation.

Emulates how Claude Code's main session model (Opus/Sonnet) generates
rg patterns as tool calls within the conversation.
"""
import os
import re

import httpx

from treeloom.domain.benchmark.metrics import count_tokens


def resolve_llm_config() -> dict:
    """Resolve LLM config from BENCHMARK_QUALITY tier, allowing overrides.

    Precedence: explicit NOMCP_LLM_* vars > tier defaults.
    Returns dict with url, key, model, timeout keys.
    """
    quality = os.environ.get("BENCHMARK_QUALITY", "").upper()

    basic_url = os.environ.get(
        "BASIC_BENCHMARK_LLM_API_BASE", "http://localhost:11434/v1"
    )
    basic_key = os.environ.get("BASIC_BENCHMARK_LLM_API_KEY", "")
    basic_model = os.environ.get(
        "BASIC_BENCHMARK_LLM_MODEL", "qwen2.5-coder:7b"
    )
    basic_timeout = int(os.environ.get("BASIC_BENCHMARK_LLM_TIMEOUT", "45"))

    standard_url = os.environ.get(
        "STANDARD_BENCHMARK_LLM_API_BASE", "https://api.deepseek.com/v1"
    )
    standard_key = os.environ.get(
        "STANDARD_BENCHMARK_LLM_API_KEY",
        os.environ.get("DEEPSEEK_API_KEY", ""),
    )
    standard_model = os.environ.get(
        "STANDARD_BENCHMARK_LLM_MODEL", "deepseek-chat"
    )
    standard_timeout = int(os.environ.get("STANDARD_BENCHMARK_LLM_TIMEOUT", "30"))

    if quality == "STANDARD":
        url, key, model, timeout = standard_url, standard_key, standard_model, standard_timeout
    else:
        url, key, model, timeout = basic_url, basic_key, basic_model, basic_timeout

    return {
        "url": os.environ.get("NOMCP_LLM_URL") or url,
        "key": os.environ.get("NOMCP_LLM_API_KEY") or key,
        "model": os.environ.get("NOMCP_LLM_MODEL") or model,
        "timeout": int(os.environ.get("NOMCP_LLM_TIMEOUT", str(timeout))),
    }


# Floating model aliases (internal-reranker-3 Phase 3): names a vendor can
# silently remap to a different model between runs. The Jun-23→Jun-25
# `deepseek-chat` V3→v4-flash remap cost a full day of ghost-hunting — never
# benchmark against these without an explicit override. Value = the dated
# alternative to pin, or None where the vendor offers no dated IDs.
FLOATING_ALIASES: dict[str, str | None] = {
    "deepseek-chat": None,       # DeepSeek publishes aliases only
    "deepseek-reasoner": None,   # ditto
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
}

_NO_DATED_IDS_MSG = (
    "DeepSeek offers no dated IDs; re-run the smoke fixture "
    "(scripts/agentic_smoke.sh) before trusting results."
)


def is_floating_alias(model: str | None) -> str | None:
    """Return a human-readable warning when `model` is a floating alias.

    Floating = the vendor may remap the name to a different underlying model
    without notice (the fingerprint of the 2026-06-25 incident). Returns None
    for dated/pinned IDs. The message names the dated alternative where the
    vendor offers one.
    """
    if not model:
        return None
    m = model.strip().lower()
    if m.endswith("-latest"):
        return (
            f"'{model}' is a floating '-latest' alias — the vendor can remap it "
            "between runs. Pin the dated model ID it currently points at instead."
        )
    if m in FLOATING_ALIASES:
        dated = FLOATING_ALIASES[m]
        if dated:
            return (
                f"'{model}' is a floating alias — pin the dated ID "
                f"'{dated}' instead."
            )
        return f"'{model}' is a floating alias. {_NO_DATED_IDS_MSG}"
    return None


def resolve_model_route(model: str | None) -> dict | None:
    """Route a model NAME to its vendor endpoint, key, and request-shape flags.

    Powers the cross-model study: the agent model is selected by name and
    each name implies a different base URL, API key, and (for reasoning models)
    a different request shape. Returns a config dict shaped like
    `resolve_llm_config()` plus a `reasoning` bool and `vendor` tag — or None if
    the name doesn't match a known vendor (caller falls back to the tier config).

    Vendor dispatch is by name prefix:
      claude*            → Anthropic OpenAI-compat endpoint (ANTHROPIC_API_KEY)
      gpt-5* / o1/o3/o4* → OpenAI reasoning  (max_completion_tokens, no temperature)
      gpt-*              → OpenAI standard
      deepseek-reasoner  → DeepSeek reasoning;  deepseek*  → DeepSeek standard
      qwen* / *:* (tag)  → local ollama / OpenAI-compat (BASIC base, no key)
    """
    if not model:
        return None
    m = model.strip().lower()

    def _cfg(url_env, url_default, key, *, vendor, timeout,
             reasoning, supports_temperature, token_param="max_tokens",
             supports_native_tools=True):
        return {
            "url": os.environ.get(url_env) or url_default,
            "key": key or "",
            "model": model,
            "timeout": int(os.environ.get("NOMCP_LLM_TIMEOUT", str(timeout))),
            "vendor": vendor,
            # `reasoning` → the model burns hidden thinking tokens, so the budget
            # is floored. `supports_temperature` and `token_param` are INDEPENDENT
            # axes (Anthropic Opus 4.8 rejects temperature yet still wants
            # `max_tokens`, while OpenAI gpt-5* rejects it AND wants
            # `max_completion_tokens`) — learned the hard way via smoke test.
            "reasoning": reasoning,
            "supports_temperature": supports_temperature,
            "token_param": token_param,
            # Whether the vendor reliably implements native tool_calls — the
            # agentic harness defaults to the native loop when true and falls
            # back to the ReAct text protocol when false (local llama.cpp /
            # ollama models can't be trusted to emit structured tool calls).
            "supports_native_tools": supports_native_tools,
        }

    if m.startswith("claude"):
        # Anthropic OpenAI-compat endpoint: Bearer key, always `max_tokens`.
        # Omit temperature for ALL claude (Opus 4.8 rejects it; harmless to drop
        # for Sonnet). Opus is reasoning-class → floor the budget.
        is_thinking = m.startswith("claude-opus") or "thinking" in m
        return _cfg("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1",
                    os.environ.get("ANTHROPIC_API_KEY"),
                    vendor="anthropic", timeout=300,
                    reasoning=is_thinking, supports_temperature=False,
                    token_param="max_tokens")

    is_openai_reasoning = (
        m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")
        or m.startswith("o4")
    )
    if is_openai_reasoning:
        # NOTE: *-codex models are Responses-API only (404 on chat/completions);
        # not supported here. Use a chat-capable gpt-5* (e.g. gpt-5.5).
        return _cfg("OPENAI_BASE_URL", "https://api.openai.com/v1",
                    os.environ.get("OPENAI_API_KEY"),
                    vendor="openai", timeout=300,
                    reasoning=True, supports_temperature=False,
                    token_param="max_completion_tokens")
    if m.startswith("gpt-") or m.startswith("chatgpt"):
        return _cfg("OPENAI_BASE_URL", "https://api.openai.com/v1",
                    os.environ.get("OPENAI_API_KEY"),
                    vendor="openai", timeout=120,
                    reasoning=False, supports_temperature=True)

    if m.startswith("deepseek"):
        is_reasoner = m.startswith("deepseek-reasoner")
        return _cfg("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1",
                    os.environ.get("DEEPSEEK_API_KEY"),
                    vendor="deepseek", timeout=120,
                    reasoning=is_reasoner,
                    supports_temperature=not is_reasoner)

    if m.startswith("qwen") or ":" in m:
        return _cfg("BASIC_BENCHMARK_LLM_API_BASE", "http://localhost:11434/v1",
                    os.environ.get("BASIC_BENCHMARK_LLM_API_KEY"),
                    vendor="local", timeout=120,
                    reasoning=False, supports_temperature=True,
                    supports_native_tools=False)

    return None


def describe_config(cfg: dict | None = None) -> tuple[str, list[str]]:
    """Summarize the resolved benchmark LLM config + any misconfig warnings.

    Returns (summary_line, warnings). Callers print these so a run never
    silently uses the wrong (e.g. weak local) model.
    """
    cfg = cfg or resolve_llm_config()
    quality = os.environ.get("BENCHMARK_QUALITY", "").upper() or "BASIC"
    overridden = bool(
        os.environ.get("NOMCP_LLM_URL") or os.environ.get("NOMCP_LLM_MODEL")
    )
    tier = "override" if overridden else ("STANDARD" if quality == "STANDARD" else "BASIC")
    line = (
        f"model={cfg['model']} url={cfg['url']} tier={tier} "
        f"key={'set' if cfg['key'] else 'MISSING'}"
    )
    warnings: list[str] = []
    if quality == "STANDARD" and not overridden and not cfg["key"]:
        warnings.append(
            "BENCHMARK_QUALITY=STANDARD but no API key is set "
            "(STANDARD_BENCHMARK_LLM_API_KEY or DEEPSEEK_API_KEY) — calls will "
            "likely fail. Did you export it in THIS shell? (.env is not auto-loaded.)"
        )
    if tier == "BASIC":
        warnings.append(
            "Using the BASIC (local) tier — a small local model is usually too weak "
            "for gold answers / the agent loop (you'll get degenerate output). "
            "Set BENCHMARK_QUALITY=STANDARD (+ key) or NOMCP_LLM_* for a capable model."
        )
    return line, warnings


async def session_model_search(
    query: str, _client: httpx.AsyncClient | None = None
) -> tuple[list[str], int]:
    """Ask the session model to generate 1-3 rg search patterns.

    Args:
        query: The user's question about the codebase.
        _client: Optional httpx client for testing.

    Returns (patterns, model_tokens_used).
    """
    cfg = resolve_llm_config()
    if not cfg["url"] or not cfg["model"]:
        return [], 0

    prompt = (
        "You are searching a codebase. Given the user's question, output "
        "1-3 ripgrep search patterns to find the relevant code.\n\n"
        "Rules:\n"
        "- Output ONE pattern per line, nothing else\n"
        "- Prefer exact identifiers: function names, class names, method names\n"
        "- NEVER output single common words that match thousands of files\n\n"
        f"User question: {query}\n\nSearch patterns:"
    )

    prompt_tokens = count_tokens(prompt)

    try:
        headers = {}
        if cfg["key"]:
            headers["Authorization"] = f"Bearer {cfg['key']}"

        async def _call(client: httpx.AsyncClient):
            resp = await client.post(
                f"{cfg['url']}/chat/completions",
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"].strip()

        if _client is not None:
            content = await _call(_client)
        else:
            async with httpx.AsyncClient(
                timeout=cfg["timeout"], headers=headers
            ) as client:
                content = await _call(client)

        completion_tokens = count_tokens(content)

        patterns = [l.strip() for l in content.split("\n") if l.strip()]
        cleaned = []
        for p in patterns:
            p = re.sub(r"^\d+[\.\)]\s*", "", p)
            p = p.strip("- ").strip()
            if p and len(p) > 2:
                cleaned.append(p)

        return cleaned[:3], prompt_tokens + completion_tokens
    except Exception:
        return [], 0
