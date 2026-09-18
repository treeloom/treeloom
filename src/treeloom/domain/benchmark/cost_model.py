"""Benchmark domain: dollar cost from token usage — pure, no I/O.

The token-opt work measured *tokens*, but the real driver is **cost**. Raw token
counts aren't comparable across models; dollars are. This module turns the
cost-bearing token detail accumulated by `TokenAccount` into USD using a
per-model price table.

PRICES ARE CONFIG, NOT GROUND TRUTH. Provider prices drift and several frontier
model names here may not have stable public pricing — every entry carries a
`verified` flag and the table a `PRICE_CAPTURE_DATE`. **Do not publish a cost
number for a model whose entry is `verified=False` without confirming its price
first** (mirror of the embedder contamination gate / the "verify before a long run"
rule). `cost_breakdown` returns `verified` so callers can refuse/flag.

Cache semantics, normalized (see TokenAccount):
- `cached_input_tokens` is a SUBSET of `prompt_tokens` for OpenAI/DeepSeek, so
  uncached input = prompt - cached - cache_write.
- `reasoning_tokens` is a SUBSET of `completion_tokens`, billed at the output
  rate — no separate term (tracked only for reporting).
"""
from __future__ import annotations

PRICE_CAPTURE_DATE = "2026-09-20"

# USD per 1,000,000 tokens. Self-hosted models cost GPU time, not API dollars,
# so they are priced at 0 here (their "cost" is out of scope for an API-dollar
# comparison) and marked verified.
PRICES: dict[str, dict] = {
    # DeepSeek — real, public pricing (cache-hit input is heavily discounted).
    "deepseek-chat": {"input": 0.27, "cached_input": 0.07, "cache_write": 0.27,
                      "output": 1.10, "verified": True},
    # Anthropic — CONFIRMED 2026-06-18 (input/cached/output); cache_write is the
    # standard 1.25x-input 5-min ephemeral write multiplier.
    "claude-sonnet-4-6": {"input": 3.0, "cached_input": 0.30, "cache_write": 3.75,
                          "output": 15.0, "verified": True},
    "claude-opus-4-8": {"input": 5.0, "cached_input": 0.50, "cache_write": 6.25,
                        "output": 25.0, "verified": True},
    # OpenAI gpt-5.5 — CONFIRMED 2026-06-18. OpenAI has no separate cache-write
    # charge (cached reads auto-discounted, writes billed at input), so
    # cache_write == input.
    "gpt-5.5": {"input": 5.0, "cached_input": 0.50, "cache_write": 5.0,
                "output": 30.0, "verified": True},
    # Self-hosted (GPU) — no API cost.
    "_local": {"input": 0.0, "cached_input": 0.0, "cache_write": 0.0,
               "output": 0.0, "verified": True},

    # ── CONFIRMED 2026-09-20 against vendor pricing pages ────────────────
    # DeepSeek publishes PEAK/OFF-PEAK rates (off-peak = half, outside
    # 01:00-04:00 and 06:00-10:00 UTC Mon-Fri). We store PEAK so estimates are
    # conservative; a run wholly in off-peak costs half these figures.
    "deepseek-flash": {"input": 0.30, "cached_input": 0.006,
                       "cache_write": 0.30, "output": 1.20,
                       "verified": True},
    # OpenAI gpt-5.6-sol — PROMOTIONAL pricing, stated as available at least
    # through 2026-11-21. Re-confirm after that date.
    # NOT USABLE BY THE AGENTIC HARNESS (verified 2026-09-20): function tools
    # + reasoning_effort are rejected on /v1/chat/completions —
    #   "Function tools with reasoning_effort are not supported for
    #    gpt-5.6-sol in /v1/chat/completions. To use function tools, use
    #    /v1/responses or set reasoning_effort to 'none'."
    # `reasoning_effort='none'` restores tool calls but disables reasoning, so
    # you would pay frontier prices for a non-reasoning model. The harness
    # speaks chat/completions only. Use gpt-5.5 for the OpenAI arm until
    # /v1/responses support exists. Priced here for reference, not for use.
    "gpt-5.6-sol": {"input": 4.00, "cached_input": 0.40,
                    "cache_write": 4.00, "output": 20.00,
                    "verified": True},
    # Anthropic — confirmed 2026-09-20. Sonnet 5's $2/$10 was introductory
    # pricing that has since become the standard price (the scheduled rise to
    # $3/$15 was cancelled), so the inherited $3/$15 was 33% too high.
    # NOTE: Claude 4.7+ use a newer tokenizer producing ~30% MORE tokens for
    # the same text, so token counts from gpt-4o/deepseek basis runs must be
    # scaled by ~1.3 before costing Sonnet 5 / Opus 5.
    "claude-sonnet-5": {"input": 2.0, "cached_input": 0.20, "cache_write": 2.50,
                        "output": 10.0, "verified": True},
    "claude-opus-5": {"input": 5.0, "cached_input": 0.50, "cache_write": 6.25,
                      "output": 25.0, "verified": True},
}


def price_for(model: str) -> dict | None:
    """Resolve the price entry for a model name, or None if unknown.

    Self-hosted local models (qwen*, *.gguf, the BASIC tier) resolve to the
    zero-cost `_local` entry; everything else must be an explicit table key.
    """
    if not model:
        return None
    m = model.strip().lower()
    if m in PRICES:
        return PRICES[m]
    if m.startswith("qwen") or m.endswith(".gguf") or ":" in m:
        return PRICES["_local"]
    return None


def cost_breakdown(usage_like: dict, model: str) -> dict | None:
    """USD cost for one accumulated usage dict (a TokenAccount.as_dict or arm row).

    Returns {"usd", "input_usd", "cached_usd", "cache_write_usd", "output_usd",
    "verified"} or None if the model has no price entry (so callers never
    fabricate a number for an unpriced model).
    """
    price = price_for(model)
    if price is None:
        return None
    prompt = int(usage_like.get("prompt_tokens") or 0)
    completion = int(usage_like.get("completion_tokens") or 0)
    cached = int(usage_like.get("cached_input_tokens") or 0)
    cache_write = int(usage_like.get("cache_write_tokens") or 0)
    uncached_input = max(prompt - cached - cache_write, 0)
    input_usd = uncached_input * price["input"] / 1e6
    cached_usd = cached * price["cached_input"] / 1e6
    cache_write_usd = cache_write * price["cache_write"] / 1e6
    output_usd = completion * price["output"] / 1e6
    return {
        "usd": round(input_usd + cached_usd + cache_write_usd + output_usd, 6),
        "input_usd": round(input_usd, 6),
        "cached_usd": round(cached_usd, 6),
        "cache_write_usd": round(cache_write_usd, 6),
        "output_usd": round(output_usd, 6),
        "verified": bool(price["verified"]),
    }


def compute_cost(usage_like: dict, model: str) -> float | None:
    """USD total for one usage dict, or None if the model is unpriced."""
    b = cost_breakdown(usage_like, model)
    return b["usd"] if b is not None else None
