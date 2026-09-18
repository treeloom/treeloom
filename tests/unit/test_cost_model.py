"""Unit tests for the dollar cost model + cost-bearing token capture."""
from treeloom.domain.benchmark.cost_model import (
    compute_cost, cost_breakdown, price_for, PRICES,
)
from treeloom.domain.benchmark.agent_metrics import TokenAccount, aggregate


# ── price_for routing ──────────────────────────────────────────────────────

def test_price_for_known_and_local_and_unknown():
    assert price_for("deepseek-chat") is PRICES["deepseek-chat"]
    assert price_for("qwen2.5-coder:7b") is PRICES["_local"]      # ollama tag
    assert price_for("Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf") is PRICES["_local"]
    assert price_for("some-unpriced-model") is None
    assert price_for("") is None


# ── cost_breakdown / compute_cost ──────────────────────────────────────────

def test_cost_breakdown_subtracts_cached_from_input():
    # deepseek-chat: input 0.27, cached 0.07, output 1.10 per Mtok.
    usage = {"prompt_tokens": 1000, "completion_tokens": 500,
             "cached_input_tokens": 200}
    b = cost_breakdown(usage, "deepseek-chat")
    # uncached input = 800; cached = 200; output = 500
    assert b["input_usd"] == round(800 * 0.27 / 1e6, 6)
    assert b["cached_usd"] == round(200 * 0.07 / 1e6, 6)
    assert b["output_usd"] == round(500 * 1.10 / 1e6, 6)
    assert b["usd"] == round(b["input_usd"] + b["cached_usd"] + b["output_usd"], 6)
    assert b["verified"] is True


def test_local_model_is_free():
    usage = {"prompt_tokens": 10000, "completion_tokens": 5000}
    assert compute_cost(usage, "qwen2.5-coder:7b") == 0.0


def test_unknown_model_returns_none_not_zero():
    # Never fabricate a cost for an unpriced model.
    assert compute_cost({"prompt_tokens": 100, "completion_tokens": 10}, "mystery") is None


def test_confirmed_models_are_verified():
    # Prices confirmed 2026-06-18 — these must read as verified.
    for m in ("deepseek-chat", "claude-sonnet-4-6", "claude-opus-4-8", "gpt-5.5"):
        b = cost_breakdown({"prompt_tokens": 100, "completion_tokens": 10}, m)
        assert b is not None and b["verified"] is True, m


def test_unverified_flag_surfaces(monkeypatch):
    # The verified=False path still surfaces (mechanism, via an injected entry).
    monkeypatch.setitem(PRICES, "_unverified_test", {
        "input": 1.0, "cached_input": 0.1, "cache_write": 1.0,
        "output": 2.0, "verified": False})
    b = cost_breakdown({"prompt_tokens": 100, "completion_tokens": 10},
                       "_unverified_test")
    assert b is not None and b["verified"] is False


def test_confirmed_opus_cheaper_than_old_estimate():
    # Opus 4.8 confirmed at $5/$25 input/output (not the $15/$75 estimate).
    b = cost_breakdown({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                       "claude-opus-4-8")
    assert b["input_usd"] == 5.0 and b["output_usd"] == 25.0


def test_cache_write_priced_separately_and_excluded_from_input():
    # Anthropic-style: cache_write priced above input, not double-counted.
    usage = {"prompt_tokens": 1000, "completion_tokens": 0,
             "cached_input_tokens": 0, "cache_write_tokens": 400}
    b = cost_breakdown(usage, "claude-opus-4-8")  # input 5.0, cache_write 6.25
    assert b["input_usd"] == round(600 * 5.0 / 1e6, 6)           # 1000-400 uncached
    assert b["cache_write_usd"] == round(400 * 6.25 / 1e6, 6)


# ── TokenAccount captures provider dialects ────────────────────────────────

def test_token_account_openai_dialect():
    acc = TokenAccount()
    acc.add({"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
             "prompt_tokens_details": {"cached_tokens": 300},
             "completion_tokens_details": {"reasoning_tokens": 200}})
    d = acc.as_dict()
    assert d["cached_input_tokens"] == 300
    assert d["reasoning_tokens"] == 200
    assert d["cache_write_tokens"] == 0


def test_token_account_deepseek_and_anthropic_dialects_accumulate():
    acc = TokenAccount()
    acc.add({"prompt_tokens": 800, "completion_tokens": 100, "total_tokens": 900,
             "prompt_cache_hit_tokens": 500})                       # deepseek
    acc.add({"prompt_tokens": 600, "completion_tokens": 50, "total_tokens": 650,
             "cache_read_input_tokens": 100,
             "cache_creation_input_tokens": 400})                   # anthropic
    d = acc.as_dict()
    assert d["cached_input_tokens"] == 500 + 100
    assert d["cache_write_tokens"] == 400


# ── aggregate wiring ───────────────────────────────────────────────────────

def _arm(prompt, completion, corr, cached=0):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "cached_input_tokens": cached,
            "turns": 5, "tool_calls": 4, "recall@5": 0.8,
            "judge": {"correctness": corr, "completeness": corr}}


def test_aggregate_reports_cost_and_ratio():
    rows = [{"arms": {"grep": _arm(2000, 400, 4.0),
                      "treeloom": _arm(1500, 300, 4.5)}} for _ in range(3)]
    s = aggregate(rows, ["grep", "treeloom"], repo="x", model="deepseek-chat")
    assert s["grep"]["mean_cost_usd"] > 0
    assert s["treeloom"]["mean_cost_usd"] > 0
    assert s["grep"]["cost_verified"] is True
    assert s["grep"]["quality_per_usd"] > 0
    comp = s["comparison"]
    assert comp["cost_ratio_treeloom_over_grep"] is not None
    # treeloom cheaper here -> positive savings
    assert comp["cost_saved_pct"] > 0


def test_aggregate_unpriced_model_omits_cost():
    rows = [{"arms": {"grep": _arm(2000, 400, 4.0)}} for _ in range(2)]
    s = aggregate(rows, ["grep"], repo="x", model="mystery-model")
    assert "mean_cost_usd" not in s["grep"]
