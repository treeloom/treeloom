"""Benchmark domain: token accounting + cross-arm aggregation — pure, no I/O."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

from treeloom.domain.benchmark.metrics import count_tokens
from treeloom.domain.benchmark.significance import sign_test, win_loss_tie

# Token categories tracked per turn's *prompt* (plus "completion" for output).
# system = system prompt + tool schemas/descriptions (re-sent every turn);
# query = the user's question; agent_output = the model's own prior
# THOUGHT/ACTION turns re-fed as context; tool_observation = tool results
# (grep matches / file reads / search_code payloads) fed back in.
CATEGORIES = ("system", "query", "agent_output", "tool_observation", "completion")


def _content_text(content) -> str:
    """Flatten a message `content` field to countable text.

    ReAct messages are plain strings; native-tools messages may carry None
    (assistant turns that are ALL tool calls) or block lists — stringify
    non-str shapes instead of crashing the categorizer."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def categorize_prompt(messages: list[dict]) -> dict[str, int]:
    """tiktoken-count the current prompt messages by category (pure).

    Native-tools shapes: a `role:"tool"` result message counts as
    tool_observation; an assistant message's `tool_calls` JSON counts as
    agent_output (it is the model's own prior output re-fed as context).
    """
    cats = {"system": 0, "query": 0, "agent_output": 0, "tool_observation": 0}
    for m in messages:
        role = m.get("role", "")
        content = _content_text(m.get("content"))
        n = count_tokens(content)
        if role == "system":
            cats["system"] += n
        elif role == "assistant":
            cats["agent_output"] += n
            tcs = m.get("tool_calls")
            if tcs:
                cats["agent_output"] += count_tokens(json.dumps(tcs, default=str))
        elif role == "tool":
            cats["tool_observation"] += n
        elif role == "user":
            if content.startswith("OBSERVATION:") or content.startswith("You have used"):
                cats["tool_observation"] += n
            else:
                cats["query"] += n
    return cats


@dataclass
class TokenAccount:
    """Accumulates real LLM token usage across the turns of an agent run.

    Prefers the API's `usage` block; if a server omits it, falls back to
    tiktoken estimates and flags `used_fallback` so totals stay auditable.

    Also tracks a per-category breakdown: each turn's real prompt tokens are
    distributed across categories by their tiktoken proportion (totals anchored
    to the authoritative `usage`, split by tiktoken), so we can report what
    share is system/tool-schema overhead vs tool payloads vs re-fed transcript
    vs actual generation.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Cost-bearing detail fields, normalized across providers:
    #   cached_input_tokens — input served from a prompt cache (cheaper); a SUBSET
    #     of prompt_tokens for OpenAI (prompt_tokens_details.cached_tokens) and
    #     DeepSeek (prompt_cache_hit_tokens). Anthropic reports it separately
    #     (cache_read_input_tokens) but we don't cache over the compat shim.
    #   cache_write_tokens — Anthropic cache-creation (priced ABOVE input).
    #   reasoning_tokens — hidden thinking, a SUBSET of completion_tokens, billed
    #     at the output rate (tracked for reporting, not a separate price term).
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    used_fallback: bool = False
    categories: dict[str, float] = field(
        default_factory=lambda: {c: 0.0 for c in CATEGORIES}
    )

    def _add_cost_detail(self, usage: dict) -> None:
        """Pull cost-bearing detail fields from a provider usage block.

        Normalizes the three provider dialects we hit:
          OpenAI    prompt_tokens_details.cached_tokens / completion_tokens_details.reasoning_tokens
          DeepSeek  prompt_cache_hit_tokens
          Anthropic cache_read_input_tokens / cache_creation_input_tokens
        """
        ptd = usage.get("prompt_tokens_details") or {}
        ctd = usage.get("completion_tokens_details") or {}
        self.cached_input_tokens += int(
            ptd.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or usage.get("cache_read_input_tokens")
            or 0
        )
        self.cache_write_tokens += int(usage.get("cache_creation_input_tokens") or 0)
        self.reasoning_tokens += int(
            ctd.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
        )

    def add(
        self,
        usage: dict | None,
        *,
        prompt_text: str = "",
        completion_text: str = "",
        prompt_categories: dict[str, int] | None = None,
    ) -> None:
        if usage and usage.get("total_tokens"):
            p = int(usage.get("prompt_tokens", 0) or 0)
            c = int(usage.get("completion_tokens", 0) or 0)
            t = int(usage["total_tokens"])
            self._add_cost_detail(usage)
        else:
            self.used_fallback = True
            p = count_tokens(prompt_text) if prompt_text else 0
            c = count_tokens(completion_text) if completion_text else 0
            t = p + c
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        # Distribute the real prompt tokens across categories by tiktoken share.
        if prompt_categories:
            tt = sum(prompt_categories.values())
            if tt > 0:
                for k, v in prompt_categories.items():
                    self.categories[k] = self.categories.get(k, 0.0) + (v / tt) * p
        self.categories["completion"] = self.categories.get("completion", 0.0) + c

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "used_token_fallback": self.used_fallback,
            "token_categories": {k: round(v, 1) for k, v in self.categories.items()},
        }


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


_ARM_NUMERIC_FIELDS = (
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "turns",
    "tool_calls",
    "latency_s",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
)


def _arm_means(rows: list[dict], arm: str, model: str | None = None) -> dict:
    arms = [r["arms"][arm] for r in rows if arm in r.get("arms", {})]
    out: dict = {}
    for f in _ARM_NUMERIC_FIELDS:
        out[f"mean_{f}"] = round(_mean([a.get(f) for a in arms]), 4)
    out["mean_correctness"] = round(
        _mean([a.get("judge", {}).get("correctness") for a in arms]), 4
    )
    out["mean_completeness"] = round(
        _mean([a.get("judge", {}).get("completeness") for a in arms]), 4
    )
    mean_tokens = out["mean_total_tokens"] or 0.0
    out["quality_per_1k_tokens"] = round(
        (out["mean_correctness"] / (mean_tokens / 1000.0)) if mean_tokens else 0.0,
        4,
    )
    # Per-category token means + percentage-of-total breakdown.
    cat_means = {
        c: round(_mean([a.get("token_categories", {}).get(c) for a in arms]), 1)
        for c in CATEGORIES
    }
    out["token_categories_mean"] = cat_means
    out["token_categories_pct"] = {
        c: round(cat_means[c] / mean_tokens * 100, 1) if mean_tokens else 0.0
        for c in CATEGORIES
    }
    # Dollar cost — the real driver. Per-query USD averaged over the
    # arm's rows (uses each row's cached/reasoning detail), plus a quality-per-USD
    # efficiency figure mirroring quality_per_1k_tokens. cost_verified surfaces
    # the price table's confidence so unverified-price models can be flagged.
    from treeloom.domain.benchmark.cost_model import compute_cost, price_for
    costs = [compute_cost(a, model) for a in arms] if model else []
    costs = [c for c in costs if c is not None]
    if costs:
        out["mean_cost_usd"] = round(_mean(costs), 6)
        out["quality_per_usd"] = round(
            out["mean_correctness"] / out["mean_cost_usd"], 2
        ) if out["mean_cost_usd"] else 0.0
        p = price_for(model)
        out["cost_verified"] = bool(p and p["verified"])
    elif model:
        # No price entry: previously `mean_cost_usd` and `cost_verified` were
        # simply ABSENT, so a paid run reported no cost at all and nothing
        # said why. Name the model instead — an unpriced run is the one you
        # most need to notice, because it is the one costing money.
        out["cost_unpriced"] = model
    out["n"] = len(arms)
    return out


def _win_rate(rows: list[dict], a: str, b: str, key: str) -> float:
    """Fraction of queries where arm `a`'s `key` strictly beats arm `b`'s.

    Delegates counting to ``win_loss_tie``; the denominator is
    wins + losses + ties (every row where both arms have a non-None value).
    """
    wins, losses, ties = win_loss_tie(rows, a, b, key)
    total = wins + losses + ties
    return round(wins / total, 4) if total else 0.0


def _field(arm: dict, key: str):
    if key in ("correctness", "completeness"):
        return arm.get("judge", {}).get(key)
    return arm.get(key)


def detect_zero_match_arms(rows: list[dict], arms: list[str]) -> list[dict]:
    """Flag arms that read >=1 file across the whole run yet matched ZERO
    ground-truth files in any query (recall stayed 0 everywhere).

    This is the zero-match signature: a path/ground-truth normalization bug makes the
    retrieved-vs-relevant comparison never string-match, so recall reads 0.0 for
    a fully-functional retriever. Real "the retriever is just bad" cases still
    land *some* hits across n queries; reads-without-a-single-match across the
    entire run is almost always a harness path artifact, not retrieval quality.

    Returns one dict per suspect arm (arm name, total reads, total ground-truth
    matches==0) for the caller to surface; also emits a loud `warnings.warn`.
    """
    suspects: list[dict] = []
    for arm in arms:
        arm_rows = [r["arms"][arm] for r in rows if arm in r.get("arms", {})]
        if not arm_rows:
            continue
        total_reads = sum(len(a.get("retrieved_files", []) or []) for a in arm_rows)
        # A row "matched" iff any recall@k > 0 (recall is 0 unless a relevant
        # file was retrieved).
        total_matches = sum(
            1
            for a in arm_rows
            if any(
                (a.get(k) or 0) > 0 for k in ("recall@1", "recall@5", "recall@10")
            )
        )
        if total_reads > 0 and total_matches == 0:
            warnings.warn(
                f"[agentic] WARNING: arm '{arm}' read {total_reads} file(s) across "
                f"{len(arm_rows)} queries but matched ZERO ground-truth files "
                f"(recall@k == 0 everywhere). This is almost certainly a path / "
                f"ground-truth normalization bug, NOT real retrieval "
                f"quality — check that relevant_files and retrieved_files reduce "
                f"to the same repo-relative form.",
                stacklevel=2,
            )
            suspects.append(
                {"arm": arm, "total_reads": total_reads, "total_matches": 0,
                 "n_queries": len(arm_rows)}
            )
    return suspects


def aggregate(rows: list[dict], arms: list[str], *, repo: str, model: str) -> dict:
    """Build the comparison summary across all per-query result rows."""
    summary: dict = {
        "repo": repo,
        "model": model,
        "n_queries": len(rows),
        "arms": arms,
    }
    for arm in arms:
        summary[arm] = _arm_means(rows, arm, model)

    # zero-match guard: an arm that read files but never matched a ground-truth file
    # across the entire run is a path/GT normalization bug, not a retriever that
    # is merely bad. Surface it loudly in the summary.
    zero_match = detect_zero_match_arms(rows, arms)
    if zero_match:
        summary["zero_match_warning"] = zero_match

    # Legacy comparison block — kept verbatim for the grep vs treeloom pairing
    # (existing dashboards/scripts read these keys).
    if "grep" in arms and "treeloom" in arms:
        g = summary["grep"]["mean_total_tokens"] or 0.0
        t = summary["treeloom"]["mean_total_tokens"] or 0.0
        _corr_w, _corr_l, _corr_t = win_loss_tie(rows, "treeloom", "grep", "correctness")
        _rec5_w, _rec5_l, _rec5_t = win_loss_tie(rows, "treeloom", "grep", "recall@5")
        _corr_total = _corr_w + _corr_l + _corr_t
        _rec5_total = _rec5_w + _rec5_l + _rec5_t
        gc = summary["grep"].get("mean_cost_usd")
        tc = summary["treeloom"].get("mean_cost_usd")
        summary["comparison"] = {
            "token_ratio_treeloom_over_grep": round(t / g, 4) if g else 0.0,
            "tokens_saved_pct": round((g - t) / g * 100, 2) if g else 0.0,
            "cost_ratio_treeloom_over_grep": round(tc / gc, 4) if gc else None,
            "cost_saved_pct": round((gc - tc) / gc * 100, 2) if gc else None,
            "correctness_win_rate_treeloom": {
                "win_rate": round(_corr_w / _corr_total, 4) if _corr_total else 0.0,
                "wins": _corr_w,
                "losses": _corr_l,
                "ties": _corr_t,
                "p_value": sign_test(_corr_w, _corr_l),
            },
            "recall@5_win_rate_treeloom": {
                "win_rate": round(_rec5_w / _rec5_total, 4) if _rec5_total else 0.0,
                "wins": _rec5_w,
                "losses": _rec5_l,
                "ties": _rec5_t,
                "p_value": sign_test(_rec5_w, _rec5_l),
            },
        }

    # Generic pairwise comparisons for any arm combination (e.g. treeloom vs
    # claude-context). For each pair in `arms` order the earlier arm is the
    # baseline and the later the challenger: token_ratio = challenger/baseline,
    # win rates = fraction of queries where the challenger strictly beats it.
    if len(arms) >= 2:
        comparisons: dict = {}
        for i, base in enumerate(arms):
            for chal in arms[i + 1:]:
                b = summary[base]["mean_total_tokens"] or 0.0
                c = summary[chal]["mean_total_tokens"] or 0.0
                _cw, _cl, _ct = win_loss_tie(rows, chal, base, "correctness")
                _rw, _rl, _rt = win_loss_tie(rows, chal, base, "recall@5")
                _c_total = _cw + _cl + _ct
                _r_total = _rw + _rl + _rt
                bc = summary[base].get("mean_cost_usd")
                cc = summary[chal].get("mean_cost_usd")
                comparisons[f"{chal}_vs_{base}"] = {
                    "token_ratio": round(c / b, 4) if b else 0.0,
                    "tokens_saved_pct": round((b - c) / b * 100, 2) if b else 0.0,
                    "cost_ratio": round(cc / bc, 4) if bc else None,
                    "cost_saved_pct": round((bc - cc) / bc * 100, 2) if bc else None,
                    "correctness_win_rate": {
                        "win_rate": round(_cw / _c_total, 4) if _c_total else 0.0,
                        "wins": _cw,
                        "losses": _cl,
                        "ties": _ct,
                        "p_value": sign_test(_cw, _cl),
                    },
                    "recall@5_win_rate": {
                        "win_rate": round(_rw / _r_total, 4) if _r_total else 0.0,
                        "wins": _rw,
                        "losses": _rl,
                        "ties": _rt,
                        "p_value": sign_test(_rw, _rl),
                    },
                }
        summary["comparisons"] = comparisons
    return summary
