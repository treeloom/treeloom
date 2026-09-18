"""Benchmark domain: multi-repo agentic rollup + markdown table formatters — pure, no I/O."""
from __future__ import annotations

from .significance import sign_test


def multi_repo_rollup(summaries: list[dict]) -> dict:
    """Aggregate per-repo agentic summary dicts into a multi-repo rollup.

    Parameters
    ----------
    summaries:
        List of dicts produced by ``agent_metrics.aggregate`` (i.e. the
        ``_summary.json`` files written by the agentic runner).  Each must have
        at least ``repo``, ``n_queries``, and per-arm means.  Summaries that
        have no ``comparison`` block (arms were not grep + treeloom) are
        included in per-repo rows with whatever means are available but
        contribute nothing to the pooled comparison counts.

    Returns
    -------
    dict with keys:
        ``"repos"``  — list of per-repo row dicts
        ``"pooled"`` — weighted-average means + summed win/loss/tie counts
                       and a pooled sign-test p_value
    """
    repos: list[dict] = []

    total_queries = 0
    # Accumulators for n_queries-weighted means
    w_treeloom_tokens = 0.0
    w_grep_tokens = 0.0
    w_treeloom_recall5 = 0.0
    w_grep_recall5 = 0.0
    w_treeloom_corr = 0.0
    w_grep_corr = 0.0

    # Pooled win/loss/tie counts (sum across repos)
    corr_wins = corr_losses = corr_ties = 0
    rec5_wins = rec5_losses = rec5_ties = 0

    for s in summaries:
        repo = s.get("repo", "unknown")
        n = s.get("n_queries", 0)
        total_queries += n

        grep_arm = s.get("grep", {})
        tl_arm = s.get("treeloom", {})

        grep_tokens = grep_arm.get("mean_total_tokens") or 0.0
        tl_tokens = tl_arm.get("mean_total_tokens") or 0.0
        grep_recall5 = grep_arm.get("mean_recall@5") or 0.0
        tl_recall5 = tl_arm.get("mean_recall@5") or 0.0
        grep_corr = grep_arm.get("mean_correctness") or 0.0
        tl_corr = tl_arm.get("mean_correctness") or 0.0

        comp = s.get("comparison")

        # Pull win/loss/tie/p_value from comparison block.  Handles both the
        # new dict shape (Task A) and the legacy flat float shape gracefully.
        if comp and isinstance(comp.get("correctness_win_rate_treeloom"), dict):
            cwd = comp["correctness_win_rate_treeloom"]
            rwd = comp["recall@5_win_rate_treeloom"]
            c_wins = cwd.get("wins", 0)
            c_losses = cwd.get("losses", 0)
            c_ties = cwd.get("ties", 0)
            c_p = cwd.get("p_value", 1.0)
            r_wins = rwd.get("wins", 0)
            r_losses = rwd.get("losses", 0)
            r_ties = rwd.get("ties", 0)
            r_p = rwd.get("p_value", 1.0)
            tokens_saved_pct = comp.get("tokens_saved_pct") or 0.0
        else:
            # Legacy / missing comparison — no resolved counts available.
            c_wins = c_losses = c_ties = 0
            c_p = 1.0
            r_wins = r_losses = r_ties = 0
            r_p = 1.0
            tokens_saved_pct = (
                round((grep_tokens - tl_tokens) / grep_tokens * 100, 2)
                if grep_tokens else 0.0
            )

        repos.append({
            "repo": repo,
            "n_queries": n,
            "grep_mean_total_tokens": round(grep_tokens, 4),
            "treeloom_mean_total_tokens": round(tl_tokens, 4),
            "tokens_saved_pct": round(tokens_saved_pct, 2),
            "grep_mean_recall@5": round(grep_recall5, 4),
            "treeloom_mean_recall@5": round(tl_recall5, 4),
            "grep_mean_correctness": round(grep_corr, 4),
            "treeloom_mean_correctness": round(tl_corr, 4),
            "correctness": {
                "win_rate": round(c_wins / (c_wins + c_losses + c_ties), 4)
                if (c_wins + c_losses + c_ties) else 0.0,
                "wins": c_wins,
                "losses": c_losses,
                "ties": c_ties,
                "p_value": round(c_p, 4),
            },
            "recall@5": {
                "win_rate": round(r_wins / (r_wins + r_losses + r_ties), 4)
                if (r_wins + r_losses + r_ties) else 0.0,
                "wins": r_wins,
                "losses": r_losses,
                "ties": r_ties,
                "p_value": round(r_p, 4),
            },
        })

        # Accumulate weighted sums
        w_treeloom_tokens += tl_tokens * n
        w_grep_tokens += grep_tokens * n
        w_treeloom_recall5 += tl_recall5 * n
        w_grep_recall5 += grep_recall5 * n
        w_treeloom_corr += tl_corr * n
        w_grep_corr += grep_corr * n

        corr_wins += c_wins
        corr_losses += c_losses
        corr_ties += c_ties
        rec5_wins += r_wins
        rec5_losses += r_losses
        rec5_ties += r_ties

    # Compute pooled means
    if total_queries > 0:
        pooled_tl_tokens = round(w_treeloom_tokens / total_queries, 4)
        pooled_grep_tokens = round(w_grep_tokens / total_queries, 4)
        pooled_tl_recall5 = round(w_treeloom_recall5 / total_queries, 4)
        pooled_grep_recall5 = round(w_grep_recall5 / total_queries, 4)
        pooled_tl_corr = round(w_treeloom_corr / total_queries, 4)
        pooled_grep_corr = round(w_grep_corr / total_queries, 4)
    else:
        pooled_tl_tokens = pooled_grep_tokens = 0.0
        pooled_tl_recall5 = pooled_grep_recall5 = 0.0
        pooled_tl_corr = pooled_grep_corr = 0.0

    pooled_saved_pct = (
        round((pooled_grep_tokens - pooled_tl_tokens) / pooled_grep_tokens * 100, 2)
        if pooled_grep_tokens else 0.0
    )

    pooled_corr_p = sign_test(corr_wins, corr_losses)
    pooled_rec5_p = sign_test(rec5_wins, rec5_losses)

    pooled = {
        "n_repos": len(summaries),
        "total_queries": total_queries,
        "treeloom_mean_total_tokens": pooled_tl_tokens,
        "grep_mean_total_tokens": pooled_grep_tokens,
        "tokens_saved_pct": pooled_saved_pct,
        "treeloom_mean_recall@5": pooled_tl_recall5,
        "grep_mean_recall@5": pooled_grep_recall5,
        "treeloom_mean_correctness": pooled_tl_corr,
        "grep_mean_correctness": pooled_grep_corr,
        "correctness": {
            "wins": corr_wins,
            "losses": corr_losses,
            "ties": corr_ties,
            "p_value": round(pooled_corr_p, 4),
        },
        "recall@5": {
            "wins": rec5_wins,
            "losses": rec5_losses,
            "ties": rec5_ties,
            "p_value": round(pooled_rec5_p, 4),
        },
    }

    return {"repos": repos, "pooled": pooled}


def format_comparison_table(summary: dict) -> str:
    """Return a GitHub-flavored markdown table for a single agentic summary.

    Columns: Metric | grep | treeloom | Δ / win-rate | p-value

    If the summary lacks a ``comparison`` block (e.g. arms weren't grep +
    treeloom), returns a short explanatory note instead of raising.
    """
    comp = summary.get("comparison")
    grep_arm = summary.get("grep", {})
    tl_arm = summary.get("treeloom", {})

    if not comp or not grep_arm or not tl_arm:
        arms = summary.get("arms", [])
        return (
            f"_No grep-vs-treeloom comparison available "
            f"(arms: {', '.join(arms) or 'unknown'})._"
        )

    grep_tokens = grep_arm.get("mean_total_tokens", 0.0) or 0.0
    tl_tokens = tl_arm.get("mean_total_tokens", 0.0) or 0.0
    grep_recall5 = grep_arm.get("mean_recall@5", 0.0) or 0.0
    tl_recall5 = tl_arm.get("mean_recall@5", 0.0) or 0.0
    grep_corr = grep_arm.get("mean_correctness", 0.0) or 0.0
    tl_corr = tl_arm.get("mean_correctness", 0.0) or 0.0

    tokens_saved_pct = comp.get("tokens_saved_pct", 0.0)
    token_ratio = comp.get("token_ratio_treeloom_over_grep", 0.0)

    # Win-rate blocks — handle both dict (new) and float (legacy) shapes.
    def _wr_info(val):
        if isinstance(val, dict):
            return val.get("win_rate", 0.0), val.get("p_value", None)
        return val, None

    corr_val = comp.get("correctness_win_rate_treeloom", 0.0)
    rec5_val = comp.get("recall@5_win_rate_treeloom", 0.0)
    corr_wr, corr_p = _wr_info(corr_val)
    rec5_wr, rec5_p = _wr_info(rec5_val)

    def _p_str(p):
        if p is None:
            return "—"
        return f"{p:.4f}"

    n = summary.get("n_queries", "?")
    repo = summary.get("repo", "?")

    header = (
        f"\n### Agentic benchmark: **{repo}** (n={n})\n\n"
        "| Metric | grep | treeloom | Δ / win-rate | p-value |\n"
        "|--------|-----:|----------:|:------------:|--------:|\n"
    )

    token_delta = f"{tokens_saved_pct:+.1f}% saved (ratio {token_ratio:.3f})"
    rows = [
        f"| Mean total tokens | {grep_tokens:,.0f} | {tl_tokens:,.0f} | {token_delta} | — |",
        f"| Mean recall@5 | {grep_recall5:.4f} | {tl_recall5:.4f} | win-rate {rec5_wr:.4f} | {_p_str(rec5_p)} |",
        f"| Mean correctness | {grep_corr:.4f} | {tl_corr:.4f} | win-rate {corr_wr:.4f} | {_p_str(corr_p)} |",
    ]

    return header + "\n".join(rows) + "\n"


def format_rollup_table(rollup: dict) -> str:
    """Return a GitHub-flavored markdown rollup table.

    One row per repo (repo, n, treeloom vs grep tokens, tokens-saved%,
    recall@5 win-rate + p, correctness win-rate + p) plus a final **Pooled**
    row.
    """
    repos = rollup.get("repos", [])
    pooled = rollup.get("pooled", {})

    header = (
        "| Repo | n | grep tok | treeloom tok | saved% "
        "| r@5 win-rt | r@5 p | corr win-rt | corr p |\n"
        "|------|--:|---------:|-------------:|------:"
        "|----------:|------:|------------:|-------:|\n"
    )

    def _row(label: str, n, grep_tok, tl_tok, saved_pct, rec5, corr) -> str:
        rec5_wr = rec5.get("win_rate", 0.0)
        rec5_p = rec5.get("p_value", 1.0)
        corr_wr = corr.get("win_rate", 0.0)
        corr_p = corr.get("p_value", 1.0)
        return (
            f"| {label} | {n} "
            f"| {grep_tok:,.0f} | {tl_tok:,.0f} | {saved_pct:+.1f}% "
            f"| {rec5_wr:.4f} | {rec5_p:.4f} "
            f"| {corr_wr:.4f} | {corr_p:.4f} |"
        )

    lines = []
    for r in repos:
        lines.append(_row(
            r.get("repo", "?"),
            r.get("n_queries", 0),
            r.get("grep_mean_total_tokens", 0.0),
            r.get("treeloom_mean_total_tokens", 0.0),
            r.get("tokens_saved_pct", 0.0),
            r.get("recall@5", {}),
            r.get("correctness", {}),
        ))

    # Pooled row — no per-repo win_rate (use wins/(wins+losses+ties))
    p_rec5 = pooled.get("recall@5", {})
    p_corr = pooled.get("correctness", {})
    p_rec5_total = p_rec5.get("wins", 0) + p_rec5.get("losses", 0) + p_rec5.get("ties", 0)
    p_corr_total = p_corr.get("wins", 0) + p_corr.get("losses", 0) + p_corr.get("ties", 0)
    p_rec5_wr = round(p_rec5.get("wins", 0) / p_rec5_total, 4) if p_rec5_total else 0.0
    p_corr_wr = round(p_corr.get("wins", 0) / p_corr_total, 4) if p_corr_total else 0.0
    pooled_row = _row(
        "**Pooled**",
        pooled.get("total_queries", 0),
        pooled.get("grep_mean_total_tokens", 0.0),
        pooled.get("treeloom_mean_total_tokens", 0.0),
        pooled.get("tokens_saved_pct", 0.0),
        {"win_rate": p_rec5_wr, "p_value": p_rec5.get("p_value", 1.0)},
        {"win_rate": p_corr_wr, "p_value": p_corr.get("p_value", 1.0)},
    )

    return header + "\n".join(lines) + "\n" + pooled_row + "\n"
