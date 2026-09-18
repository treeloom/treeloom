# Cost vs Quality Across Models: the value prop, in dollars

> **Superseded for current models (2026-09-21):** [benchmark-results-2026-09.md](benchmark-results-2026-09.md)
> re-ran this comparison on deepseek-flash, claude-sonnet-5 and gpt-5.5 against featbit
> pinned at `v5.4.9`, with an independent judge. The cost verdict holds and is larger
> (−11% to −47%, significant per query); the "and quality" half does not — correctness is
> statistically tied in every cell. Opus was not re-run; its −16.3% below is the latest
> Opus figure. The numbers below remain accurate for the configuration and date they state,
> but they were produced on the pre-2026-07-05 benchmark harness (ReAct text protocol,
> `deepseek-chat` judging its own tier) and are **not directly comparable** with later
> native-tool-call runs.

This is a dated (2026-06-18) study for readers asking "is treeloom worth it in dollars,
on the model I actually use?" A **cell** is one benchmark run of both **arms** (the
agent with grep/glob/read tools vs. with treeloom `search_code`) over 100 queries on
one agent model; **correctness** is a 1–5 LLM-judge score against a cached gold answer.

**Question.** treeloom's whole reason to exist: does it reduce the **cost** and/or
raise the **quality** of code search versus the default **grep / ripgrep /
read_file** agent loop — and does that hold **across models**? Tokens are a proxy;
the real driver is **dollars**, which (unlike raw token counts) *are* comparable
across models.

**Short answer.** Yes, at every tier measured. With per-model cache-aware pricing,
**`treeloom-full` beats the grep/read agent on both cost (−6% to −24%) and quality
(+0.2 to +0.3 correctness), with consistently fewer turns** — including on the most
expensive frontier model (Opus 4.8).

This extends the earlier token-based, deepseek-only finding (see
[benchmark-findings.md](benchmark-findings.md)) along the two axes it didn't cover: the
**model spectrum** and **dollars instead of tokens**.

---

## Results

featbit-clean-100, `--obs-char-limit 16000`, arms `grep` vs `treeloom` (full),
judge fixed at `deepseek-chat`. Cost is real, **cache-aware** USD from the captured
provider `usage` (OpenAI/DeepSeek auto-cache; Anthropic via the native Messages API
with `cache_control` on the transcript). Prices confirmed 2026-06-18 (see
`domain/benchmark/cost_model.py`).

| agent model | grep $/q | treeloom $/q | **cost saved** | corr (grep→tl) | turns (grep→tl) | recall@5 | quality/USD |
|---|---|---|---|---|---|---|---|
| deepseek-chat | 0.0039 | 0.0037 | **−6.5%** | 3.84 → 4.14 | 6.8 → 4.5 | 0.86 → 0.84 | 981 → 1130 |
| claude-sonnet-4-6 | 0.0985 | 0.0924 | **−6.2%** | 3.62 → 3.83 | 6.9 → 4.9 | 0.82 → 0.87 | 37 → 42 |
| gpt-5.5 | 0.0844 | 0.0645 | **−23.5%** | 4.11 → 4.37 | 4.1 → 2.6 | 0.76 → 0.84 | 49 → 68 |
| claude-opus-4-8 † | 0.125 | 0.104 | **−16.3%** | 4.30 → 4.58 | 4.2 → 2.5 | 0.75 → 0.81 | 35 → 44 |

† Opus n=67: partway through the run the provider's API began returning HTTP 400
on every request, so 33 of 100 queries produced no answer from either arm and are
excluded as error rows. The table shows the 67 valid paired queries, which are
on-trend with the first 41 completed before the interruption. The other three
models are full n=100.

**Significance:** deepseek correctness win is significant (sign test p=0.014). The
others are directional (treeloom wins more queries than it loses) but dominated by
**ties** — the stronger the model, the more often both arms reach the same answer,
so the separation shows up in **cost and turns** more than in a binary correctness
flip.

---

## What the numbers say

- **treeloom is cheaper on every model**, by −6% (deepseek/sonnet) up to −24%
  (gpt-5.5). Note treeloom's *payload* is similar-or-larger than grep's, yet it's
  cheaper end-to-end: it resolves in **fewer turns** (≈2–4 vs ≈4–7), and prompt
  caching makes the re-sent transcript cheap. grep pays for *thrash* — extra
  grep/read round-trips, and on gpt-5.5 a 17k-vs-9.7k-token blowup.
- **treeloom is at least as good on quality, usually better** (+0.2 to +0.3 mean
  correctness, higher recall@5), and never worse.
- **quality-per-dollar favours treeloom on every model** (e.g. gpt-5.5 68 vs 49,
  deepseek 1130 vs 981).
- **The win holds for the frontier tier** (Opus 4.8, gpt-5.5) — the models where
  per-query dollars actually sting — not just the cheap models.

## Token volume & caching efficiency (read before citing absolute Claude $)

Agentic input-token volume is **inherently huge**: a multi-turn agent re-sends the
*entire growing transcript every turn*, so the ~16k-token search payloads are
re-fed on each subsequent turn. The opus cell alone (67 queries × 2 arms) sent
**~2.1M input tokens** — that is normal for an agent loop, not an anomaly. (An
account-level dashboard will show *more* than the per-query totals here: it also
includes aborted and retried attempts — an earlier opus run stopped mid-stream and
the 33 interrupted queries above — which are correctly **excluded** from the
per-query cost.)

**Prompt caching re-prices that volume, but its benefit scales with loop length.**
The cache_write (creation) premium is 1.25× input; cache_read is 0.1×. You only come
out ahead when cached content is **re-read enough times** to amortize the write
premium. On the opus cell, cache_write was actually the **largest** cost line
($7.08) — because opus resolves in only ~2.5 turns, so content is re-read barely
once. Caching still netted ~25% off input ($7.99 vs $10.72 uncached), but **far
less than the 10× headline** — short frontier-model loops under-amortize it; long
loops (deepseek/grep at 7+ turns) benefit much more. **Absolute Claude dollars
therefore carry a caching-efficiency caveat; the treeloom-vs-grep *delta* does not**
(both arms use identical caching). A smarter cache-breakpoint policy (cache only
when re-reads will pay for the write) is a known future optimization.

## Benchmark economics (which model to run)

The cost spread *between* models is enormous, and it dictates how to use this
harness day-to-day:

| model | cost for a full 100-query cell (grep + treeloom) |
|---|---|
| deepseek-chat | **~$0.39** |
| gpt-5.5 | ~$15 |
| claude-sonnet-4-6 | ~$18 |
| claude-opus-4-8 | **~$23** (≈$0.23/query × 2 arms) |

An opus cell is **~60× a deepseek cell**. Guidance:

- **Default to `deepseek-chat` for iteration** — it's ~1/60th the cost and gave the
  *statistically significant* correctness result here (p=0.014). Develop and
  regression-test on it.
- **Spend frontier dollars (opus/gpt-5.5) sparingly** — only when *cross-model
  generalization* is the specific question (as in this study). One confirming run
  is enough; don't re-run a frontier cell to nudge n.
- Set provider-side limits (spend caps, quotas) per frontier cell up front, and
  read the body of any mid-run HTTP 400: a provider-side rejection names its cause
  only there, looks exactly like a code bug from the harness's side, and silently
  truncates the cell.

## Caveats (stated plainly)

- One repo (featbit), one judge model (deepseek-chat), one query set
  (clean-100). Prices as of 2026-06-18 (`PRICE_CAPTURE_DATE`); re-price before
  re-citing absolute dollars.
- Opus cell is n=67, not a full 100 — the run was cut short by provider-side API
  rejections (see the table footnote).
- gpt-5.5 ran at `REASONING_EFFORT=low` (pinned; effort affects both arms equally,
  so the delta is fair, but absolute reasoning-token cost would rise at higher
  effort).
- A weak local anchor (Qwen2.5-Coder-3B) was attempted and dropped: it is too weak
  to drive the ReAct tool protocol, so both arms floor (corr ~1, recall 0) at ~$0
  API cost — below some capability floor the search method is irrelevant because the
  model can't use either tool.

## Reproduce

```bash
# per model (judge fixed; --model routes the agent via resolve_model_route)
python -m treeloom.benchmark agentic \
  --queries benchmarks/queries/featbit-clean-100.jsonl \
  --repo <featbit> --search-url http://localhost:8001 \
  --arms grep,treeloom --model <MODEL> --judge-model deepseek-chat \
  --max-turns 12 --obs-char-limit 16000
```

Cost is computed by `domain/benchmark/cost_model.py` from the captured cache/reasoning
token detail (`TokenAccount`); the summary reports `mean_cost_usd`, `quality_per_usd`,
and `cost_saved_pct`. Result data was written to `benchmarks/results/agentic/` (local run
output; that directory is gitignored and not in the repository).

Related: [benchmark-findings.md](benchmark-findings.md) (the deepseek-only token precedent
this generalizes) and [token-optimization-rejected.md](token-optimization-rejected.md) (the
serialization defaults that were kept, and why the payload is not trimmed).
