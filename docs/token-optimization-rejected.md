# Token Optimization: What We Tried, What We Rejected, and Why

This is the negative-results ledger for shrinking treeloom's search response. Read it
before proposing any change that makes `search_code` return less: every idea below was
built, benchmarked with a real LLM agent, and rejected, and the opt-in flags it left
behind are documented here so they aren't mistaken for dead code. Terms: an **arm** is
one tool configuration for the agent (`grep` = grep/glob/read tools; `treeloom` and its
variants = treeloom `search_code` in a given `response_mode`); a **cell** is one run of
every arm over one query set with one agent model; **need_more** is the fraction of
queries where a single-call screen (below) says the payload was insufficient to answer.

**Question.** treeloom's search response is ~79% of an agent's token cost (measured:
14.8k of 18.7k tokens on featbit, n=100), and that payload is re-sent every turn — so
cost grows roughly quadratically in turns. The obvious lever: make the search response
smaller. We spent a full milestone testing that obvious lever every way we could think
of.

**Short answer.** **Don't trim the search payload. `treeloom-full` is already the
token-efficient configuration in an agent loop.** Every payload-shrinking idea lost or
washed — not because the payloads weren't smaller (they were, by 30–70%), but because a
smaller response makes the agent declare it needs more and `read_file`/re-query, which
raises **mean turns** and erases (or reverses) the saving. We call this **compensatory
fetch**, and it is the through-line of every result below. The one optimization that
*did* stick was **serialization** (how we encode the payload), not **content** (what's
in it) — because serialization removes bytes the agent never re-fetches.

This document records the rejected experiments deliberately. A negative result that
cost real measurement is evidence, not waste; preserving it is the point.

---

## The governing metric: mean turns × payload

The mistake that makes payload trimming *look* like a win is measuring a single search
response in isolation. It shrinks — of course it does. But the agent operates in a
**loop**, and the transcript (including every prior search response) is re-sent on every
turn. So the real cost is driven by **how many turns** the agent takes, and trimming the
payload tends to *add* turns.

> **The decision rule, used for every experiment below: a token drop accompanied by a
> turn rise is a NET LOSS.** Judge on `mean turns` first, payload tokens second.

This is why the headline number for each rejected idea is the **turn** delta, not the
byte delta.

---

## What we KEPT (the only wins were serialization, not content)

The serialization changes below shipped on by default because none of them removes
information the agent then re-fetches:

| change | effect | why it's safe |
|---|---|---|
| markdown `response_format` | **−26%** per payload | pure encoding change — same information, cheaper to tokenize than JSON |
| `commit_sha` → top-level `sources` dedup | smaller payload | provenance was repeated per-chunk; dedup loses nothing |
| snippet blank-line / trailing-whitespace strip | small | line ranges live in metadata; nothing lost |

These are content-preserving. Everything below is content-*removing*, and that's the
difference.

---

## What we REJECTED (every content trim)

### `response_mode=facet` — drop all code bodies, keep headers

Return only metadata + the chunk header (enclosing class + defined symbols + signature);
the agent fetches bodies via `read_file`. **Result: turns +18%, tokens +20% — rejected.**
The agent almost always needs the implementation, so dropping it just forces an extra
`read_file` turn (which returns *more* than the original snippet). Classic compensatory
fetch, and the worst offender.

### `response_mode=summary_tail` — top-2 full snippets, tail replaced by LLM summary

The gentler cousin: keep the top ranks at full fidelity, replace lower-ranked bodies with
the indexed one-sentence summary. A cautionary tale in measurement:

1. **First look:** turns flat, ~−7% tokens — looked like a marginal opt-in win.
2. **Re-run overturned it:** the −7% was a **cache artifact** — the terse summaries weren't
   actually indexed for that repo, so it had silently fallen back to full snippets. With
   the cache populated it was **+5.3% / +5.7% tokens** — a loss.
3. **Verbosity sweep:** maybe the summaries were the wrong length? Swept one-line /
   brief / medium tiers. **It loses at every tier** — brief was worst at **+8.8% tokens,
   +0.35 turns.** A one-sentence summary is ~15–40 tokens; for a short tail chunk it's
   nearly as big as the snippet it replaced, so it degrades information without saving
   much — and still triggers the re-fetch.

### Smaller trims: community-off / adaptive_topk / strip_imports / query_class_payload

Four smaller, well-motivated trims (drop the community-summary block; return fewer hits;
strip import lines; reshape class-level payload). **Combined vs baseline (two cells —
symbol-bearing and symbol-free, n=100 each — `--obs-char-limit 16000`): turns ROSE in both cells (clean +0.16, symbol-free +0.28),
tokens +4.9% / −1.4% — net loss by the turns gate.** All shipped as opt-in, default-off.

### Combinations of the above

We had only ever tested trims individually. Maybe they stack favorably? Screened the three
non-redundant combinations (most of the grid is redundant — `facet` makes `strip_imports`
and `summary_tail` no-ops, and `facet`/`summary_tail` are mutually exclusive):

| combo | need_more | answered/100 | payload Δ |
|---|---|---|---|
| `full` baseline | 0.03 | 97 | — |
| summary_tail + strip_imports + community-off + adaptive_topk | 0.05 | 95 | −41% |
| **facet + community-off + adaptive_topk** | **0.72** | **28** | −69% |
| summary_tail + community-off | 0.05 | 95 | −39% |

**Stacking doesn't compound** (the 4-trim combo screens identically to the 2-trim one —
summary_tail dominates, the rest is noise), and the facet stack is **catastrophic** (it
could only answer 28 of 100 queries). No combination crossed the bar — the sixth time
compensatory fetch showed up in these experiments.

### Does it depend on the model?

The whole verdict had been measured on one agent model (`deepseek-chat` drove 55 of the
60 agentic runs behind this document).
The natural objection: *maybe a more capable model is selective enough that trimming
pays.* We built a cross-model adapter and screened `full` / `facet` /
`summary_tail` across the capability spectrum, judge fixed:

| model | tier | `full` need_more | `facet` need_more | corr (full) |
|---|---|---|---|---|
| Qwen2.5-Coder-3B | weak | 0.02 | 0.14 | 1.09 (floor) |
| deepseek-chat | mid | 0.02 | 0.63 | 3.71 |
| claude-sonnet-4-6 | high | 0.03 | 0.59 | 3.90 |
| claude-opus-4-8 | frontier | 0.09 | **0.84** | 4.26 |
| gpt-5.5 | frontier-reasoning | 0.13 | **0.86** | 4.23 |

**The hypothesis was refuted — and the opposite is true.** The frontier reasoning models
reject body-less payloads *hardest* (facet need_more 0.84 / 0.86) and are the pickiest
even on full snippets — they are *more* rigorous about declaring a payload insufficient,
not more willing to make do with less. The weak model's low facet need_more (0.14) is a
**false negative**: it answers at correctness ~1.1 in every mode — it doesn't recognize
insufficiency, it just answers badly. So trimming gets **worse, not better, as capability
rises**, and the case for `treeloom-full` is *strongest* for the high-value frontier
models.

### facet + on-demand batch hydrate — the most charitable variant

The standing objection to facet's rejection: it was only ever paired with `read_file`
(one whole-range read per file). What if we give the agent a purpose-built tool to
rehydrate the *exact* dropped bodies in a single **batched** call — bounding compensatory
fetch at a fixed +1 turn (search→hydrate→answer)? We built it (a `hit_id` per facet hit, a
`/hydrate-chunks` endpoint, an MCP tool) and benchmarked it. n=100, agent genuinely used
it (248 hydrate calls):

| arm | turns | tokens | judge | recall@5 | 12-turn caps |
|---|---|---|---|---|---|
| **treeloom (full)** | **4.5** | **26,329** | **4.21** | 0.84 | **4** |
| treeloom-facet | 6.3 | 34,208 | 3.95 | 0.85 | 16 |
| treeloom-facet-hydrate | 7.8 | 41,798 | 3.72 | 0.87 | 29 |

**It fails *harder* than plain facet** — vs full: turns +3.3, tokens +59%, judge −0.49
(the lowest of any arm, below grep), and **29/100 queries hit the turn cap**. Two findings:

1. **Agents won't even adopt the tool when `read_file` exists** — with both available the
   agent called hydrate **0 times**; we had to remove `read_file` to force the mechanism.
2. **Even forced, it spirals.** The +1-turn theory assumes search→hydrate→answer; in
   practice the agent does search→hydrate→**re-search**→hydrate→… and caps out. The
   catalog→fetch round-trip *is* the cost, and the transcript re-send compounds it.
   recall@5 (0.87) is the only "win" and it's hollow — it can't convert retrieval into
   answers.

The most charitable form of facet loses hardest. The design space is closed.

---

## Methodology rules (the durable lessons)

These cost us real time to learn; they're the reusable part.

1. **Judge on mean turns, not payload bytes.** A token drop with a turn rise is a net
   loss (the compensatory-fetch trap).
2. **Raise `--obs-char-limit` to 16000 when measuring payload deltas.** The default
   4000 cap truncates observations and *hides the very deltas you're testing* — baseline
   read 26k at obs-16000 vs ~20k at the 4000 cap.
3. **Use the sufficiency-proxy screen as a cheap pre-filter**
   (`src/treeloom/application/benchmark/screen.py`): one LLM call/query → answer-or-`NEED_MORE`, ~15 min
   vs ~1 h for an agentic cell. **`need_more_rate` is the trustworthy leading indicator of
   compensatory fetch.** But **`mean_correctness_answered` is confounded by punt rate** —
   a config that punts hard queries via `NEED_MORE` inflates its own mean-over-answered
   (selection bias, not quality). Read need_more *with* correctness, especially for weak
   models.
4. **Verify the actually-served models before any long run.** `.env` and `nvidia-smi`
   both lie — `.env` named a 3B while the box served a 35B; `nvidia-smi` misses
   containerized GPU procs. Confirm via the container startup log + a smoke query.
5. **Tokens are a proxy; cost (dollars) is the real driver, and it's model-agnostic.**
   Raw token counts aren't comparable across models, but dollars are (published
   input/cached/output prices). For reasoning models the search payload is likely a small
   fraction of cost (reasoning tokens dominate) — which makes "don't trim" even more
   correct for them. (Cost study: [cost-across-models.md](cost-across-models.md).)

---

## Open threads (not yet rejected)

- **Cost vs quality vs grep, across models, in dollars** — the real value-prop
  question. Answered since: [cost-across-models.md](cost-across-models.md) (2026-06) and
  the [2026-09 refresh](benchmark-results-2026-09.md).
- **A CodeSearchNet-based credibility benchmark** — considered, but gated on a
  train/test-contamination check of the candidate embedding models, and not pursued.
- **Embedder upgrade** off the now-deprecated `jina-embeddings-v2-base-code`
  (`Qwen3-Embedding-0.6B` already beat it on the clean set; gated on a licence check for
  the candidate replacement).

---

## Reproduce

```bash
# Sufficiency-proxy screen (cheap pre-filter)
#   application/benchmark/screen.py — one LLM call/query, need_more_rate

# Agentic cell (the real measurement) — raise the obs cap when measuring payload deltas
python -m treeloom.benchmark agentic \
  --queries benchmarks/queries/featbit-clean-100.jsonl \
  --repo <featbit> --search-url http://localhost:8001 \
  --arms grep,treeloom,treeloom-facet,treeloom-summary-tail \
  --obs-char-limit 16000
# add --llm-symbol-free for the symbol-free cell
```

Result data was written to `benchmarks/results/agentic/` and
`benchmarks/results/embedding-ab/` (local run output; that directory is gitignored and not
in the repository). The opt-in flags for the rejected modes are retained for
reproducibility (default `full`/off) — they are the evidence, not dead code.
