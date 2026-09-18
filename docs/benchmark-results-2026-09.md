# Benchmark results — 2026-09 refresh (featbit v5.4.9)

**Run:** 2026-09-21. **Corpus:** featbit, public tag `v5.4.9`
(`c16207823d7b26e25bcbbe2a2eb032b602eea519`). **Harness:** treeloom v0.3.0 (this repository).
**Judge and gold answers:** `gpt-4o-2024-08-06` for every cell.

This is the refresh planned in [benchmark-refresh-runbook.md](benchmark-refresh-runbook.md).
It replaces the June 2026 table, which predates the 2026-07-05 harness rebuild
(a different, non-comparable epoch) and was measured against an unpinned working checkout.
Earlier results and their methodology remain in [benchmark-findings.md](benchmark-findings.md).

**How to read this.** An **arm** is one tool configuration for the LLM agent: `grep`
(grep/glob/read_file tools), `treeloom` (treeloom's `search_code`) or `claude-context`
(a third-party semantic code-search MCP server). A **cell** is one run of every arm over
one query set on one agent model; "Cell A" and "Cell B" are the two designs from the
runbook (A = all three arms on the cheapest model, B = grep vs treeloom on each frontier
model). **Symbol-bearing** queries contain the identifier they ask about; **symbol-free**
queries describe behaviour only. **Correctness** is an independent LLM judge's 1–5 score
of the final answer against a cached gold answer; **recall@k** and **MRR** count whether
and how early a ground-truth file appeared among the files the agent read; **turn-cap
hits** are queries where the agent used all 12 tool-call turns without finishing. Every
`p` is a two-sided sign test over per-query wins and losses, ties excluded; "not
significant" means not detectable at n=100, not that the arms are equal.

## Summary

Across three models and two query populations, **treeloom answered as well as a
grep/glob/read agent at 11–47% lower mean cost in every cell**, with fewer turns. The
cost win is significant per query in 5 of 6 cells and on symbol-free queries on every
model (p ≤ 5.6e-07); on symbol-free queries treeloom also exhausts the turn cap
significantly less often. Its correctness score was higher in 5 of 6 cells, **but that
difference is not statistically significant in any cell** (closest: p = 0.061,
gpt-5.5, symbol-free).

The defensible claim from this run is **"the same answer quality at substantially lower
cost"** — not "better answers". See [Relationship to earlier findings](#relationship-to-earlier-findings).

## Headline: treeloom vs grep

Tokens and cost are treeloom's change relative to grep (negative = treeloom cheaper).
W/L/T are per-query wins/losses/ties for treeloom; p is a two-sided sign test with ties excluded.

| Model | Query set | Tokens | Cost | Turns (grep → treeloom) | Turn-cap hits (grep / treeloom) | Correctness W/L/T | Correctness p | Recall@5 W/L | Recall@5 p |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-flash | symbol-bearing | -27% | -11% | 4.7 → 3.6 | 0 / 0 | 15/13/72 | 0.851 | 2/6 | 0.289 |
| deepseek-flash | symbol-free | -44% | -30% | 6.7 → 4.2 | 11 / 4 | 12/7/81 | 0.359 | 11/22 | 0.080 |
| claude-sonnet-5 | symbol-bearing | -27% | -14% | 4.8 → 3.3 | 2 / 1 | 14/8/78 | 0.286 | 4/5 | 1.000 |
| claude-sonnet-5 | symbol-free | -43% | -25% | 6.2 → 3.3 | 11 / 3 | 18/10/72 | 0.185 | 11/13 | 0.839 |
| gpt-5.5 | symbol-bearing | -24% | -20% | 5.5 → 5.0 | 5 / 6 | 17/11/72 | 0.345 | 4/1 | 0.375 |
| gpt-5.5 | symbol-free | -64% | -47% | 8.2 → 5.2 | 24 / 8 | 20/9/71 | 0.061 | 12/11 | 1.000 |

## Significance beyond correctness

Correctness is not the only per-query comparison. Sign tests on the other outcomes
(treeloom wins/losses vs grep, ties excluded):

**Cost per query** (win = treeloom cheaper on that query):

| Model | Query set | Cheaper / dearer | p |
|---|---|---|---|
| deepseek-flash | symbol-bearing | 60/40 | 0.057 |
| deepseek-flash | symbol-free | 75/25 | 5.6e-07 |
| claude-sonnet-5 | symbol-bearing | 68/32 | 0.00041 |
| claude-sonnet-5 | symbol-free | 73/27 | 4.7e-06 |
| gpt-5.5 | symbol-bearing | 61/39 | 0.035 |
| gpt-5.5 | symbol-free | 90/10 | 3.1e-17 |

**Turn-cap exhaustion on symbol-free queries** (win = grep capped and treeloom did not):
deepseek-flash 8/1 (p = 0.039), claude-sonnet-5 8/0 (p = 0.0078), gpt-5.5 17/1 (p = 0.00015).
On symbol-bearing queries caps were rare for both arms and no difference is detectable.

**Ranking, treeloom vs grep.** Recall@1 and MRR are statistical ties in five cells. In the
sixth — deepseek-flash, symbol-bearing — **grep ranks the right file first significantly
more often** (recall@1 6/18, p = 0.023; MRR 8/23, p = 0.011): with the identifier in the
question, string matching is hard to beat. This is the behaviour the project's positioning
already states: semantic search, not symbol lookup.

**Ranking, treeloom vs claude-context** (deepseek-flash, same embedding model): treeloom
ranks the right file first significantly more often on both populations — recall@1 29/9
(p = 0.0017) symbol-bearing and 24/8 (p = 0.007) symbol-free; MRR 39/17 (p = 0.0046) and
47/28 (p = 0.037). Correctness is tied.

## Per-cell detail

Dollar figures come from the verified price table in `domain/benchmark/cost_model.py`
(`PRICE_CAPTURE_DATE` 2026-09-20). **DeepSeek is priced at its peak rate**; these
cells ran off-peak, so DeepSeek's billed cost was roughly half the figure shown.

### deepseek-flash (Cell A)

Symbol-bearing:

| Arm | Correctness | Recall@1 | Recall@5 | MRR | Turns | Tokens/query | $/query | Turn-cap hits |
|---|---|---|---|---|---|---|---|---|
| grep | 4.52 | 0.80 | 0.92 | 0.857 | 4.7 | 20,699 | $0.0031 | 0 |
| treeloom | 4.49 | 0.68 | 0.88 | 0.773 | 3.6 | 15,094 | $0.0028 | 0 |
| claude-context | 4.49 | 0.48 | 0.84 | 0.637 | 4.2 | 19,274 | $0.0032 | 0 |

Symbol-free:

| Arm | Correctness | Recall@1 | Recall@5 | MRR | Turns | Tokens/query | $/query | Turn-cap hits |
|---|---|---|---|---|---|---|---|---|
| grep | 4.40 | 0.49 | 0.81 | 0.619 | 6.7 | 51,417 | $0.0063 | 11 |
| treeloom | 4.56 | 0.44 | 0.70 | 0.575 | 4.2 | 28,623 | $0.0044 | 4 |
| claude-context | 4.40 | 0.28 | 0.60 | 0.437 | 4.6 | 32,244 | $0.0046 | 4 |

claude-context vs grep, recall@5: 2/10 (p = 0.039) symbol-bearing;
7/28 (p = 0.0005) symbol-free —
**significantly worse than grep on both populations**, and not better on correctness.

### claude-sonnet-5 (Cell B)

Symbol-bearing:

| Arm | Correctness | Recall@1 | Recall@5 | MRR | Turns | Tokens/query | $/query | Turn-cap hits |
|---|---|---|---|---|---|---|---|---|
| grep | 4.41 | 0.76 | 0.89 | 0.823 | 4.8 | 25,479 | $0.0397 | 2 |
| treeloom | 4.47 | 0.69 | 0.88 | 0.785 | 3.3 | 18,527 | $0.0340 | 1 |

Symbol-free:

| Arm | Correctness | Recall@1 | Recall@5 | MRR | Turns | Tokens/query | $/query | Turn-cap hits |
|---|---|---|---|---|---|---|---|---|
| grep | 4.14 | 0.51 | 0.76 | 0.618 | 6.2 | 41,574 | $0.0493 | 11 |
| treeloom | 4.43 | 0.41 | 0.74 | 0.563 | 3.3 | 23,795 | $0.0368 | 3 |

### gpt-5.5 (Cell B)

Served as `gpt-5.5-2026-04-23`. Symbol-bearing:

| Arm | Correctness | Recall@1 | Recall@5 | MRR | Turns | Tokens/query | $/query | Turn-cap hits |
|---|---|---|---|---|---|---|---|---|
| grep | 4.30 | 0.74 | 0.90 | 0.812 | 5.5 | 27,828 | $0.0910 | 5 |
| treeloom | 4.38 | 0.74 | 0.93 | 0.828 | 5.0 | 21,080 | $0.0724 | 6 |

Symbol-free:

| Arm | Correctness | Recall@1 | Recall@5 | MRR | Turns | Tokens/query | $/query | Turn-cap hits |
|---|---|---|---|---|---|---|---|---|
| grep | 4.05 | 0.53 | 0.75 | 0.624 | 8.2 | 57,129 | $0.1353 | 24 |
| treeloom | 4.41 | 0.50 | 0.76 | 0.628 | 5.2 | 20,805 | $0.0721 | 8 |

## What the results show

1. **Cost falls in every cell, and the saving is largest on the most expensive model.**
   On symbol-bearing queries it rises with model price (−11% → −14% → −20%). On
   symbol-free queries it is −30% (deepseek-flash), −25% (sonnet-5) and −47% (gpt-5.5):
   largest where grep's extra exploratory turns are priciest, but not strictly
   monotonic, since sonnet-5 comes in below deepseek-flash.
2. **Token savings overstate cost savings.** Symbol-bearing queries show −24 to −27%
   tokens but −11 to −20% cost. Much of grep's extra context is re-sent history that
   prompt caching bills at a discount.
3. **Grep's failure mode on symbol-free queries is exhaustion, not wrong answers.**
   It hit the 12-turn cap on 11, 11 and 24 of 100 queries (vs treeloom's 4, 3, 8), yet
   its correctness stayed close. Capable agents keep searching until they get there;
   they just pay for it.
4. **On symbol-bearing queries grep's recall is competitive or better.** With the
   identifier in the query, grep can string-match to the file; deepseek-flash's grep
   arm led recall@1 by 0.12. This is the expected behaviour for this population.

## Relationship to earlier findings

- **Cost/efficiency replicates.** It matches the new-epoch deepseek symbol-free cell
  (2026-07-05: −41% tokens, −33% cost) closely, and now holds on sonnet-5 and gpt-5.5.
- **The strong correctness results do not replicate here — and the difference looks
  agent-capability-dependent.** [benchmark-eval.md](benchmark-eval.md) records a
  symbol-free correctness win at p = 2.3e-10 on featbit with **gpt-4o** as the agent (a
  weaker model, which also judged its own answers, on an unpinned checkout), and
  [benchmark-findings.md](benchmark-findings.md) a pooled p < 0.0001 across four repos
  with **deepseek-chat** on the pre-rebuild harness, including PowerToys where grep
  collapsed outright. When grep has nothing to match, a weaker agent gives up or guesses;
  today's capable agents keep searching until they find it. So with current models the
  correctness gap shrinks below significance at n = 100 while the cost gap remains.
  Positioning should lead with cost, not answer quality.
- **Six cells are not six independent tests.** Every model answered the same 200
  queries, so the cells cannot be pooled into a single stronger sign test.

## Threats to validity

- **Judge ceiling.** Correctness means sit at 4.05–4.56 on a 1–5 scale, with 71–81 ties
  per 100 queries. A compressed scale makes small real differences hard to detect.
- **Recall is read-recall.** It counts files the agent *opened*, so an arm that takes
  more turns opens more files and scores higher mechanically. Grep's recall edge
  partly reflects effort spent rather than ranking quality.
- **Ground-truth ambiguity from C# partial classes.** 11 of the 200 queries target a
  class split across `X.cs` and `X.Log.cs`. They carry `relevant_file_alternatives`
  so finding either half counts; genuinely specific partials were left untagged.
- **One repository.** featbit is C#-heavy (back-end) with a TypeScript front-end. The
  symbol-free set is 63% C# / 34% TypeScript / 3% Python.
- **Single run per cell.** No repeated-run variance estimate.

## Setup

| Component | Value |
|---|---|
| Embeddings | `jinaai/jina-embeddings-v2-base-code` (TEI, fp16) |
| Reranker | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` |
| HyDE + per-chunk summaries | `Qwen2.5-Coder-3B-Instruct` Q4_K_M (llama.cpp) |
| treeloom index | 6,424 chunks, 1,943 files, 0 indexing errors |
| claude-context | `@zilliz/claude-context-mcp@0.1.15`, same embedding model, 9,504 chunks over 1,541 files |
| Agent protocol | native tool calls (all three models) |
| Harness defaults | `--max-turns 12`, `--obs-char-limit 4000`, scoped treeloom search — matching the pinned anchor cells |

**Models excluded.** `claude-opus-5` (budget). `gpt-5.6-sol`: it rejects function tools
on `/v1/chat/completions` unless `reasoning_effort=none`, i.e. it could only be run as a
non-reasoning model at frontier prices. `repomap`: excluded on relevance, per the runbook.

## Spend

| Model | Symbol-bearing (both arms) | Symbol-free (both arms) | Total |
|---|---|---|---|
| deepseek-flash (3 arms) | $0.91 | $1.53 | $2.44 |
| claude-sonnet-5 | $7.37 | $8.61 | $15.98 |
| gpt-5.5 | $16.34 | $20.74 | $37.08 |
| **All cells** | | | **$55.50** |

Figures are peak-priced for DeepSeek (billed ≈ half). They exclude the `gpt-4o-2024-08-06`
judge and gold generation.

## Reproduce

The committed query sets and gold answers are the reference; use them rather than
regenerating. Ground-truth paths are rooted at the placeholder
`/home/user/source/featbit`, and agentic scoring compares repo-relative paths, so any
checkout whose directory is named `featbit` scores correctly. The sets were generated
with an earlier ordering in `sample_entities`; the released version returns seeded
random order, so regenerating would select the same entities under different query ids.

```bash
git -C <featbit> checkout v5.4.9        # c16207823d7b26e25bcbbe2a2eb032b602eea519

# Index, then confirm the index matches the pinned tree before spending:
curl -X POST http://localhost:8001/index-repo -H 'Content-Type: application/json' \
  -d '{"path": "<featbit>", "force": true}'
curl http://localhost:8001/sources/<source_id>/staleness      # expect is_stale: false

# Pre-flight gate (3 queries x each model):
SMOKE_REPO=<featbit> SMOKE_QUERIES=benchmarks/queries/featbit-v549-clean-100.jsonl \
  scripts/agentic_smoke.sh deepseek-flash claude-sonnet-5 gpt-5.5

# One cell per (model, query set):
python -m treeloom.benchmark agentic \
  --queries benchmarks/queries/featbit-v549-clean-100.jsonl \   # or featbit-v549-symfree-100.jsonl
  --repo <featbit> --search-url http://localhost:8001 \
  --arms grep,treeloom \                                          # Cell A adds claude-context
  --model <model> --judge-model gpt-4o-2024-08-06
```

**claude-context needs isolated configuration.** It reads its embedding endpoint from
`OPENAI_BASE_URL`, which the harness's own OpenAI route (the judge) also reads, and
the adapter passes the full environment through. Setting it globally sends every
judge call to the embedding server. Scope it to the subprocess instead:

```bash
export CLAUDE_CONTEXT_MCP_CMD='env OPENAI_BASE_URL=<tei-host>/v1 OPENAI_API_KEY=unused \
  EMBEDDING_PROVIDER=OpenAI EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code \
  MILVUS_ADDRESS=<milvus-uri> npx -y @zilliz/claude-context-mcp@0.1.15'
```

Pin the version: the adapter's default is `@latest`.

**Pre-build claude-context's index and verify it by row count.** Version 0.1.15 reports
"fully indexed and ready for search" while its background indexer is still running
(observed at 727 of 1,541 files), and it serves searches against the partial index.
The harness returns on that message and starts querying, so early queries would search
an incomplete index. Build it first, keeping the server alive until the collection's
distinct-file count stops growing. For this run: 9,504 chunks, 1,541 files.

## Harness safeguards

Several failure modes would corrupt a benchmark like this **silently** — producing
plausible numbers with nothing in the output to reveal the problem. The harness
guards against each.

| Failure mode | Safeguard |
|---|---|
| A token budget truncates generated queries mid-sentence, leaving unanswerable rows | a 400-token budget plus a check that drops unfinished queries (it caught 89 of 486 in this run) |
| `clean`'s name-based answerability rules reject every symbol-free query, since those queries omit the name by design | `clean --symbol-free`; `clean` also refuses to write an empty set |
| An interrupted generation run loses every paid call made so far | generation checkpoints to `<output>.partial` and resumes on re-run |
| A partial generation run covers only part of the repo, because entities are processed in path order | `sample_entities` returns seeded random order, so any prefix is a uniform sample |
| A C# partial class split across files scores zero when the agent finds the other half | `relevant_file_alternatives` equivalence groups |
| Two runs on the same repo starting in the same second write into one results file | results files are claimed atomically (`O_EXCL`), with `-N` on collision |
| A planned model id does not exist, or a price is inherited rather than checked | every cost-model entry is vendor-verified; non-existent ids stay unpriced |
