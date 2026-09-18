# Benchmark eval: grep vs treeloom (with significance)

This document is for anyone who wants to measure, on their own repositories, whether
treeloom search helps an LLM coding agent compared with a plain grep/glob/read agent —
in answer quality and in token cost, with statistical significance — before deciding to
adopt it. It covers the four-step sequence (`queries` → `clean` → `agentic` → `rollup`),
query hygiene pitfalls, how to read p-values, and an end-to-end example.

Terms used throughout:

- **arm** — one tool configuration the agent runs with. `grep` gives the agent
  grep/glob/read_file tools; `treeloom` gives it treeloom's `search_code`.
- **cell** — one benchmark run: a query set × a repo × an agent model, every arm.
- **symbol-bearing / symbol-free query** — whether the question text contains the
  identifier it is about (`How does AuthTokenService refresh tokens?`) or describes
  behaviour only (`Where is the logic that rotates bearer credentials on expiry?`).
- **recall@5** — fraction of queries where a ground-truth file appears in the top 5
  results the agent actually used; **correctness** — an LLM judge's 1–5 score of the
  final answer against a cached gold answer.

---

## When to use this

Use this workflow when you want a reproducible, numbers-first answer to "does treeloom
actually help vs a plain grep agent on my codebase?" The output is a markdown comparison
table (mean tokens, recall@5, correctness win-rates, and sign-test p-values) from a real
LLM agent loop — not synthetic retrieval metrics.

---

## Prerequisites

1. **Repo indexed into treeloom.** Index with:

   ```bash
   curl -X POST http://localhost:8001/index-repo \
     -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $TREELOOM_KEY" \
     -d '{"path": "/path/to/your/repo"}'
   ```

   Or for a remote URL: `-d '{"url": "https://github.com/org/repo"}'`. Monitor with
   `curl http://localhost:8001/status`.

2. **Indexer reachable** at `--search-url` (default `http://localhost:8001`). Run the
   stack with `./run.sh` or start the indexer alone with:

   ```bash
   uvicorn treeloom.indexer_service:app --host 127.0.0.1 --port 8001
   ```

3. **Capable tool-calling LLM.** The agentic harness drives a real LLM agent through
   multiple tool-call turns per query. The agent protocol defaults **per vendor**
   (since 2026-07-05): hosted openai/deepseek/anthropic routes use native OpenAI-style
   `tool_calls`; local models fall back to a ReAct text protocol (the agent writes
   "Thought / Action / Observation" text that the harness parses, instead of structured
   tool calls; a small local model that can't follow it will produce meaningless results). Force either with
   `--native-tools` / `--no-native-tools`; the effective protocol is recorded as
   `agent_protocol` in `_summary.json`. Configure the endpoint via:

   - `BENCHMARK_QUALITY=STANDARD` — selects the hosted `STANDARD` tier of
     `resolve_llm_config` (`src/treeloom/adapters/benchmark/llm_client.py`); the
     `BASIC` tier defaults to a local Ollama `qwen2.5-coder:7b`, too weak for this — or
   - `NOMCP_LLM_URL` + `NOMCP_LLM_MODEL` — point at any OpenAI-compatible endpoint.

4. **Repo on disk.** The grep arm needs the repo present at the path you pass to
   `--repo`. The treeloom arm calls the indexer `/search` endpoint. Both must be
   available during the run.

5. **Pinned (non-floating) model IDs.** Agentic runs refuse floating model aliases
   (`deepseek-chat`, `deepseek-reasoner`, bare `gpt-4o`/`gpt-4o-mini`, `*-latest`) for
   both the agent and the judge — vendors silently remap them, which corrupts
   cross-run comparisons. Override with `--allow-floating-model` or
   `TREELOOM_ALLOW_FLOATING_MODELS=1`; a permitted run stamps
   `floating_model_warning: true` into `_summary.json`. DeepSeek publishes no dated
   IDs, so DeepSeek runs always need the override. Every summary also records the
   vendor-reported `served_models`, so alias drift is visible after the fact.

6. **Run the smoke gate before paid cells.** `scripts/agentic_smoke.sh` (asserted by
   `scripts/agentic_smoke_check.py`) runs a 3-query probe per model and checks
   termination, answer grounding, mean judge correctness >= 3.0, and that
   `served_models` / `agent_protocol` were recorded. A failure means alias drift or
   protocol rot — do not start a paid full cell until it passes.

---

## The documented command sequence

### Step 1 — Generate queries

Generate entity-anchored queries from the indexed graph:

```bash
python -m treeloom.benchmark queries \
  --strategy all \
  --source-dir /path/to/your/repo \
  [--source-id <source_id>] \
  [--output benchmarks/queries/myrepo-enhanced.jsonl]
```

`--strategy all` runs all seven strategies (`base`, `docstring`, `namespaced`,
`entity_context`, `intent`, `cross_cutting`, `problem_driven`). You can pass a comma-separated subset.
`--source-dir` doubles as a path-prefix filter for graph entities — it must match how
paths were stored at index time. `--source-id` scopes to one exact indexed source
(combine both to be precise).

**Grep-resistant variants.** By default, generated queries embed the target identifier
(e.g. "How does `AuthTokenService` refresh tokens?"). A grep agent can cheat by
grepping for `AuthTokenService` and finding the right file trivially — the comparison
tells you nothing interesting. Two flags harden the set:

- `--symbol-free` — keeps only queries that do not contain the target identifier
  verbatim (stresses semantic retrieval). Fast; deterministic.
- `--llm-symbol-free` — rewrites each query into a behavioral description via an LLM
  call so no identifier leaks (e.g. "Where is the logic that rotates bearer credentials
  on expiry?"). Requires `BENCHMARK_QUALITY=STANDARD` or `--model`.

**The symbol-free cell is the one that matters — run it as standard.** A large n=100
multi-repo study (see [benchmark-findings.md](benchmark-findings.md)) showed the verdict
*flips* on query style: on symbol-**bearing** queries grep wins or ties (it just string-matches the
identifier); on symbol-**free** queries treeloom significantly beats a grep/read agent on
answer quality (pooled correctness p<0.0001) at lower token cost. **treeloom is for
semantic/behavioral questions, not identifier lookup** — so a symbol-bearing-only
benchmark systematically understates it. Always include a symbol-free cell; prefer
`--llm-symbol-free` (behavioral, fully grep-resistant) for the headline number, with
`--symbol-free` as the fast deterministic floor. Report both styles side by side.

> **C#/Java/undocumented sources: use `--llm-symbol-free`, not `--symbol-free`.**
> `--symbol-free` keeps only queries whose text shares no identifier with the
> target — in practice the docstring/behavioral variants. Languages and codebases
> with few/no docstrings (most C#, Java, decorator-heavy or undocumented code)
> therefore yield **~0** symbol-free queries, and the harness used to fail
> silently with an empty set. The harness now warns and recommends
> `--llm-symbol-free` as the headline when `--symbol-free` keeps fewer than
> `--symbol-free-min-yield` (default 5) queries. `--llm-symbol-free` paraphrases
> each entity into a behavioral query with an LLM, so it does not depend on
> source docstrings. See `docs/benchmark-findings.md` for the full verdict.
>
> *Future follow-up:* an `--auto-symbol-free` flag could automatically fall
> back to the LLM path when the deterministic `--symbol-free` filter yields
> below `--symbol-free-min-yield`. Not implemented — today the harness only
> warns and recommends the switch.

```bash
# Headline: LLM-paraphrased behavioral queries (works on docstring-free C#/Java).
python -m treeloom.benchmark queries \
  --strategy all \
  --source-dir /path/to/your/repo \
  --llm-symbol-free \
  --output benchmarks/queries/myrepo-enhanced.jsonl

# Fast deterministic floor (only useful on docstring-rich sources, e.g. Python):
python -m treeloom.benchmark queries \
  --strategy all \
  --source-dir /path/to/your/repo \
  --symbol-free \
  --output benchmarks/queries/myrepo-symfree.jsonl
```

### Step 2 — Sanity-filter with `clean`

The raw enhanced set can contain non-answerable rows (generic entity names that map to
many files) and duplicate query texts with different ground truths. Both corrupt
benchmark scores. Always clean before running the agentic harness:

```bash
python -m treeloom.benchmark clean \
  --in benchmarks/queries/myrepo-enhanced.jsonl \
  --out benchmarks/queries/myrepo-clean.jsonl \
  [--repo /path/to/your/repo] \
  [--limit 100] \
  [--per-strategy docstring=20,intent=20,namespaced=15,entity_context=15,cross_cutting=15,problem_driven=15]
```

What `clean` does:

- **Drops non-answerable rows.** A query is answerable when its entity name is
  mentioned in the query text AND the entity name is discriminating — it maps to 3 or
  fewer distinct files in the corpus — OR the query carries a path/namespace hint from
  a relevant file. Generic names like `constructor` or `Handle` that appear in 20 files
  are dropped.
- **Deduplicates by query text.** Rows whose stripped, lowercased query text was already
  seen are dropped (keeping the first occurrence). Duplicate texts with different ground
  truths would otherwise silently corrupt scoring.
- **`--repo` stop parts.** Path components from `--repo` (length > 3) are added to the
  answerability filter's stop list so repo-specific directory names (e.g. `featbit`,
  `upstream-project`) don't accidentally count as discriminating path hints.
- **`--per-strategy quota`** caps rows per strategy before dedup, preventing one
  strategy from dominating. Omit to keep all answerable rows.
- **`--limit N`** applies a final row cap after dedup.

### Step 3 — Run the agentic benchmark

Run grep arm vs treeloom arm in parallel, with LLM-judge scoring:

```bash
python -m treeloom.benchmark agentic \
  --queries benchmarks/queries/myrepo-clean.jsonl \
  --repo /path/to/your/repo \
  --search-url http://localhost:8001 \
  --arms grep,treeloom \
  --regen-gold \
  [--model <model-id>] \
  [--judge-model <stronger-model-id>] \
  [--max-turns 12] \
  [--limit N] \
  [--difficulty easy|medium|hard] \
  [--strategy <name>]
```

Key flags:

- `--regen-gold` — generates and caches gold reference answers on this run. Omit on
  subsequent runs to reuse cached gold (faster, consistent scoring).
- `--model` — the LLM for the measured agent arms (grep + treeloom). Prefer a
  non-reasoning chat model for fair token counts.
- `--judge-model` — the LLM for gold answer generation and per-query judging. Defaults
  to `--model`. Use a stronger model here for better scoring without inflating the
  measured arms' token counts.
- `--max-turns` — turn budget per query per arm (default 12).
- `--native-tools` / `--no-native-tools` — force the native `tool_calls` loop or the
  ReAct text protocol. Omit both to take the per-vendor default (see prerequisite 3);
  the effective protocol lands in `_summary.json` as `agent_protocol`.
- `--allow-floating-model` — permit floating model aliases (see prerequisite 5);
  the run stamps `floating_model_warning: true` into `_summary.json`.
- `--debug-transcripts` — persist the full agent transcript (including tool
  observations) into each result row, for post-hoc diagnosis of a bad run.
- `--obs-char-limit` — per-tool-observation char cap fed back to the agent (default
  4000, which models a context-capped client). **When measuring payload-size or
  serialization changes (e.g. markdown vs JSON, response_mode, payload trims), RAISE
  this (e.g. `--obs-char-limit 16000`)** — at the 4000 default a ~9–13k-char
  `search_code` payload is truncated to the cap for *both* arms, so the harness shows
  ~0 token difference and **hides** the very cut you're measuring. Keep
  4000 only for runs that intentionally model a capped client. The value is recorded
  in `_summary.json` so results are self-describing.
- `--lean-search` — strips neighbor and community-summary context from treeloom
  responses (chunks only, capped snippets). Cuts the dominant tool_observation token
  cost; useful for a pure retrieval token comparison.
- `--vector-only` — treeloom arm uses pure Milvus vector+rerank with no graph
  rescoring and no neighbor/community payload. Isolates whether the graph layer adds
  value above the vector layer.
- `--no-graph-scoring` — turns off graph ranking (rank by reranker only) while keeping
  the full payload. Decoupled from `--vector-only`; isolates ranking value from payload
  value.
- `--context-window N` — prunes the agent transcript to system + query + last N
  messages per turn. Attacks quadratic turn-cost growth at high turn counts.

The runner writes output to `benchmarks/results/agentic/`:

- `<UTC-timestamp>_<repo>.jsonl` — one row per query with per-arm metrics.
- `<UTC-timestamp>_<repo>_summary.json` — per-arm means + win/loss/tie counts +
  sign-test p-values.

At the end of the run, a markdown comparison table is printed to stdout (illustrative
values):

```
### Agentic benchmark: myrepo (n=50)

| Metric             | grep  | treeloom |     Δ / win-rate     | p-value |
|--------------------|---------:|---------:|:--------------------:|--------:|
| Mean total tokens  | 28,400 | 19,200 | +32.3% saved (ratio 0.677) |  —  |
| Mean recall@5      | 0.6400 | 0.8800 |    win-rate 0.7273   |  0.0031 |
| Mean correctness   | 3.4000 | 4.1000 |    win-rate 0.6818   |  0.0428 |
```

#### Benchmarking a cloud reranker

To benchmark the GPU-less cloud-rerank path (Cohere or Voyage), point `--search-url`
at an indexer instance started with `RERANKER_PROVIDER=cohere` (or `voyage`) set in
its environment. This is an indexer-side env switch at startup — there is no benchmark
CLI flag for it. See `docs/simple-mode.md` (its "How it's wired" table lists every
`RERANKER_PROVIDER` value) for provider configuration (`COHERE_API_KEY`, `VOYAGE_API_KEY`, `RERANKER_MODEL`).

### Step 4 — Multi-repo rollup

After running the agentic benchmark across multiple repos, aggregate into a single
pooled table:

```bash
python -m treeloom.benchmark rollup \
  benchmarks/results/agentic/<repoA>_*_summary.json \
  benchmarks/results/agentic/<repoB>_*_summary.json \
  benchmarks/results/agentic/<repoC>_*_summary.json \
  [--out benchmarks/results/agentic/rollup_combined.json]
```

`rollup` accepts any number of `_summary.json` paths as positional arguments. It
prints a markdown table (one row per repo plus a **Pooled** row) and writes
`rollup_<UTC-timestamp>.json` to `benchmarks/results/agentic/` (or `--out` path).

Pooled means are `n_queries`-weighted across repos. The pooled sign-test p-values sum
win/loss counts across all repos before running the binomial test — giving higher
statistical power than any single-repo run.

---

## Reading the output

### Single-repo comparison table

| Column | What it means |
|--------|---------------|
| Mean total tokens | Average tokens consumed by the agent (input + output) per query |
| tokens-saved % | `(grep_tokens - treeloom_tokens) / grep_tokens * 100` — positive = treeloom cheaper |
| token_ratio | `treeloom_tokens / grep_tokens` — below 1.0 = treeloom used fewer tokens |
| Mean recall@5 | Fraction of queries where a ground-truth file appeared in the top-5 search results used |
| Mean correctness | LLM-judge score (1–5) for factual accuracy of the agent's final answer against the cached gold answer |
| win-rate | Fraction of resolved (non-tie) queries where treeloom arm scored strictly higher |
| p-value | Two-sided binomial sign-test p-value (ties excluded) |

The `_summary.json` file also carries `quality_per_1k_tokens` — correctness divided by
tokens-per-query normalized to 1000 — for comparing arms that trade quality for
frugality.

### How to read p-values

The sign test asks: if there were truly no difference between the two arms, what is
the probability of seeing a per-query win/loss split this extreme or more extreme?

- **p < 0.05** — the split is statistically significant at the 5% level. Reject the
  null of no difference.
- **p >= 0.05 ("ns", not significant)** — the split is *consistent with* no difference
  at this sample size. This does **not** prove the arms are equal. Point estimates
  (mean scores, win rates) remain the best directional guess; "ns" means the evidence
  is insufficient at this n to confidently rule out chance.

The sign test excludes ties. At n=100 with ~45 ties excluded (~55 resolved queries),
the test has modest power for small true differences. A split of 28W/18L (p≈0.18)
could easily arise by chance even if treeloom is genuinely a few points better. Larger
n and/or larger effects give tighter conclusions.

### Multi-repo rollup table

| Column | What it means |
|--------|---------------|
| Repo / n | Repo name and query count |
| grep tok / treeloom tok | Per-arm mean total tokens |
| saved% | Token savings % for treeloom |
| r@5 win-rt / r@5 p | Recall@5 win-rate and its sign-test p-value |
| corr win-rt / corr p | Correctness win-rate and its sign-test p-value |
| **Pooled** row | n_queries-weighted means; pooled win/loss counts → pooled p-value |

### Baseline epochs

The 2026-07-05 harness rebuild changed the default agent protocol from ReAct text to
per-vendor native `tool_calls` and raised the agent completion budget from 512 to 2048
tokens. **Agentic numbers produced before 2026-07-05 (ReAct epoch) are not comparable
with numbers produced after (native-loop epoch)** — do not mix epochs in a rollup or
cite a pre-rebuild cell against a post-rebuild one. Check `agent_protocol` in a
`_summary.json` to see which epoch a result belongs to.

The reference cells named below are the maintainers' local run outputs under
`benchmarks/results/agentic/` — that directory is gitignored, so the files are **not in
the repository**; their numbers are reproduced in the docs cited. The new-epoch featbit
reference cells are `benchmarks/results/agentic/2026-07-05T205924_featbit_summary.json`
(clean-100) and `2026-07-05T215301_featbit-symfree_summary.json` (symbol-free) on the
deepseek cost tier (floating alias — `served_models: deepseek-v4-flash`), plus a
**pinned anchor pair** on `gpt-4o-2024-08-06` that no vendor alias remap can
invalidate: `2026-07-05T232746_featbit_summary.json` (clean-100: treeloom correctness
3.89 vs grep 3.41, sign test p=0.006, −44.5% tokens, treeloom read-recall@5 0.87 vs
0.70) and `2026-07-05T234145_featbit-symfree_summary.json` (symbol-free: treeloom 3.23
vs grep 2.13, p=2.3e-10, −87.7% tokens; grep hits the 12-turn cap on 25.8% of queries
vs treeloom's 0.8% — the strongest treeloom-vs-grep cell on record, consistent with
the known grep-collapses-on-symbol-free pattern).

**Current reference cells (2026-09-21).** featbit pinned at tag `v5.4.9`, an independent
judge (`gpt-4o-2024-08-06`, never the agent), agents deepseek-flash, claude-sonnet-5 and
gpt-5.5, on `featbit-v549-clean-100` and `featbit-v549-symfree-100`: summaries
`2026-09-21T{132010,135248,150023,154716,150024,154211}_featbit-v549*_summary.json`.
They supersede the deepseek-tier pair above as the reference for current models:
treeloom is 11–47% cheaper than grep in every cell with correctness statistically tied.
The gpt-4o anchor pair remains the record for a weaker agent, where the correctness gap
is significant — the edge is agent-capability-dependent. Results and caveats:
[benchmark-results-2026-09.md](benchmark-results-2026-09.md).

---

## Query-set hygiene (read before trusting any number)

These are hard-won lessons from building the canonical
`benchmarks/queries/featbit-clean-100.jsonl` set.

**Non-discriminating entity names.** Names like `constructor`, `Handle`, `Execute`, or
`Update` appear in dozens of files. A query about them cannot identify a single
ground-truth file — the query is unanswerable and any arm that "finds" a file gets
credit by luck. The `clean` filter drops any row where the entity name maps to more
than 3 distinct files in the corpus AND the query carries no path/namespace hint.

**Duplicate query texts.** Auto-generated sets can produce the same question text
(same lowercased/stripped form) anchored to different ground-truth files. These corrupt
scoring because only one can be "right" per text. `clean` deduplicates by text,
keeping the first occurrence.

**Why this matters more than it seems.** A prior artifact-heavy query set for the
featbit repo was approximately 26% unanswerable. Running a reranker A/B on that set
**inverted the verdict** relative to the clean set — the worse reranker appeared to
win because non-answerable queries are random noise that washes out real signal
unevenly. Never draw reranker or feature-flag conclusions from an unfiltered
auto-generated query set. Always `clean` first.

**Symbol-free queries and fair comparison.** If queries contain the target identifier,
grep can locate the right file in one tool call. The comparison then measures
"how fast can you grep" not "can you reason about code". Use `--symbol-free` (or
`--llm-symbol-free` for behavioral paraphrasing) to produce a set where grep must
search semantically, making the comparison informative.

**Mix difficulties and keep n reasonable.** The sign test has low power at small n
(especially with many ties). Aim for at least 50 resolved (non-tie) queries per arm
pair for meaningful p-values. Mix `easy`/`medium`/`hard` so the comparison isn't
dominated by trivially easy or impossibly hard queries.

**The canonical reference set** is `benchmarks/queries/featbit-clean-100.jsonl`
(100 rows, fully cleaned, with baseline r@1 0.66 / r@5 0.84 / MRR 0.730 on the
retrieval `run` subcommand with jina+bge @ pool 50). It predates the `v5.4.9` pin (it was
generated on a 2026-06 working checkout), but 99 of its 100 ground-truth files exist
unchanged at `v5.4.9`, so it runs against the pinned tag; the agentic reference sets are
`featbit-v549-clean-100.jsonl` / `featbit-v549-symfree-100.jsonl`. Other docs quote 0.63 / 0.80 / 0.70
for the same set and reranker: the spread comes from HyDE (an LLM call, nondeterministic
across restarts) and the community-detection state on the day, which is why A/B runs use
`--no-hyde --no-graph`. Use the set to sanity-check retrieval changes; use your own
cleaned set for agentic comparisons.

---

## End-to-end example

### Single repo

```bash
# 1. Index the repo (if not already done)
curl -X POST http://localhost:8001/index-repo \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TREELOOM_KEY" \
  -d '{"path": "~/repos/myapp"}'
# Poll until done:
curl http://localhost:8001/status

# 2. Generate symbol-free queries from the graph.
#    --llm-symbol-free is the headline (behavioral paraphrases; works even on
#    docstring-free C#/Java). Use --symbol-free only on docstring-rich sources.
python -m treeloom.benchmark queries \
  --strategy all \
  --source-dir ~/repos/myapp \
  --llm-symbol-free \
  --output benchmarks/queries/myapp-enhanced.jsonl

# 3. Clean: drop non-answerable + deduplicate
python -m treeloom.benchmark clean \
  --in benchmarks/queries/myapp-enhanced.jsonl \
  --out benchmarks/queries/myapp-clean.jsonl \
  --repo ~/repos/myapp \
  --limit 80

# 4. Run grep vs treeloom (generates and caches gold answers on first run)
python -m treeloom.benchmark agentic \
  --queries benchmarks/queries/myapp-clean.jsonl \
  --repo ~/repos/myapp \
  --search-url http://localhost:8001 \
  --arms grep,treeloom \
  --regen-gold \
  --max-turns 12
# Comparison table is printed at the end; results in benchmarks/results/agentic/
```

### Multi-repo rollup (including a C#/Java repo)

```bash
# Index each repo (shown once; skip if already indexed)
for REPO in ~/repos/python-service \
            ~/repos/java-api \
            ~/repos/csharp-backend; do
  curl -X POST http://localhost:8001/index-repo \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $TREELOOM_KEY" \
    -d "{\"path\": \"$REPO\"}"
done

# Generate + clean queries per repo.
# Use --llm-symbol-free here: the java-api / csharp-backend repos have few/no
# docstrings, so --symbol-free would yield ~0 queries for them.
for REPO in python-service java-api csharp-backend; do
  python -m treeloom.benchmark queries \
    --strategy all --source-dir ~/repos/$REPO \
    --llm-symbol-free --output benchmarks/queries/${REPO}-enhanced.jsonl

  python -m treeloom.benchmark clean \
    --in benchmarks/queries/${REPO}-enhanced.jsonl \
    --out benchmarks/queries/${REPO}-clean.jsonl \
    --repo ~/repos/$REPO --limit 50
done

# Run agentic benchmark per repo (omit --regen-gold on later runs to reuse cached gold)
for REPO in python-service java-api csharp-backend; do
  python -m treeloom.benchmark agentic \
    --queries benchmarks/queries/${REPO}-clean.jsonl \
    --repo ~/repos/$REPO \
    --search-url http://localhost:8001 \
    --arms grep,treeloom \
    --regen-gold --max-turns 12
done

# Aggregate into a pooled rollup table
python -m treeloom.benchmark rollup \
  benchmarks/results/agentic/*python-service*_summary.json \
  benchmarks/results/agentic/*java-api*_summary.json \
  benchmarks/results/agentic/*csharp-backend*_summary.json \
  --out benchmarks/results/agentic/rollup_combined.json
```

The rollup prints a table like (illustrative values):

```
| Repo             |  n | grep tok | treeloom tok | saved% | r@5 win-rt | r@5 p | corr win-rt | corr p |
|------------------|---:|---------:|-------------:|-------:|-----------:|------:|------------:|-------:|
| python-service   | 50 |   27,100 |       18,400 | +32.1% |     0.7143 | 0.009 |      0.6800 |  0.031 |
| java-api         | 50 |   31,500 |       21,200 | +32.7% |     0.6923 | 0.024 |      0.6538 |  0.068 |
| csharp-backend   | 50 |   29,800 |       19,900 | +33.2% |     0.7333 | 0.003 |      0.7143 |  0.009 |
| **Pooled**       |150 |   29,467 |       19,833 | +32.7% |     0.7130 | 0.0001|      0.6820 |  0.0022|
```

---

## See also

- `docs/simple-mode.md` — cloud reranker (Cohere/Voyage) configuration and benchmark results
- `docs/fleet-operations.md` — bulk indexing many repos
- `README.md` — running the stack and the benchmark CLI quick-reference
- `benchmarks/queries/featbit-clean-100.jsonl` — canonical reference query set
- `benchmarks/make_clean_queries.py` — script that generated the reference set
