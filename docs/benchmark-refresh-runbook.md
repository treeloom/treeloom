# Benchmark refresh runbook

> **Run 2026-09-21.** Results: [benchmark-results-2026-09.md](benchmark-results-2026-09.md).

For re-running the agentic comparison (LLM agent with grep/glob/read tools vs.
the same agent with treeloom `search_code`) on a current model generation, at
known cost, against a pinned corpus. Written for the 2026-09 refresh; the shape
generalizes, and anyone re-running the benchmark on their own budget can use it
as the template. An **arm** is one tool configuration, a **cell** one run of
every arm over one query set with one agent model.

The governing constraint is **cost**, and the dominant cost is not the models.

## 1. Where the money goes

Real per-query token profiles from the last full multi-arm run (a 2026-06-11
featbit run whose output is local, not in the repository), priced per **100
queries**, uncached. The token profiles are from June; the prices are each
tier's 2026-09 list prices, including models (gpt-5.6, opus-5) that did not
exist in June — this is a cost projection, not a measurement of those models:

| tier | grep | treeloom | claude-context | repomap | 4-arm | grep+treeloom |
|---|---|---|---|---|---|---|
| deepseek flash | $0.63 | $0.60 | $0.78 | $2.93 | **$4.95** | $1.24 |
| sonnet-5 | $7.29 | $6.92 | $8.92 | $32.87 | **$56** | $14.21 |
| gpt-5.6 | $12.58 | $11.95 | $15.34 | $55.27 | **$95** | $24.52 |
| opus-5 | $12.14 | $11.54 | $14.87 | $54.78 | **$93** | $23.68 |

**repomap is 62% of all prompt tokens in a 4-arm cell.** It re-sends an 8k
map every turn, so it scales with turns and it caps out often (42/100 in the
2026-06 run). It also lost decisively — judge 3.35 vs treeloom's 4.26. In that
run repomap averaged 105.7k tokens per query against grep's 20.8k (5.1×, the
figure the README cites) with recall@5 0.67 (local run output, not in the
repository).

repomap is **excluded from the standard suite** (see §2). Its column stays
in this table because the number is the reason.

The naive matrix (4 arms × 4 models) is ~$250. The design below answers the
same questions for **~$64**.

## 2. Recommended cells

**Cell A — the headline table.** `grep`, `treeloom`, `claude-context`,
cheapest tier, n=100. **~$2.** Arm-vs-arm comparison does not need a frontier
model; it needs identical conditions across arms, which is what a single
cheap cell gives.

**Cell B — "it holds across tiers".** `grep` + `treeloom` only, on each
frontier model, n=100. ~$62 for three models. This is the claim that
genuinely needs expensive models, and it needs exactly two arms.

**repomap is excluded from the suite** — on relevance, not on benchmark
cost. Be precise about that: dropping it entirely saves **$2.93**, because
the $143 it would have cost at frontier tiers was already avoided by not
running it there. The reason to exclude it is that it answers a question this
benchmark is not asking.

It is ambient context rather than search: a fixed-budget symbol map re-sent
in the system prompt every turn. Its cost therefore scales with turns while
its coverage of the repo shrinks as the repo grows — so the approach is at
its best on repos small enough for the map to cover most of them, and at its
worst on exactly the professional-scale codebases treeloom targets. That is a
property of the design, not a defect, and re-measuring it each cycle
re-confirms a settled result at this scale.

The arm stays in the harness (`--arms repomap`). It is the right tool to
reach for if the question ever becomes "how does this behave on a small
repo" — the regime it is built for, and one we have not measured.

**Add the symbol-free cell.** Symbol-bearing queries understate treeloom —
grep just string-matches the identifier — so a symbol-free cell is standard
(see `docs/benchmark-findings.md`). Budget it as a second cell at the same
cost per model.

Note this is a **query set**, not an `agentic` flag: `--symbol-free` and
`--llm-symbol-free` belong to the `queries` subcommand, which generates the
set that `agentic --queries` then consumes. The shipped symbol-free set is
`featbit-v549-symfree-100.jsonl`, pinned at `v5.4.9`; a re-pinned corpus needs
it regenerated, since ground-truth paths follow the tree:

```bash
python -m treeloom.benchmark queries --source-dir <pinned corpus> \
  --llm-symbol-free --count 120 --sample-seed 0 \
  --output benchmarks/queries/<corpus>-symfree.jsonl
python -m treeloom.benchmark clean --symbol-free \
  --in  benchmarks/queries/<corpus>-symfree.jsonl \
  --out benchmarks/queries/<corpus>-symfree-clean.jsonl \
  --repo <pinned corpus> --limit 100
```

`--symbol-free` is required. `clean`'s answerability rules demand that a query
*mention* its entity name, while a symbol-free query is generated to *omit* it —
without the flag every row is rejected. `clean` used to write an empty file and
exit 0 in that case; it now refuses and names the flag. Generation is also
checkpointed to `<output>.partial`, so re-running an interrupted `queries`
command resumes instead of re-paying for finished entities.

`--sample-seed` matters on large repos (without it the sample is not
reproducible), and `clean` drops unanswerable and duplicate rows — an
uncleaned set previously inverted a reranker verdict.

## 3. Before spending anything

Cheap gates, in order. Each one has caught a wasted run before.

```bash
# 1. Confirm the served model is the model you asked for. .env and
#    nvidia-smi both lie; read the vendor's reported id.
scripts/agentic_smoke.sh          # 3 queries/model: termination, grounding,
                                  # mean correctness >= 3.0, served_models present
```

2. **Sufficiency screen** — one LLM call per query instead of ~6 turns
   (~15 min/config vs ~1h). `need_more_rate` is the trustworthy leading
   indicator; `mean_correctness_answered` is CONFOUNDED by punt rate, because
   a config that punts hard queries inflates its own mean over the ones it
   answered.

   It has **no CLI subcommand** — `run_screen` is an async Python API:

   ```python
   import asyncio
   from treeloom.application.benchmark.screen import run_screen
   from treeloom.domain.benchmark.queries import load_queries

   qs = load_queries("benchmarks/queries/<set>.jsonl")
   print(asyncio.run(run_screen(
       qs, "<repo-name>",
       search_url="http://localhost:8001",
       path_prefix="<pinned corpus>",
       configs=[{"name": "full", "body": {}}],
       model="<pinned model id>",
   )))
   ```

   Keep `concurrency` at its default 8 — 40 concurrent `/search` calls
   ReadTimeout the HyDE and reranker backends.

3. **Confirm prices.** Every model you plan to run must have a
   `verified: True` entry in `src/treeloom/domain/benchmark/cost_model.py`
   with a current `PRICE_CAPTURE_DATE`. Entries added ahead of a run start
   `verified: False` — the price inherited from a predecessor tier and the
   model ID expected rather than observed. Confirm both against the vendor's
   page, set `verified: True`, bump `PRICE_CAPTURE_DATE`. An unpriced model
   reports `cost_unpriced: <model>` in the summary rather than silently
   dropping the cost fields, so check for that key before trusting a cost
   number. (The 2026-09 entries were verified on 2026-09-20.)

4. **Check provider account limits** — spend caps, quotas, and any prepaid
   balance — before a frontier cell. An earlier Opus run was cut short 33
   queries from the end by a provider-side HTTP 400; such rejections name
   their cause only in the response body, so they are easy to misread as a
   request error.

## 4. Pin the corpus — and re-index it

The single biggest methodology fix available. Previous runs used a working
checkout that moved between epochs, so file counts drifted (four different
counts are recorded across the docs) and different cells silently measured
different trees.

```bash
git -C <corpus> checkout <tag-or-sha>     # a PUBLIC tag, so results are checkable
git -C <corpus> rev-parse HEAD            # record this in the run notes
```

**Re-index after pinning.** The `treeloom` arm reads the index; the `grep`
arm walks the on-disk tree. Pin one without re-indexing the other and the
arms are searching different codebases — which is a real failure that has
already happened once here.

```bash
curl -X POST http://localhost:8001/index-repo \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $TREELOOM_KEY" \
  -d '{"path": "<corpus>", "force": true}'
```

Then confirm the index matches the pin before running:

```sql
SELECT id, path, commit_sha, indexed_at FROM source_records WHERE path LIKE '%<corpus>%';
```

`commit_sha` must equal the `rev-parse` above. Nothing in the benchmark
artifacts records the corpus revision, so this check is the only thing
standing between a pinned run and an unpinned one.

## 5. Running

```bash
python -m treeloom.benchmark agentic \
  --queries benchmarks/queries/<set>.jsonl \
  --repo <pinned corpus> --search-url http://localhost:8001 \
  --arms grep,treeloom[,claude-context,repomap] \
  --model <pinned model id> --max-turns 12 \
  --obs-char-limit 16000
```

- `--obs-char-limit 16000`: the default 4000 truncates observations and hides
  the very payload deltas a comparison is measuring. Use 16000 when the
  question is about payload size; keep the default when the goal is
  comparability with earlier cells run at 4000 (the 2026-09 refresh kept 4000
  for that reason).
- **Versioned DeepSeek IDs need no override.** `deepseek-v4.1-flash` names a
  specific version and passes the floating-alias guard; only the bare
  `deepseek-chat` / `deepseek-reasoner` aliases require
  `--allow-floating-model`.
- The `claude-context` arm needs `MILVUS_ADDRESS` plus embedding env or it
  exits at startup; the `repomap` arm builds aider in an isolated env
  (`uv run --with aider-chat` — never install it into this venv).
- Gold answers cache to `benchmarks/gold/<repo>.jsonl`; regenerate with
  `--regen-gold` when the corpus pin changes, or the judge scores against
  answers from a different tree.

## 6. Reporting

**Never mix epochs.** The harness was rebuilt 2026-07-05 (native tool_calls
replacing the ReAct loop); numbers from before that are not comparable with
numbers after. A refresh starts a third epoch — say so in the results, and do
not merge rows across them into one table.

State the corpus pin, the exact model IDs, and whether the judge shared a
model with the agent (it did in the 2026-06 runs, which is a self-judging
bias worth disclosing rather than leaving for a reader to infer).
