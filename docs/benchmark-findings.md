# Agentic Benchmark Findings: treeloom MCP search vs grep/read

> **Current results:** [benchmark-results-2026-09.md](benchmark-results-2026-09.md)
> (featbit v5.4.9, deepseek-flash / claude-sonnet-5 / gpt-5.5). That run confirms the
> cost advantage on current models, but the strong correctness results below do **not**
> replicate with capable agents — see its "Relationship to earlier findings".

**How to read this document.** It is an append-only ledger of the studies that asked
"does treeloom help an LLM coding agent compared with grep?" Each section is dated and
later sections qualify or overturn earlier ones, so **the current bottom line is the last
section** and the first section is the oldest, most limited study. In order:

1. **Original study** (2026-06, `deepseek-chat`, n=20 per cell, two repos) — grep looked
   more token-efficient at tied quality. Superseded.
2. **Multi-repo replication at n=100** (2026-06-15) — the featbit quality lift did not
   generalize on symbol-bearing queries.
3. **D1: symbol-free queries** — the query style, not the repo, was the confound;
   treeloom won on quality and cost when the identifier is absent from the question.
4. **D2/D3: graph-layer ablations** — what the code graph contributes, and why graph
   rescoring stays on by default.
5. **2026-09 refresh** — current agents, pinned corpus, independent judge: the cost win
   holds (11–47% cheaper), the quality edge is a statistical tie.

Vocabulary: an **arm** is one tool configuration for the agent (`grep` = grep/glob/read
tools, `treeloom` = treeloom `search_code`); a **cell** is one run of every arm over one
query set on one repo with one agent model; **symbol-bearing / symbol-free** says whether
the question text contains the identifier it is about; **recall@k** is the fraction of
queries whose ground-truth file is in the top k the agent used; **correctness** is a 1–5
LLM-judge score against a cached gold answer. In tables, `g/t` and `tl` abbreviate
grep / treeloom, `r@5` recall@5, `corr` correctness, `tok` mean tokens per query, and
`k5` / `cw6` the `--search-top-k 5` / `--context-window 6` settings. **Community
summaries** are LLM-written summaries of clusters of related code found by Louvain
community detection over the code graph, **centrality** is a node's importance in that
graph, and **HyDE** expands a query with an LLM-written hypothetical answer before
embedding it. Every `p` is a
two-sided sign test over per-query wins and losses (ties excluded); "ns" means not
significant at that n, which is not evidence of equality. Result files named as
`benchmarks/results/...` are the maintainers' local run output — that directory is
gitignored and not in the repository. Likewise only the featbit query sets are shipped
(`benchmarks/queries/featbit-v549-*` for the 2026-09 refresh, and `featbit-clean-100`
as the retrieval reference set); the fastapi, godot, guava, PowerToys and three.js sets
behind the older sections were generated on unpinned checkouts and are not in the
repository — the Reproduce blocks show how to regenerate equivalents.

---

## Original study (2026-06): deepseek-chat, n=20 per cell

**Question.** For an LLM agent answering questions about a codebase, does Treeloom's
MCP `search_code` use fewer tokens and/or give better answers than a Claude-Code-style
`grep`/`glob`/`read` loop?

**Short answer.** With the agent scoped to one repo, **answer quality is roughly tied**,
but **grep/read is more token-efficient in most conditions** (treeloom ~1.4–2.7×). Treeloom's
retrieval **scales better with repo size** (it leads recall@10 on the large repo) and its
token premium is largely an engineering problem (weak recall@1 + bulky snippets). Treeloom's
clear, structural wins — repos not on local disk, cross-repo questions, very large codebases —
are not stressed by this benchmark.

All runs: model `deepseek-chat` (agent + judge), 20 queries/cell, `--search-top-k 5`,
`--context-window 6`, repo-scoped search unless noted. Repos: **fastapi**
(tiangolo/fastapi, mid-sized Python) and **godot** (godotengine/godot, large C++);
"unscoped" means the search ran over the maintainers' shared index of 98 repositories
instead of being filtered to the repo under test.

---

## Methodology

**Two arms, same agent.** A real ReAct agent answers each query twice; only the tools differ:
- **grep arm:** `grep` (ripgrep), `glob`, `read_file`, scoped to `--repo`.
- **treeloom arm:** one `search_code` tool → indexer `POST /search` (semantic + graph + rerank).

Same model, system-prompt skeleton, turn cap, and temperature (0.0). Tokens are the **real
API `usage`** summed across every turn (not estimated).

**Ground truth = entity-anchored.** Queries are reverse-generated from indexed Neo4j entities
(`Class`/`Function`/`Method`): each query targets a known entity, and that entity's defining
file is the gold `relevant_files`. Two flavors:
- **symbol-containing** (`cross_cutting`/`problem_driven` strategies): the query names the
  entity (e.g. *"errors when calling `Engine::set_physics_ticks_per_second`"*) — grep can
  exact-match the identifier.
- **symbol-free** (`queries --llm-symbol-free`): an LLM paraphrases a *behavioral* query that
  names neither the entity nor its word-parts (validated) — grep can't shortcut to the symbol.

**Quality, two ways:** retrieval `recall@k`/`MRR` (files the agent read / treeloom returned vs
ground truth) **and** an LLM judge scoring the final answer (correctness/completeness 1–5)
against a cached **gold answer** generated once from the ground-truth source.

**Token categorization.** Each turn's prompt is split (anchored to real `usage`, divided by
tiktoken) into `system` (prompt + tool schemas), `query`, `agent_output` (re-fed transcript),
`tool_observation` (search/read payloads), and `completion`.

**Knobs added during the investigation:** `--search-top-k` (fix payload size), `--lean-search`
(chunks-only payload), `--context-window N` (prune transcript), `--no-scope-treeloom`
(turns off repo-scoped search, which is **on by default**), `--judge-model` (separate eval model).

---

## Results

### The 2×2 (scoped, k5, cw6, deepseek-chat, n=20)

| cell | correctness g/t | recall@5 g/t | recall@10 g/t | tokens g/t | treeloom/grep |
|---|---|---|---|---|---|
| godot (large) · **symbol-containing** | 4.4 / **4.35** | 0.95 / 0.85 | 0.95 / **0.95** | 10.1k / 26.8k | **2.66×** |
| fastapi (mid) · **symbol-free** | **4.2** / 3.0 | **0.85** / 0.45 | **0.85** / 0.55 | 16.8k / 14.5k | **0.86×** |
| godot (large) · **symbol-free** | **3.5** / 3.05 | 0.50 / 0.40 | 0.50 / **0.65** | 28.5k / 39.4k | **1.38×** |

(*fastapi symbol-containing was only measured **unscoped**: grep 10.9k/3.55 vs treeloom 28k/3.25,
2.58× — same grep-dominant pattern.*)

### Key sub-findings

1. **Token cost is dominated by `tool_observation` (75–85% for treeloom).** The agent re-reads
   search payloads every turn; with the full transcript this grows ~quadratically in turns.

2. **Lean payloads backfired.** Dropping neighbors/community summaries and capping snippets
   (`--lean-search`) made treeloom *worse* (+53% tokens, lower quality): smaller per-call
   results starved the agent, which then searched more and ran to the turn cap. The bottleneck
   is **turns × transcript**, not per-call size.

3. **Context-window pruning helped** (`--context-window 6`): ~36% fewer treeloom tokens with
   ~flat quality (fastapi-hard 43.5k → 28k).

4. **Unscoped search was the biggest confound.** With no source filter, `search_code` hit the
   whole 98-repo index and returned wrong-repo matches (tensorflow/langchain/ruff for a FastAPI
   query). Scoping to `--repo` cut treeloom's tokens **−58%** and **tripled recall** on
   fastapi symbol-free (recall@5 0.15 → 0.45, 34k → 14.5k). **`search_code` should default to
   source-scoped.**

5. **Retrieval scales with repo size.** Grep's recall@5 fell 0.85 → 0.50 from fastapi → godot
   (symbol-free); treeloom's held and **led recall@10 on godot (0.65 vs 0.50)**.

6. **The token premium is a recall@1 problem.** On symbol-containing godot, quality and
   recall@10 are tied, but treeloom's **recall@1 is 0.45 vs grep's 0.90** — the right file is
   ranked lower, so the agent does more search/read rounds → 2.66× tokens.

---

## Verdict (three independent dimensions)

- **Answer quality: treeloom ≈ grep.** Tied on symbol-containing (both repos), modestly behind
  only on symbol-free. Treeloom rarely gives *worse* answers at equal scope.
- **Token cost: grep wins (treeloom 1.4–2.7× in 3 of 4 cells).** The one cheaper cell (fastapi
  symbol-free) was also lower quality.
- **Retrieval at scale: trends toward treeloom.** It loses recall on mid repos but ties
  (symbol) or leads (symbol-free) recall@10 on the large repo.

**Bottom line:** for an agentic code-answering loop with a capable model on repos that are
**both indexed and on local disk**, grep/read is the more token-efficient path at equal answer
quality — *today*. This benchmark is close to grep's best case (capable model, mid-sized
greppable repos, queries that often name the symbol).

**Where treeloom is expected to win (not stressed here):** repos **not on disk** (grep can't
run), genuinely **cross-repo** questions (grep can't scope), and **very large** codebases (the
godot recall trend extrapolated). Its token premium is mostly fixable engineering: improve
recall@1 (rank the exact-symbol/target file #1) and trim snippet bulk.

---

## Recommendations

1. **Default `search_code` to source-scoped** (caller's repo / `source_id`). Single highest-impact
   finding: −58% tokens and 3× recall vs unscoped on a multi-repo index.
2. **Investigate recall@1** — treeloom's reranker isn't ranking the target file first on
   symbol-containing queries (0.45 vs grep 0.90). Check the name-in-query graph signal
   (`GRAPH_DELTA`) and rerank quality.
3. Keep transcript pruning available for agent loops; lean payloads are not worth it.

## Threats to validity

- One model (`deepseek-chat`), 20 queries/cell, two repos — directional, not definitive.
- Queries are entity-anchored; symbol-free queries are LLM-paraphrased (some ambiguity).
- Judge/agent are non-deterministic even at temp 0; treat ±0.3 correctness / ±15% tokens as
  noise (grep, which is cw/topk-invariant, wobbled that much run-to-run).
- Both arms search/read on disk; treeloom additionally requires the index. The "not on disk"
  and "cross-repo" advantages of the MCP are unmeasured.

## Reproduce

```bash
export BENCHMARK_QUALITY=STANDARD DEEPSEEK_API_KEY=sk-...

# 1. symbol-containing (entity-anchored, deterministic, free):
python -m treeloom.benchmark queries --strategy cross_cutting,problem_driven \
  --source-dir /data/repos/<repo> --repo-name <repo>-hard --max-entities 150 --max-per-entity 2 \
  --output benchmarks/queries/<repo>-hard.jsonl

# 2. symbol-free (LLM-paraphrase, grep-resistant):
python -m treeloom.benchmark queries --llm-symbol-free \
  --source-dir /data/repos/<repo> --repo-name <repo>-symfree --max-entities 60

# 3. run a cell (scoping default; gold cached, --regen to rebuild):
python -m treeloom.benchmark gold    --queries benchmarks/queries/<set>.jsonl --regen
python -m treeloom.benchmark agentic --queries benchmarks/queries/<set>.jsonl \
  --repo /data/repos/<repo> --model deepseek-chat --search-top-k 5 --context-window 6 --limit 20
```
Each run writes `benchmarks/results/agentic/<ts>_<repo>[_k5][_cw6][_unscoped]_summary.json`
(the suffixes record the `--search-top-k 5` / `--context-window 6` / `--no-scope-treeloom`
settings) with per-arm means, `token_categories_pct`, and a `comparison` block (tokens-saved %, win-rates).

---

## Multi-repo replication at n=100 (2026-06-15) — does the featbit lift generalize?

**Why.** Every earlier number rested on **featbit** (the repo Treeloom was developed
against). The open question was whether the agent quality/cost lift reproduces on *other*
repos. We replicated the full featbit pipeline at **n=100 per repo** on three external
repos spanning the legacy Java/C# long tail, with **featbit re-run as a control**.

**Repos.** featbit (control, C#/TS), `microsoft_PowerToys` (C#), `google_guava` (Java),
`mrdoob_three.js` (JS). All searched repo-scoped; agent + judge = `deepseek-chat`
(`BENCHMARK_QUALITY=STANDARD`), `--max-turns 10`.

**Pipeline (faithful to featbit-clean-100).** Deterministic entity-anchored strategy
generation (`prompt_enhancer`: base, namespaced, entity_context, intent, cross_cutting,
problem_driven; `--max-per-entity 7`) over the indexed graph → `benchmark clean`
(answerable-filter + dedup + per-strategy quota) → 100 queries/repo. Generation scoped to
each repo's **real source subtree** (not vendored/bundled/test trees).

### Pipeline corrections made for a valid run (all matter)
- **`base` strategy restored.** 15% of the reference set's rows come from the `base`
  strategy, which came from an
  entity-anchored deterministic template generator that had been dropped from the code
  (it was never an LLM step — the queries are templates: "What does X do?" / "Explain the X
  class"). Re-added as `prompt_enhancer._gen_base`.
- **`--max-per-entity` truncation fixed.** The enhancer returns `variants[:cap]` in a fixed
  strategy order; at the default cap 3 it silently never emitted `problem_driven`/
  `cross_cutting`. Generate with cap ≥ 7.
- **Build/vendor ground truth excluded.** `DEFAULT_EXCLUDES` lacked `build`/`dist`/`.min.`;
  three.js queries had been anchored to `build/three.core.js` (a minified bundle), which is
  unanswerable and silently zeroed recall for *both* arms. Also scoped generation to
  real source subtrees (PowerToys' Monaco JS bundle, three.js editor libs, guava test
  trees otherwise dominate the first-N-by-path entity pool).
- **PowerToys C# graph rebuilt.** All 3,596 `.cs` files were Module-only (graphless — a
  since-fixed language-key mismatch in the graph extractor had left C# files with no
  entities). Re-extracted with the current extractor → 20,168 C#
  Class/Function/Method entities, so PowerToys is a *real* C# repo here.

### Results — grep vs treeloom (default reranker), n=100/repo

| Repo | grep | treeloom (default) | corr p (tl vs grep) |
|------|------|--------------------|---------------------|
| **featbit** (control) | r@5 0.83 · corr 3.71 · 19k tok | r@5 0.84 · **corr 4.09** · 19k | **0.0045** ✅ |
| PowerToys (C#) | r@5 0.80 · corr 3.74 · 21k | r@5 0.76 · corr 3.86 · 26k | 0.74 ns |
| guava (Java) | r@5 0.76 · corr 3.54 · 25k | r@5 0.68 · corr 3.73 · 32k | 0.63 ns |
| three.js (JS) | r@5 0.93 · corr 4.09 · 25k | r@5 0.75 · corr 4.05 · 33k | 0.63 ns |
| **Pooled (400)** | — | tokens −20%, r@5 grep ahead **p=0.003** | corr p=0.079 (driven by featbit) |

### Results — adding the dominant lever (Cohere `rerank-v4.0-pro`)

Re-ran the treeloom arm against a search-only indexer with `RERANKER_PROVIDER=cohere`
(same Milvus/Neo4j, cached gold).

| Repo | treeloom (default) | treeloom (**Cohere**) |
|------|--------------------|------------------------|
| PowerToys (C#) | r@5 0.76 · corr 3.86 | r@5 **0.81** · corr **3.92** |
| guava (Java) | r@5 0.68 · corr 3.73 | r@5 0.74 · corr **3.83** |
| three.js (JS) | r@5 0.75 · corr 4.05 | r@5 0.75 · corr **4.19** |

Pooled treeloom-Cohere **vs grep**, 3 external repos (n=300, per-query sign test):
- **correctness: W58 / L50 / T192 → p=0.50** (statistical tie)
- **recall@5: W24 / L43 / T233 → p=0.027** (grep still ahead)

### Conclusion
The featbit quality lift is **real and reproduced on the control** (correctness 4.09 vs
3.71, **p=0.0045**, at token parity) — but it **does not generalize** to the three external
repos. There, treeloom is **quality-tied** with a plain grep/read agent, **behind on
retrieval recall**, and **costs 20–29% more tokens**. The dominant quality lever (Cohere
rerank-v4.0-pro) **narrows the gap from "behind" to "even"** (correctness parity, p=0.50)
but does not produce a win, and grep keeps a significant recall@5 edge (p=0.027).

Net for adoption: on the home repo Treeloom genuinely helps; on fresh repos under faithful
n=100 evaluation the current default configuration is **parity-at-a-token-premium**, even
with the best reranker. That is the "does it generalize?" question answered with
significance and a control — and, as the next section shows, answered for the wrong
query style.

**Caveats.** Queries are symbol-bearing (the identifier appears in the query), which favors
grep on retrieval — though featbit's queries are symbol-bearing too and treeloom still won
on quality there, so that is not the whole story. High tie counts (≈190/300 correctness)
mean both arms answer most clean queries equally. Treeloom's structural wins (repos not on
local disk, cross-repo, very large codebases) are not stressed here.

---

## D1: symbol-free queries (2026-06-15) — the corrected verdict

D1, D2 and D3 are the three follow-up experiments to the n=100 study: D1 re-runs it
with symbol-free queries, D2 repairs the graph index signals, D3 ablates the graph layer.

The n=100 section above used **symbol-bearing** clean queries (the identifier appears in
the query). That structurally favors grep — it can string-match the symbol — and
under-tests treeloom's semantic retrieval. The follow-up investigation traced the "doesn't
generalize" result to this: on the subset where grep *can't* match the symbol, treeloom
won decisively (grep-miss + symbol-free, pooled W28/L2/T13, **p<0.0001**).

So we re-ran with **100%-symbol-free** query sets (`benchmark queries --llm-symbol-free`:
an LLM paraphrases each entity into a behavioral, identifier-free question; entity-anchored
ground truth; validated grep-resistant). n=100/repo, grep vs treeloom (default reranker),
deepseek-chat.

| Repo | tokens (grep→tl) | recall@5 (p) | correctness (p) |
|------|------------------|--------------|-----------------|
| PowerToys (C#) | 43.3k → 32.0k (**+26% saved**) | 0.27 → **0.53** (**p<0.0001**) | 2.27 → **3.39** (**p<0.0001**) |
| guava (Java) | parity (+0.5%) | 0.81 → 0.80 (ns) | 3.83 → **4.09** (**p=0.043**) |
| three.js (JS) | 31.8k → 35.3k (−11%) | 0.76 → 0.70 (ns) | 3.80 → **4.01** (p=0.099) |
| **Pooled (300)** | **+7.5% saved** | tl ahead (p=0.056) | tl ahead (**p<0.0001**) |

### The swing vs the symbol-bearing baseline

| metric (pooled, 3 external repos) | symbol-bearing | symbol-free |
|-----------------------------------|----------------|-------------|
| correctness | p=0.079 (featbit-driven) | **treeloom p<0.0001** |
| recall@5 | **grep** ahead, p=0.003 | treeloom ahead, p=0.056 |
| tokens | treeloom **+20%** | treeloom **−7.5% (cheaper)** |

### Conclusion
On the realistic "I don't know the symbol name" query — the case treeloom exists for —
**treeloom significantly beats a grep/read agent on answer quality (pooled p<0.0001) at
better token efficiency (−7.5%)**, and edges ahead on retrieval. PowerToys is the clearest:
grep collapses (recall 0.27, correctness 2.27, 43k tokens of failed greps) while treeloom
holds (0.53 / 3.39 / 32k). The earlier "parity-at-a-premium" result was an artifact of
symbol-bearing queries; **treeloom's value is semantic search, not symbol lookup** — grep
already wins symbol lookup, and that should be stated plainly in positioning.

**Caveats.** PowerToys still carries the community-detection confound (49% of its C#
entities lack `community_id` after the graph rebuild) and won anyway — repairing it should
only strengthen the result. three.js is weakest (its grep recall stays high even
symbol-free) but still trends treeloom on correctness. The Cohere arm on the symbol-free
sets and the D2/D3 follow-ups (index-signal repair + ablations) are in the next section.

---

## D2/D3 (2026-06-15): what the graph layer actually contributes

Follow-up ablations on the symbol-free sets (Cohere reranker).

**D2 — repairing index signals had no effect.** PowerToys' C# graph rebuild had
left 49% of entities without `community_id` (community detection wasn't re-run).
Running `run_post_index_signals` brought it to 100% community_id + centrality +
community embeddings (1637 communities). Re-measured: 0.62→0.60 recall, 3.50→3.38
correctness — **within noise.** The index-quality gap was not suppressing treeloom.

**D3 — the value is the payload, not the rescoring.** Symbol-free, Cohere, treeloom:

| config | PowerToys (r@5/corr/tok) | guava (r@5/corr/tok) |
|--------|--------------------------|----------------------|
| full (graph rescore + neighbor/community payload) | 0.60 / 3.38 / 26k | 0.88 / 4.04 / 25k |
| **no-graph-scoring** (payload kept, reranker-only rank) | 0.60 / **3.43** / 26.6k | 0.88 / **4.12** / **22.8k** |
| vector-only (no graph at all, lean payload) | 0.60 / 3.15 / 33k | 0.88 / 3.93 / 36k |

- **recall@5 is reranker-driven** — identical whether graph scoring is on or off.
- **Graph *rescoring* (GRAPH_ALPHA/BETA/GAMMA/DELTA) is ~neutral** — `no-graph-scoring`
  ties or beats `full` on correctness and is cheaper. (Explains D2's null: community
  signals only feed rescoring.)
- **The neighbor/community *payload* is the real graph contribution** — dropping it
  (vector-only) lowers correctness and inflates tokens (lean → more agent rounds).

**Recommendation (Cohere/strong reranker):** `use_graph_scoring=false` while keeping
the neighbor/community payload — equal/better answer quality, identical retrieval,
fewer tokens. The structural value is feeding the agent context, not re-ranking by it.

**Status — the default was NOT flipped; here is why.** Re-validating the
flip against live infra surfaced that D3's "rescoring adds ~0" is **reranker-specific**.
D3 used Cohere `rerank-v4.0-pro` (a strong reranker that already nails rank-1). Under
the **production** reranker (`Qwen3-Reranker-0.6B`, `:8086`), a deterministic retrieval
re-run (off vs on, same indexer, per-request flag) found rescoring still does real work:

| repo (symbol-free, n=100) | recall@1 off→on | recall@5 off→on | MRR off→on |
|---|---|---|---|
| **guava** | **0.16 → 0.49** | 0.80 → 0.80 | **0.448 → 0.621** |
| PowerToys | 0.50 → 0.51 | 0.59 → 0.59 | 0.537 → 0.541 (noise) |
| featbit (non-regression) | 0.66 → 0.66 | 0.87 → 0.87 | 0.750 → 0.750 (identical) |

`recall@5` is reranker-driven (identical), but on guava rescoring lifts **recall@1 by
33 points** — the documented graph "symbol promotion" fixing rank-1, which a weaker
reranker leaves room for. The ±0.04 HyDE/Louvain noise can't explain it, and off/on ran
back-to-back in one process (HyDE cache held constant).

**Decision:** keep `USE_GRAPH_SCORING=1` (rescoring on) as the default so the production
reranker config doesn't regress rank-1; ship **payload-without-rescoring as a first-class
opt-in** (`use_graph_scoring=false` / `USE_GRAPH_SCORING=0`), recommended only with a
Cohere-class reranker where D3 showed it's a free win. The takeaway: the
rescore-vs-payload tradeoff is a function of reranker strength, not absolute.

## 2026-09 refresh (2026-09-21): current models, pinned corpus — cost holds, the quality edge does not

**Question.** Do the verdicts above hold for current agent models, on a pinned corpus, with
an independent judge?

**Short answer.** The **cost** verdict holds and strengthens; the **answer-quality** verdict
does not replicate. Across deepseek-flash, claude-sonnet-5 and gpt-5.5, treeloom matched a
grep/glob/read agent's answer quality at **11–47% lower mean cost** in all six cells
(significant per query in 5 of 6, and on symbol-free questions for every model,
p ≤ 5.6e-07). Its correctness was higher in 5 of 6 cells but **not significant in any**
(closest p = 0.061, gpt-5.5 symbol-free). This qualifies **D1** above: its pooled
symbol-free correctness win (p < 0.0001) came from `deepseek-chat` on the pre-2026-07-05
harness, across four repos including PowerToys where grep collapsed. Current capable
agents on featbit keep grep competitive — they reach the answer, at far higher cost.

The quality edge looks **agent-capability-dependent** rather than simply absent. The
same-epoch gpt-4o anchor pair (2026-07-05, see [benchmark-eval.md](benchmark-eval.md))
found treeloom significantly better on correctness on featbit — symbol-free p = 2.3e-10,
with grep capping out on 25.8% of queries. A weaker agent collapses when grep has nothing
to match; a capable one recovers. That anchor carries two caveats of its own: gpt-4o
judged its own answers, and it ran on an unpinned checkout.

**Configuration.** featbit pinned at tag `v5.4.9` (`c16207823d7b26e25bcbbe2a2eb032b602eea519`,
verified against the index); `featbit-v549-clean-100` (symbol-bearing) and
`featbit-v549-symfree-100` (symbol-free), n=100 each; judge and gold `gpt-4o-2024-08-06`
for every cell; production reranker `Qwen3-Reranker-0.6B`; harness defaults matching the
pinned anchor cells (obs 4000, native tools, scoped search, 12-turn cap).

| agent | questions | cost vs grep | per-query cost W/L (p) | correctness grep → treeloom (p) | grep / treeloom 12-turn caps |
|---|---|---|---|---|---|
| deepseek-flash | symbol-bearing | −11% | 60/40 (0.057) | 4.52 → 4.49 (0.85) | 0 / 0 |
| deepseek-flash | symbol-free | −30% | 75/25 (5.6e-07) | 4.40 → 4.56 (0.36) | 11 / 4 |
| claude-sonnet-5 | symbol-bearing | −14% | 68/32 (0.00041) | 4.41 → 4.47 (0.29) | 2 / 1 |
| claude-sonnet-5 | symbol-free | −25% | 73/27 (4.7e-06) | 4.14 → 4.43 (0.19) | 11 / 3 |
| gpt-5.5 | symbol-bearing | −20% | 61/39 (0.035) | 4.30 → 4.38 (0.35) | 5 / 6 |
| gpt-5.5 | symbol-free | −47% | 90/10 (3.1e-17) | 4.05 → 4.41 (0.061) | 24 / 8 |

What else it settles:

- **Symbol lookup is grep's ground.** Ranking vs grep is a tie in five cells; in the
  sixth (deepseek-flash, symbol-bearing) grep ranks the right file first significantly
  more often (recall@1 6/18, p = 0.023). Consistent with the verdict above: semantic
  search, not symbol lookup.
- **Turn-cap exhaustion.** On symbol-free questions treeloom caps out significantly less
  than grep on every model (p = 0.039, 0.0078, 0.00015).
- **vs claude-context** (deepseek-flash, same embedding model): treeloom ranks the right
  file first significantly more often — recall@1 29/9 (p = 0.0017) and 24/8 (p = 0.007) —
  at 11–22% fewer tokens, with correctness tied. An earlier June 2026 comparison
  (recall@1 0.71 vs 0.39, pre-rebuild harness, not reproduced in these docs) is
  superseded by 0.68 vs 0.48 (symbol-bearing) and 0.44 vs 0.28 (symbol-free).

**Decision.** Public positioning leads with **cost**, not answer quality: *the same answers
for less*, largest on symbol-free questions. Do not cite D1's p < 0.0001 without its model,
epoch and repo set. Full tables, threats to validity, setup and reproduction steps:
[benchmark-results-2026-09.md](benchmark-results-2026-09.md).
