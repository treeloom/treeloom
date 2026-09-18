"""Benchmark CLI — query generation, scoring, and AB comparison."""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import sys
from pathlib import Path


def _resolve_scope(args) -> str | None:
    """Repo-scope or fail loud — never silently search the whole index.

    Returns the path_prefix to scope to, or None when the caller explicitly
    opted into a cross-repo search. Exits immediately if neither --path-prefix
    nor --cross-repo-search was given: an unscoped whole-index search lets
    cross-repo symbol collisions corrupt name/definition signals (the silent
    default that skewed earlier numbers), so it must be a deliberate choice.
    """
    if args.path_prefix:
        return args.path_prefix
    if getattr(args, "cross_repo_search", False):
        return None
    sys.exit(
        "error: refusing an unscoped (whole-index) search.\n"
        "  Pass --path-prefix <repo path> to scope to one repo, or\n"
        "  --cross-repo-search to deliberately search the whole multi-repo index."
    )


def _add_scope_args(p: argparse.ArgumentParser) -> None:
    """Attach the mutually-aware --path-prefix / --cross-repo-search pair."""
    p.add_argument("--path-prefix", default=None,
                   help="Scope search to one repo (e.g. /data/repos/godotengine_godot) "
                        "— matches the agentic arm. Required unless --cross-repo-search.")
    p.add_argument("--cross-repo-search", action="store_true",
                   help="Deliberately search the whole multi-repo index (no scope). "
                        "Without this AND without --path-prefix, the search errors out.")


def _load_queries(path: str) -> list[dict]:
    # Query files in benchmarks/queries/ are JSONL (one object per line), but
    # earlier ad-hoc sets were a single JSON array. Accept both: array if the
    # first non-space char is '[', else line-delimited JSON.
    with open(path) as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _warn_curation(queries: list[dict], path: str) -> None:
    """Loudly warn when a set contains unreviewed pre-screen-flagged rows.

    Acceptance rule: a benchmark run must not silently consume rows the
    off-domain pre-screen flagged before a human ruled on them — flagged rows
    have already produced wrong ground truth (the featbit "insurance" rows).
    Curated sets carry a per-row `curation` block (curation_export_argilla);
    sets without any block predate the workflow and stay silent.
    """
    from treeloom.domain.benchmark.curation import curation_summary

    counts = curation_summary(queries)
    flagged = counts.get("flagged", 0)
    if flagged:
        print(f"  WARNING: {path}: {flagged} row(s) are pre-screen FLAGGED and "
              f"unreviewed — results include potentially wrong ground truth. "
              f"Review them in Argilla and re-export before citing this run.")
    unreviewed = counts.get("unreviewed", 0)
    if (flagged or unreviewed) and counts.get("uncurated", 0) == 0:
        print(f"  curation status of {path}: {counts}")


def cmd_queries(args):
    """Generate queries for a repo.

    Default: LLM-invented developer questions over a file listing.
    With --strategy: deterministic, entity-anchored queries from the indexed
    graph via the prompt-enhancer (rich JSONL schema for the agentic benchmark).
    """
    if args.llm_symbol_free:
        _cmd_queries_llm_symfree(args)
        return
    if args.strategy:
        _cmd_queries_enhanced(args)
        return

    if not args.source_dir:
        raise SystemExit("--source-dir is required (or pass --strategy)")
    from treeloom.application.benchmark.query_gen import generate_queries

    queries = asyncio.run(generate_queries(args.source_dir, n=args.count))
    out_dir = Path(args.output or "benchmarks/queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source_dir).name
    out_path = out_dir / f"{stem}.json"
    with open(out_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Generated {len(queries)} queries → {out_path}")
    for q in queries:
        print(f"  Q: {q['query'][:80]}...")


# Below this many kept symbol-free queries the generated set is too thin to
# benchmark on; recommend the LLM-paraphrase path instead.
SYMBOL_FREE_MIN_YIELD = 5


def _symbol_free_yield_advice(
    kept: int, total: int, scope: str, *, min_yield: int = SYMBOL_FREE_MIN_YIELD
) -> list[str] | None:
    """Headline + hint lines when --symbol-free yields too few queries.

    Pure (no I/O) so it's unit-testable. Returns the lines to print, or None
    when the kept count clears `min_yield` (no warning needed). `--llm-symbol-free`
    is always the FIRST, explicit recommendation: C#/Java/undocumented sources
    have few/no docstrings, so the deterministic symbol-free filter strips nearly
    everything, while the LLM-paraphrase path produces a grep-resistant set.
    """
    if kept >= min_yield:
        return None
    return [
        f"WARNING: --symbol-free kept only {kept}/{total} queries for {scope}.",
        "  C#/Java/undocumented sources have few/no docstrings; use "
        "--llm-symbol-free (LLM-paraphrased behavioral queries) for a "
        "grep-resistant set instead.",
        "  See docs/benchmark-eval.md (\"symbol-free vs llm-symbol-free\") and "
        "docs/benchmark-findings.md.",
        "  Alternatives: drop --symbol-free, or pick a docstring-rich repo.",
    ]


def _cmd_queries_enhanced(args):
    """Strategy-based, entity-anchored query generation from the graph."""
    import difflib
    from treeloom.application.benchmark.enhanced_query_gen import (
        DEFAULT_EXCLUDES,
        available_repo_prefixes,
        generate_enhanced_queries,
        resolve_strategies,
    )

    if not args.source_dir and not args.source_id:
        raise SystemExit("--strategy requires --source-dir and/or --source-id")
    strategies = resolve_strategies(args.strategy)
    # Use the prefix verbatim — it must match the file paths stored in the
    # graph at index time (which may be container paths like /data/repos/...,
    # not the host realpath). Don't canonicalize.
    path_prefix = args.source_dir.rstrip("/") if args.source_dir else None
    repo = args.repo_name or (Path(args.source_dir).name if args.source_dir else "repo")
    # --exclude unset -> sensible defaults; --exclude "" -> no filtering;
    # --exclude "a,b" -> those substrings.
    if args.exclude is None:
        excludes = list(DEFAULT_EXCLUDES)
    else:
        excludes = [s.strip() for s in args.exclude.split(",") if s.strip()]
    if excludes:
        print(f"Excluding paths containing: {excludes}")

    min_yield = getattr(args, "symbol_free_min_yield", SYMBOL_FREE_MIN_YIELD)

    async def _run():
        rows, dropped = await generate_enhanced_queries(
            repo=repo, strategies=strategies, source_id=args.source_id or None,
            path_prefix=path_prefix, exclude=excludes, symbol_free=args.symbol_free,
            max_entities=args.max_entities, max_per_entity=args.max_per_entity,
            sample_seed=args.sample_seed,
        )
        # Diagnose an empty result in the same event loop (the async Neo4j
        # driver is bound to it) so we can suggest valid targets.
        prefixes = [] if rows else await available_repo_prefixes()
        return rows, dropped, prefixes

    rows, dropped, prefixes = asyncio.run(_run())

    scope = (f"source_id {args.source_id!r}" if args.source_id
             else f"path prefix {path_prefix!r}")

    # Low-yield (incl. zero) symbol-free set: steer to --llm-symbol-free as the
    # headline. The pre-filter total is kept + dropped, so even when every query
    # was stripped (rows empty) we report what was generated before filtering.
    if args.symbol_free:
        kept = len(rows)
        total = kept + dropped
        # Only fire when entities actually existed (total > 0) — a truly empty
        # graph match is a different problem handled by the not-indexed branch.
        if total > 0:
            advice = _symbol_free_yield_advice(
                kept, total, scope, min_yield=min_yield
            )
            if advice is not None:
                for ln in advice:
                    print(ln)
                if not rows:
                    raise SystemExit(1)

    if not rows:
        target = args.source_id or path_prefix or ""
        indexed = bool(args.source_id) or any(
            path_prefix and (path_prefix == p or path_prefix.startswith(p)
                             or p.startswith(path_prefix))
            for p, _ in prefixes
        )
        if args.symbol_free and indexed:
            # Entities exist; --symbol-free filtered everything out. The
            # --llm-symbol-free recommendation was already printed above.
            raise SystemExit(1)
        print(f"WARNING: no Class/Function/Method entities matched {scope}.")
        print("  Nothing written — query generation only works for sources that "
              "are already indexed into the graph. Is that repo indexed?")
        names = [p for p, _ in prefixes]
        counts = dict(prefixes)
        close = difflib.get_close_matches(target, names, n=5, cutoff=0.2) if target else []
        if close:
            print("  Closest indexed prefixes:")
            for n in close:
                print(f"    {n}  ({counts.get(n)} entities)")
        if prefixes:
            print("  Largest indexed repos:")
            for n, c in prefixes[:8]:
                print(f"    {n}  ({c})")
        raise SystemExit(1)
    out_dir = Path("benchmarks/queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{repo}-enhanced.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_strat: dict[str, int] = {}
    for r in rows:
        by_strat[r["strategy"]] = by_strat.get(r["strategy"], 0) + 1
    print(f"Generated {len(rows)} enhanced queries → {out_path}")
    print(f"  by strategy: {by_strat}")
    for r in rows[:3]:
        print(f"  [{r['strategy']}/{r['difficulty']}] {r['query'][:70]}")


def _cmd_queries_llm_symfree(args):
    """LLM-paraphrased, symbol-free query generation (grep-resistant set)."""
    from treeloom.adapters.benchmark.agent_llm import AgentLLM
    from treeloom.adapters.benchmark.llm_client import describe_config
    from treeloom.application.benchmark.enhanced_query_gen import DEFAULT_EXCLUDES
    from treeloom.application.benchmark.llm_query_gen import generate_symbol_free_queries

    if not args.source_dir and not args.source_id:
        raise SystemExit("--llm-symbol-free requires --source-dir and/or --source-id")
    llm = AgentLLM(model=args.model) if args.model else AgentLLM()
    line, warnings = describe_config(
        {"model": llm.model, "url": llm.url, "key": llm.key, "timeout": llm.timeout}
    )
    print(f"llm-symbol-free: generating with {line}")
    for w in warnings:
        print(f"  WARNING: {w}")
    path_prefix = args.source_dir.rstrip("/") if args.source_dir else None
    repo = args.repo_name or (Path(args.source_dir).name if args.source_dir else "repo")
    excludes = (DEFAULT_EXCLUDES if args.exclude is None
                else [s.strip() for s in args.exclude.split(",") if s.strip()])

    out_dir = Path("benchmarks/queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{repo}-symfree.jsonl"
    # Sequential generation at ~1 LLM round trip per entity means a few
    # thousand entities is ~an hour of PAID calls. Stream them to <out>.partial
    # so a kill costs the remainder, not the whole run; re-running the same
    # command resumes from it.
    checkpoint = str(out_path) + ".partial"

    rows = asyncio.run(generate_symbol_free_queries(
        repo=repo, llm=llm, source_id=args.source_id or None,
        path_prefix=path_prefix, exclude=excludes, max_entities=args.max_entities,
        sample_seed=args.sample_seed, checkpoint_path=checkpoint,
    ))
    if not rows:
        print("No symbol-free queries produced (model couldn't avoid the symbol, "
              "or no readable source). Try a different repo or a stronger --model.")
        raise SystemExit(1)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # Only now is the checkpoint redundant.
    try:
        os.remove(checkpoint)
    except OSError:
        pass
    print(f"Generated {len(rows)} symbol-free queries → {out_path}")
    for r in rows[:3]:
        print(f"  {r['query'][:72]!r} -> {r['relevant_files'][0].split('/')[-1]}")


def cmd_run(args):
    """Run benchmark against search API."""
    from treeloom.application.benchmark.runner import run_benchmark

    queries = _load_queries(args.queries)
    _warn_curation(queries, args.queries)
    flags = {}
    if args.no_hyde:
        flags["use_hyde"] = False
    if args.no_hybrid:
        flags["use_hybrid"] = False
    # Graph RESCORING defaults ON (kept on because it lifts rank-1 under
    # the production reranker; payload-only is a Cohere-class opt-in).
    # --no-graph / --no-graph-scoring turn the post-rerank rescore OFF;
    # --graph-scoring pins it ON explicitly.
    if getattr(args, "graph_scoring", False):
        flags["use_graph_scoring"] = True
    if args.no_graph:
        flags["use_graph_scoring"] = False

    path_prefix = _resolve_scope(args)
    result = asyncio.run(run_benchmark(
        queries, args.search_url, feature_flags=flags or None,
        path_prefix=path_prefix, cross_repo=args.cross_repo_search,
    ))
    print(json.dumps(result, indent=2))


def cmd_ab(args):
    """Compare multiple feature flag configs."""
    from treeloom.application.benchmark.ab_compare import ab_compare

    queries = _load_queries(args.queries)
    _warn_curation(queries, args.queries)
    configs = [
        {"name": "baseline", "flags": {}},
        {"name": "+HyDE", "flags": {"use_hyde": True, "use_hybrid": False}},
        {"name": "+Hybrid", "flags": {"use_hyde": False, "use_hybrid": True}},
        {"name": "HyDE+Hybrid", "flags": {"use_hyde": True, "use_hybrid": True}},
    ]
    if args.graph:
        configs.append({"name": "Full (HyDE+Hybrid+Graph)",
                        "flags": {"use_hyde": True, "use_hybrid": True,
                                  "use_graph_scoring": True}})

    path_prefix = _resolve_scope(args)
    result = asyncio.run(ab_compare(queries, args.search_url, configs,
                                    path_prefix=path_prefix,
                                    cross_repo=args.cross_repo_search))
    print(json.dumps(result, indent=2))


def cmd_agentic(args):
    """Real agentic loop: grep / treeloom / claude-context arms (tokens + answer quality)."""
    from treeloom.domain.benchmark.queries import load_queries
    from treeloom.application.benchmark.agentic_runner import run_agentic
    from treeloom.domain.benchmark.rollup import format_comparison_table

    queries = load_queries(args.queries, args.difficulty, args.strategy)
    _warn_curation(queries, args.queries)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("No queries matched filters.")
        return
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    repo_name = queries[0].get("repo") or Path(args.repo).name

    # Tri-state protocol override: --native-tools forces the native loop,
    # --no-native-tools forces ReAct, neither (None) = per-vendor default.
    native_tools = True if args.native_tools else (
        False if args.no_native_tools else None)

    result = asyncio.run(run_agentic(
        queries, repo=args.repo, search_url=args.search_url, arms=arms,
        max_turns=args.max_turns, model=args.model, judge_model=args.judge_model,
        regen_gold=args.regen_gold, repo_name=repo_name, search_top_k=args.search_top_k,
        lean_search=args.lean_search, context_window=args.context_window,
        scope_treeloom=not args.no_scope_treeloom, vector_only=args.vector_only,
        graph_scoring=not args.no_graph_scoring, repomap_tokens=args.repomap_tokens,
        obs_char_limit=args.obs_char_limit,
        debug_transcripts=args.debug_transcripts,
        native_tools=native_tools,
        allow_floating_model=args.allow_floating_model,
    ))
    print(json.dumps(result, indent=2))
    print(format_comparison_table(result))


def cmd_rollup(args):
    """Aggregate per-repo agentic summary JSON files into a multi-repo rollup."""
    import datetime
    from treeloom.domain.benchmark.rollup import format_rollup_table, multi_repo_rollup

    summaries = []
    for path in args.summaries:
        with open(path) as f:
            summaries.append(json.load(f))

    rollup = multi_repo_rollup(summaries)
    print(format_rollup_table(rollup))

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = Path("benchmarks/results/agentic") / f"rollup_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rollup, f, indent=2)
    print(f"Rollup written → {out_path}")


def cmd_gold(args):
    """Build & cache gold (reference) answers from ground truth."""
    from treeloom.domain.benchmark.queries import load_queries
    from treeloom.application.benchmark.gold import build_gold, gold_path
    from treeloom.adapters.benchmark.agent_llm import AgentLLM
    from treeloom.adapters.benchmark.llm_client import describe_config

    gold_llm = AgentLLM(model=args.model, max_tokens=512) if args.model else AgentLLM(max_tokens=512)
    line, warnings = describe_config(
        {"model": gold_llm.model, "url": gold_llm.url, "key": gold_llm.key,
         "timeout": gold_llm.timeout}
    )
    print(f"gold: generating with {line}")
    for w in warnings:
        print(f"  WARNING: {w}")

    queries = load_queries(args.queries)
    _warn_curation(queries, args.queries)
    if args.limit:
        queries = queries[: args.limit]
    repo_name = queries[0].get("repo") or Path(args.queries).stem
    cache = asyncio.run(build_gold(queries, repo_name, gold_llm, regen=args.regen))
    path = gold_path(repo_name)
    print(f"Built {len(cache)} gold answers → {path}")
    for rec in list(cache.values())[:3]:
        print(f"  {rec['id']}: {rec.get('gold_answer','')[:90]}...")


def cmd_clean(args):
    """Filter a query JSONL to answerable, deduplicated rows."""
    from treeloom.domain.benchmark.query_hygiene import (
        apply_quota,
        dedup,
        filter_answerable,
    )

    rows = _load_queries(args.inp)

    # Derive extra stop parts from --repo path components (if given).
    stop_parts: set[str] = set()
    if args.repo:
        for part in Path(args.repo).parts:
            token = part.lower()
            if len(token) > 3:
                stop_parts.add(token)

    if args.symbol_free:
        # is_answerable requires the query to MENTION its entity name; a
        # symbol-free query is generated to OMIT it. The two contracts are
        # contradictory, so applying the name rules here would drop every row.
        # Such a query is answerable by construction -- it was written from the
        # entity's own source -- so keep any row with a query and ground truth.
        candidates = [r for r in rows
                      if (r.get("query") or "").strip() and r.get("relevant_files")]
    else:
        candidates = filter_answerable(rows, stop_parts=stop_parts)
    random.seed(args.seed)
    random.shuffle(candidates)

    # Parse --per-strategy into a quota dict (None means no capping).
    quota: dict[str, int] | None = None
    if args.per_strategy:
        quota = {}
        for pair in args.per_strategy.split(","):
            pair = pair.strip()
            if "=" not in pair:
                sys.exit(f"error: --per-strategy expects name=N pairs, got {pair!r}")
            k, _, v = pair.partition("=")
            quota[k.strip()] = int(v.strip())

    candidates = apply_quota(candidates, quota)
    candidates = dedup(candidates)

    if args.limit:
        candidates = candidates[: args.limit]

    if rows and not candidates:
        # This used to write an empty file and report "wrote 0 rows" with exit
        # 0 -- which is exactly what piping a symbol-free set through clean
        # produced, per the runbook. A 0-row benchmark set is never the goal.
        hint = ""
        if not args.symbol_free:
            name_free = sum(
                1 for r in rows
                if (r.get("entity") or {}).get("name", "").lower()
                not in (r.get("query") or "").lower()
            )
            if name_free >= len(rows) * 0.9:
                hint = (f" {name_free}/{len(rows)} rows never mention their entity "
                        "name -- this looks like a symbol-free set; re-run with "
                        "--symbol-free.")
        sys.exit(f"error: all {len(rows)} input rows were filtered out; "
                 f"refusing to write an empty set to {args.out}.{hint}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in candidates:
            f.write(json.dumps(r) + "\n")

    by_strat = collections.Counter((r.get("strategy") or "base") for r in candidates)
    print(f"wrote {len(candidates)} rows to {args.out}: {dict(by_strat)}")


def main():
    parser = argparse.ArgumentParser("treeloom.benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("queries", help="Generate test queries for a repo")
    p.add_argument("--source-dir", help="Repo root (LLM mode; entity path-prefix filter for --strategy)")
    p.add_argument("--count", type=int, default=5, help="LLM mode: number of queries")
    p.add_argument("--output")
    # --strategy switches to deterministic, entity-anchored generation from the
    # indexed graph via the prompt-enhancer.
    p.add_argument("--strategy", default="",
                   help="'all' or csv of: docstring,namespaced,entity_context,intent,cross_cutting,problem_driven")
    p.add_argument("--source-id", default="", help="Scope entities to one indexed source")
    p.add_argument("--repo-name", default="", help="Repo label for query ids/output (default: source-dir basename)")
    p.add_argument("--exclude", default=None,
                   help="csv of path substrings to exclude (default: docs/tests/examples/vendored; "
                        "pass --exclude '' to disable)")
    p.add_argument("--max-entities", type=int, default=200)
    p.add_argument("--sample-seed", type=int, default=None,
                   help="Seeded uniform sample of max-entities over the WHOLE entity "
                        "population instead of the first N in file-path order (which "
                        "clusters in the alphabetically-first subtree on large repos)")
    p.add_argument("--max-per-entity", type=int, default=3)
    p.add_argument("--symbol-free", action="store_true",
                   help="Keep only queries that don't contain the target's identifier "
                        "(stresses semantic retrieval; grep can't grep the symbol). "
                        "Yields ~0 on C#/Java/undocumented sources — prefer "
                        "--llm-symbol-free there.")
    p.add_argument("--symbol-free-min-yield", type=int, default=SYMBOL_FREE_MIN_YIELD,
                   help="Recommend --llm-symbol-free when --symbol-free keeps fewer "
                        f"than this many queries (default {SYMBOL_FREE_MIN_YIELD})")
    p.add_argument("--llm-symbol-free", action="store_true",
                   help="LLM-paraphrase a behavioral, symbol-free query per entity "
                        "(grep-resistant set; uses the benchmark LLM — set "
                        "BENCHMARK_QUALITY=STANDARD or --model)")
    p.add_argument("--model", default=None, help="LLM for --llm-symbol-free generation")
    p.set_defaults(func=cmd_queries)

    p = sub.add_parser("run", help="Run benchmark against search API")
    p.add_argument("--queries", required=True)
    p.add_argument("--search-url", default="http://localhost:8001")
    _add_scope_args(p)
    p.add_argument("--no-hyde", action="store_true")
    p.add_argument("--no-hybrid", action="store_true")
    p.add_argument("--no-graph", action="store_true",
                   help="Disable graph RESCORING (on by default; payload always on)")
    p.add_argument("--graph-scoring", action="store_true",
                   help="Pin the post-rerank graph RESCORING on (it is the default; payload is always on)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ab", help="AB compare feature flag configs")
    p.add_argument("--queries", required=True)
    p.add_argument("--search-url", default="http://localhost:8001")
    _add_scope_args(p)
    p.add_argument("--graph", action="store_true", help="Include graph scoring config")
    p.set_defaults(func=cmd_ab)

    p = sub.add_parser("agentic", help="Real agentic loop: grep vs treeloom vs claude-context arms")
    p.add_argument("--queries", required=True)
    p.add_argument("--repo", required=True, help="Repo root for grep/glob/read tools")
    p.add_argument("--search-url", default="http://localhost:8001")
    p.add_argument("--arms", default="grep,treeloom",
                   help="csv of arms to run: grep, treeloom, treeloom-facet, "
                        "treeloom-facet-hydrate, treeloom-summary-tail, claude-context, "
                        "repomap, code-graph-rag. The treeloom-facet arm runs treeloom with "
                        "response_mode=facet (metadata + chunk header, no code body; "
                        "the agent fetches ranges via read_file); treeloom-facet-hydrate "
                        " is the same facet payload but the agent gets a batch "
                        "hydrate_chunks tool to fetch dropped bodies by hit_id; "
                        "treeloom-summary-tail "
                        "runs response_mode=summary_tail (top-2 full snippets, tail "
                        "replaced by the indexed LLM summary — needs summary_cache "
                        "populated for the repo). The code-graph-rag arm spawns vitali87's "
                        "`cgr mcp-server` in an isolated py3.12 env (uvx; override "
                        "with CGR_MCP_CMD) and indexes --repo into Memgraph "
                        "(localhost:7688 by default — NOT 7687, that's our Neo4j; "
                        "docker run -d -p 7688:7687 memgraph/memgraph); needs "
                        "CYPHER_PROVIDER/MODEL/API_KEY/ENDPOINT for its "
                        "text-to-Cypher LLM. "
                        "The claude-context arm spawns zilliztech/claude-context's MCP "
                        "server via npx and indexes --repo through it on first run; "
                        "it needs MILVUS_ADDRESS (+MILVUS_TOKEN) and an embedding "
                        "provider env (EMBEDDING_PROVIDER/EMBEDDING_MODEL/"
                        "OPENAI_API_KEY/OPENAI_BASE_URL) in the environment. The "
                        "repomap arm builds aider's repo map once (via `uv run --with "
                        "aider-chat`, override with REPOMAP_PYTHON_CMD) and re-sends "
                        "it in the system prompt every turn — read_file/glob only, "
                        "no search tool")
    p.add_argument("--repomap-tokens", type=int, default=8192,
                   help="Token budget for the repomap arm's map (aider's effective "
                        "no-files-in-chat default is 8192)")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--obs-char-limit", type=int, default=4000,
                   help="Per-tool-observation char cap fed back to the agent (default "
                        "4000, modelling a capped client). RAISE it (e.g. 16000) when "
                        "measuring payload-size/serialization changes, or the cap "
                        "truncates both arms to the same size and HIDES the token "
                        "difference. Recorded in the run's _summary.json.")
    p.add_argument("--model", default=None,
                   help="Model for the measured agent arms (grep + treeloom). "
                        "Prefer a clean non-reasoning chat model for fair token counts.")
    p.add_argument("--judge-model", default=None,
                   help="Model for the evaluation side — gold reference answers + the "
                        "judge. Defaults to --model. Use a stronger model here for better "
                        "scoring without inflating the measured arms' tokens.")
    p.add_argument("--regen-gold", action="store_true")
    p.add_argument("--search-top-k", type=int, default=None,
                   help="Fix the treeloom search_code top_k (sweep payload size); "
                        "default = agent-chosen")
    p.add_argument("--lean-search", action="store_true",
                   help="Lean treeloom payload: chunks only (drop neighbors + "
                        "community summaries), snippets capped — cuts the dominant "
                        "tool_observation token cost")
    p.add_argument("--context-window", type=int, default=None,
                   help="Prune the agent transcript to system+query+last N messages "
                        "(both arms); attacks quadratic turn-cost growth. Default: full")
    p.add_argument("--no-scope-treeloom", action="store_true",
                   help="Don't scope treeloom search_code to --repo (search the whole "
                        "multi-repo index). Default: scoped, matching the repo-scoped grep arm")
    p.add_argument("--vector-only", action="store_true",
                   help="treeloom arm = pure Milvus vector+rerank, no graph rescoring and "
                        "no neighbor/community payload (implies lean). Isolates whether the "
                        "graph layer adds value or just tokens")
    p.add_argument("--no-graph-scoring", action="store_true",
                   help="Turn off graph *ranking* (rank by reranker only) while KEEPING the "
                        "full payload (neighbors + community + full snippets). Decoupled from "
                        "--vector-only: isolates the graph layer's ranking value from its "
                        "payload value")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--native-tools", action="store_true",
                   help="Force the native structured tool_calls protocol for the "
                        "agent loop (no ReAct sentinel parsing; termination = a "
                        "tool-less assistant message). Default: per-vendor — "
                        "native for openai/deepseek/anthropic models, ReAct for "
                        "local models. Protocol change = new baseline epoch: "
                        "never compare native runs against ReAct runs.")
    g.add_argument("--no-native-tools", action="store_true",
                   help="Force the legacy ReAct text protocol even for vendors "
                        "that support native tool_calls.")
    p.add_argument("--debug-transcripts", action="store_true",
                   help="Persist each arm's full agent transcript (assistant turns, "
                        "tool calls AND tool observations) into the per-query rows "
                        "JSONL. Off by default to keep row size sane; turn on when "
                        "diagnosing floored/ungrounded runs — without it a failure "
                        "like the 2026-06-25 incident is undecidable post-hoc.")
    p.add_argument("--allow-floating-model", action="store_true",
                   help="Permit a floating model alias (deepseek-chat, bare "
                        "gpt-4o, *-latest, ...) as --model/--judge-model. By "
                        "default the run refuses to start — vendors silently "
                        "remap these aliases (the 2026-06-25 incident). A "
                        "permitted run stamps floating_model_warning: true "
                        "into _summary.json. Env: TREELOOM_ALLOW_FLOATING_MODELS=1.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--difficulty", default="")
    p.add_argument("--strategy", default="")
    p.set_defaults(func=cmd_agentic)

    p = sub.add_parser("gold", help="Build & cache gold answers from ground truth")
    p.add_argument("--queries", required=True)
    p.add_argument("--repo", default="", help="(unused; repo name taken from queries)")
    p.add_argument("--model", default=None,
                   help="Model for gold answers (match the agentic --judge-model so the "
                        "cached gold is consistent with how it'll be judged)")
    p.add_argument("--regen", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_gold)

    p = sub.add_parser("rollup", help="Aggregate per-repo agentic summary JSON files into a multi-repo rollup")
    p.add_argument("summaries", nargs="+",
                   help="Paths to _summary.json files produced by the agentic subcommand")
    p.add_argument("--out", default=None,
                   help="Output path for the rollup JSON "
                        "(default: benchmarks/results/agentic/rollup_<UTC-timestamp>.json)")
    p.set_defaults(func=cmd_rollup)

    p = sub.add_parser("clean", help="Filter a query JSONL to answerable, deduplicated rows")
    p.add_argument("--in", dest="inp", required=True,
                   help="Input JSONL query file (enhanced query set)")
    p.add_argument("--out", required=True,
                   help="Output JSONL path for the cleaned query set")
    p.add_argument("--repo", default=None,
                   help="Optional repo path; path components (len>3) are added as stop "
                        "parts so repo-specific tokens don't count as path hints")
    p.add_argument("--limit", type=int, default=None,
                   help="Total row cap applied after dedup (truncates final list)")
    p.add_argument("--per-strategy", default=None,
                   help="Per-strategy quota, format name=N[,name2=M,...] "
                        "(e.g. 'base=15,intent=20'); omit to keep all answerable rows")
    p.add_argument("--symbol-free", action="store_true",
                   help="The input is a symbol-free set (--llm-symbol-free / "
                        "--symbol-free). Skips the name-mention rules, which such "
                        "queries fail BY DESIGN; dedup and the seeded sample still apply.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffle before quota/dedup (default: 42)")
    p.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
