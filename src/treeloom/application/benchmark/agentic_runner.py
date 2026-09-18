"""Benchmark application: agentic runner — grep / treeloom / claude-context arms.

Per query: run the selected arms (concurrently), compute read-based retrieval
recall/MRR, judge each arm's final answer against the cached gold answer, and
assemble a result row. Writes per-query JSONL + an aggregate summary.

The claude-context arm spawns zilliztech/claude-context's MCP server once for
the whole run (and indexes the repo through it before the first query) — see
adapters/benchmark/claude_context_tool.py for its env requirements.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from treeloom.adapters.benchmark.agent_llm import AgentLLM
from treeloom.adapters.benchmark.claude_context_tool import (
    ClaudeContextClient,
    claude_context_arm_tools,
)
from treeloom.adapters.benchmark.code_graph_rag_tool import (
    CodeGraphRagClient,
    code_graph_rag_arm_tools,
)
from treeloom.adapters.benchmark.grep_tools import grep_arm_tools
from treeloom.adapters.benchmark.repomap_tool import build_repo_map, repomap_arm_tools
from treeloom.adapters.benchmark.treeloom_tool import treeloom_arm_tools
from treeloom.application.benchmark.agent_loop import AgentRunResult, run_agent
from treeloom.application.benchmark.gold import build_gold
from treeloom.application.benchmark.judge import judge_answer
from treeloom.domain.benchmark.agent_metrics import aggregate
from treeloom.domain.benchmark.metrics import canonicalize_retrieved
from treeloom.domain.benchmark.metrics import compute_mrr, recall_at_k

RESULTS_DIR = Path("benchmarks/results/agentic")


def _claim_result_paths(results_dir: Path, stem: str) -> tuple[Path, Path]:
    """Atomically claim a unique rows/summary pair: <stem>[-N].jsonl + _summary.json.

    Names are `<UTC second>_<repo><suffix>`, so two runs on the same repo that
    start within the same second used to get the SAME path. Rows were opened
    with "w", so the second run truncated the first's file and both then kept
    writing into it through separate handles -- paid-for rows silently lost or
    interleaved. Observed risk, not theory: two parallel Cell B runs of the
    2026-09 refresh started one second apart.

    The rows file is created with O_EXCL ("x"), so the OS guarantees exactly
    one process wins a name; a loser takes the next `-N`. The disambiguator
    sits in the STEM, so rows and summary still pair by swapping `.jsonl` for
    `_summary.json` -- what scripts/agentic_smoke*.{sh,py} rely on. A model tag
    alone would not fix this: two runs of the same model collide just the same.
    """
    n = 1
    while True:
        s = stem if n == 1 else f"{stem}-{n}"
        rows = results_dir / f"{s}.jsonl"
        try:
            with open(rows, "x"):
                pass
        except FileExistsError:
            n += 1
            continue
        return rows, results_dir / f"{s}_summary.json"


def _rel(paths: list[str], repo_root: str) -> list[str]:
    """Reduce each path to its repo-root-relative form for cross-side comparison.

    Ground-truth `relevant_files` are container-style absolute paths
    (`/data/repos/<repo>/src/Foo.js`) while the agent's `retrieved_files` are
    absolute paths under the on-disk `--repo` checkout (e.g.
    `/home/user/source/<repo>/src/Foo.js`). The two never string-match unless the
    repo physically sits at the container path — which is the zero-recall path-normalization bug.

    We strip the *known basename* of the repo root from BOTH sides: anything at
    or after the final `<repo>/` path component becomes the comparison key, so
    `/data/repos/mrdoob_three.js/src/Foo.js` and
    `/home/user/source/mrdoob_three.js/src/Foo.js` both reduce to `src/Foo.js`.
    realpath is applied first to resolve symlinks within a side (but only when it
    doesn't erase the repo-name anchor we rely on).
    """
    root_name = os.path.basename(os.path.normpath(repo_root)) if repo_root else ""
    out: list[str] = []
    for p in paths:
        rel = _strip_repo_prefix(p, repo_root, root_name)
        if rel not in out:
            out.append(rel)
    return out


def _strip_repo_prefix(path: str, repo_root: str, root_name: str) -> str:
    """Return `path` made relative to `repo_root`, falling back to the repo-name
    anchor when it isn't a literal subpath of the on-disk root."""
    norm = os.path.normpath(path)
    # 1) Direct on-disk subpath of the actual repo root (the retrieved side).
    if repo_root:
        root_norm = os.path.normpath(repo_root)
        try:
            if os.path.commonpath([root_norm, norm]) == root_norm:
                return os.path.relpath(norm, root_norm)
        except ValueError:
            pass  # different drives / mixed abs+rel — fall through
    # 2) Container-style path that contains the repo-name component anywhere
    #    (the ground-truth side, e.g. /data/repos/<repo>/src/Foo.js).
    if root_name:
        parts = norm.split(os.sep)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == root_name:
                tail = parts[i + 1:]
                if tail:
                    return os.path.join(*tail)
    # 3) Already-relative or un-anchorable — return as-is (normalized).
    return norm.lstrip(os.sep) if os.path.isabs(norm) else norm


def _mrr_multi(retrieved: list[str], relevant: list[str]) -> float:
    """Best (earliest-rank) reciprocal rank across multiple ground-truth files."""
    return max((compute_mrr(retrieved, r) for r in relevant), default=0.0)


def enforce_floating_model_policy(
    model: str | None, judge_model: str | None, *, allow: bool = False
) -> bool:
    """Refuse (exit non-zero) when a benchmark run names a floating model alias.

    Floating aliases (see FLOATING_ALIASES in llm_client.py) can be silently
    remapped by the vendor — the 2026-06-25 `deepseek-chat` V3→v4-flash remap
    invalidated a day of agentic numbers. Applies to the AGENTIC path only
    (gold/queries/judge helpers elsewhere are unaffected), but checks the judge
    model too since it scores every answer.

    Override with `--allow-floating-model` or TREELOOM_ALLOW_FLOATING_MODELS=1;
    a permitted run returns True so the caller stamps
    `floating_model_warning: true` into _summary.json. Returns False when no
    floating alias is in play. Raises SystemExit(2) on refusal.
    """
    from treeloom.adapters.benchmark.llm_client import is_floating_alias

    allow = allow or os.environ.get("TREELOOM_ALLOW_FLOATING_MODELS", "") == "1"
    warnings: list[str] = []
    for role, name in (("agent model", model), ("judge model", judge_model)):
        msg = is_floating_alias(name)
        if msg:
            warnings.append(f"[agentic] FLOATING ALIAS ({role}): {msg}")
    if not warnings:
        return False
    for w in warnings:
        print(w)
    if allow:
        print("[agentic] floating alias permitted by override — stamping "
              "floating_model_warning: true into the summary.")
        return True
    print("[agentic] refusing to start: benchmark runs must not use floating "
          "model aliases. Re-run with --allow-floating-model or "
          "TREELOOM_ALLOW_FLOATING_MODELS=1 to override.")
    raise SystemExit(2)


def _effective_native_tools(override: bool | None, llm: AgentLLM) -> bool:
    """Resolve the agent protocol: an explicit CLI override (--native-tools /
    --no-native-tools) wins; otherwise default to the vendor's
    `supports_native_tools` route axis (native for openai/deepseek/anthropic,
    ReAct for local models)."""
    return llm.supports_native_tools if override is None else override


def _arm_row(rr: AgentRunResult, retrieved: list[str], relevant: list[str],
             judge_dict: dict, *, debug_transcripts: bool = False) -> dict:
    """Shape one arm's per-query result row.

    `transcript` (the loop's turn log INCLUDING the tool observations actually
    fed back) is written only with --debug-transcripts — rows stay lean by
    default, but a floored/ungrounded run can be diagnosed post-hoc.
    """
    row = {
        "final_answer": rr.final_answer,
        "turns": rr.turns,
        "tool_calls": rr.tool_calls,
        "hit_cap": rr.hit_cap,
        "error": rr.error,
        "latency_s": rr.latency_s,
        **rr.token_account.as_dict(),
        "retrieved_files": retrieved,
        "recall@1": recall_at_k(retrieved, relevant, 1),
        "recall@5": recall_at_k(retrieved, relevant, 5),
        "recall@10": recall_at_k(retrieved, relevant, 10),
        "mrr": round(_mrr_multi(retrieved, relevant), 4),
        "judge": judge_dict,
    }
    if debug_transcripts:
        row["transcript"] = rr.transcript
    return row


async def _run_arm(arm: str, query: str, repo: str, search_url: str,
                   llm: AgentLLM, max_turns: int,
                   search_top_k: int | None = None,
                   lean_search: bool = False,
                   context_window: int | None = None,
                   scope_treeloom: bool = True,
                   vector_only: bool = False,
                   graph_scoring: bool = True,
                   cc_client: ClaudeContextClient | None = None,
                   repomap_text: str = "",
                   cgr_client: CodeGraphRagClient | None = None,
                   obs_char_limit: int = 4000,
                   native_tools: bool = False) -> AgentRunResult:
    system_extra = ""
    if arm == "grep":
        log: list[str] = []
        tools = grep_arm_tools(repo, log)
    elif arm == "code-graph-rag":
        if cgr_client is None:
            raise ValueError("code-graph-rag arm requires a started CodeGraphRagClient")
        log = []
        tools = code_graph_rag_arm_tools(cgr_client, repo, log)
    elif arm == "claude-context":
        if cc_client is None:
            raise ValueError("claude-context arm requires a started ClaudeContextClient")
        log = []
        tools = claude_context_arm_tools(cc_client, repo, log, search_top_k)
    elif arm == "repomap":
        if not repomap_text:
            raise ValueError("repomap arm requires a pre-built repo map")
        log = []
        tools = repomap_arm_tools(repo, log)
        system_extra = repomap_text
    elif arm in ("treeloom", "treeloom-facet", "treeloom-summary-tail",
                 "treeloom-trimmed", "treeloom-st-brief",
                 "treeloom-facet-hydrate"):
        log = []
        # Opt-in response modes: the facet/summary-tail arms run
        # the treeloom arm with the respective server-side response_mode. The
        # facet arm relies on read_file (appended by treeloom_arm_tools when
        # repo_path is set) to fetch the body-less ranges it returns.
        # treeloom-facet-hydrate instead gets a batch hydrate_chunks tool
        # to fetch the dropped bodies by hit_id — the charitable facet variant.
        response_mode = {
            "treeloom": "full",
            "treeloom-facet": "facet",
            "treeloom-summary-tail": "summary_tail",
            "treeloom-trimmed": "full",
            "treeloom-st-brief": "summary_tail",
            "treeloom-facet-hydrate": "facet",
        }[arm]
        # Rejected payload trims (all three at once vs baseline `treeloom`):
        # drop community summaries + cut the low-confidence tail + strip import
        # lines. read_file is available (repo_path) so the agent can recover any
        # detail the trims removed — the turns metric catches it if it can't.
        trims = None
        if arm == "treeloom-trimmed":
            trims = {
                "include_community_summaries": False,
                "adaptive_topk": True,
                "strip_imports": True,
            }
        elif arm == "treeloom-st-brief":
            # summary-verbosity sweep: summary_tail using the ~41-token "brief"
            # summary tier (PROMPT_VERSION 9002) — the screen's best-quality
            # candidate; this confirms it in the full agentic loop.
            trims = {"summary_prompt_version": 9002}
        # Graph ranking is off when EITHER --vector-only (off + lean payload) or
        # --no-graph-scoring (off, full payload) is set. Lean is implied only by
        # --vector-only, so --no-graph-scoring isolates the ranking effect.
        tools = treeloom_arm_tools(
            search_url, log, search_top_k,
            lean=lean_search or vector_only,
            path_prefix=repo if scope_treeloom else None,
            use_graph_scoring=graph_scoring and not vector_only,
            repo_path=repo,
            response_mode=response_mode,
            trims=trims,
            with_hydrate=(arm == "treeloom-facet-hydrate"),
        )
    else:
        raise ValueError(f"unknown arm: {arm}")
    return await run_agent(
        query=query, tools=tools, llm=llm,
        retrieved_files=log, max_turns=max_turns, context_window=context_window,
        system_extra=system_extra, obs_char_limit=obs_char_limit,
        native_tools=native_tools,
    )


async def run_one_query(
    q: dict, gold_rec: dict, *, repo: str, search_url: str,
    agent_llm: AgentLLM, judge_llm: AgentLLM, arms: list[str], max_turns: int,
    search_top_k: int | None = None, lean_search: bool = False,
    context_window: int | None = None, scope_treeloom: bool = True,
    vector_only: bool = False, graph_scoring: bool = True,
    cc_client: ClaudeContextClient | None = None,
    repomap_text: str = "",
    cgr_client: CodeGraphRagClient | None = None,
    obs_char_limit: int = 4000,
    debug_transcripts: bool = False,
    native_tools: bool = False,
) -> dict:
    primary = _rel(q.get("relevant_files", []), repo)
    additional = _rel(q.get("additional_relevant_files") or [], repo)
    # Ground truth is `relevant_files` + optional `additional_relevant_files` --
    # the documented schema, and what runner.py (`gt |= additional...`) and
    # gold.py already honour. Scoring against `relevant_files` alone meant a
    # multi-file answer (e.g. a C# partial class split across Foo.cs and
    # Foo.Log.cs) scored zero recall in agentic cells while scoring correctly
    # in `benchmark run` -- the same query, two verdicts.
    relevant = primary + [f for f in additional if f not in primary]
    alternatives = {
        _rel([k], repo)[0]: _rel(v or [], repo)
        for k, v in (q.get("relevant_file_alternatives") or {}).items()
        if _rel([k], repo)
    }

    runs = await asyncio.gather(
        *[_run_arm(a, q["query"], repo, search_url, agent_llm, max_turns,
                   search_top_k, lean_search, context_window, scope_treeloom,
                   vector_only, graph_scoring, cc_client, repomap_text, cgr_client,
                   obs_char_limit=obs_char_limit, native_tools=native_tools)
          for a in arms]
    )
    arm_runs = dict(zip(arms, runs))

    # Judge all arms' answers concurrently against the gold answer.
    gold_answer = gold_rec.get("gold_answer", "")
    judge_scores = await asyncio.gather(
        *[judge_answer(q["query"], gold_answer, arm_runs[a].final_answer, judge_llm)
          for a in arms]
    )
    arm_judges = dict(zip(arms, judge_scores))

    arms_out: dict = {}
    for a in arms:
        rr = arm_runs[a]
        retrieved = canonicalize_retrieved(_rel(rr.retrieved_files, repo), alternatives)
        arms_out[a] = _arm_row(rr, retrieved, relevant, arm_judges[a].as_dict(),
                               debug_transcripts=debug_transcripts)

    return {
        "id": q["id"],
        "query": q["query"],
        "relevant_files": primary,
        "additional_relevant_files": [f for f in additional if f not in primary],
        "gold_answer": gold_answer,
        "arms": arms_out,
    }


async def run_agentic(
    queries: list[dict], *, repo: str, search_url: str, arms: list[str],
    max_turns: int = 12, model: str | None = None, judge_model: str | None = None,
    regen_gold: bool = False, repo_name: str = "repo", search_top_k: int | None = None,
    lean_search: bool = False, context_window: int | None = None,
    scope_treeloom: bool = True, vector_only: bool = False,
    graph_scoring: bool = True, repomap_tokens: int = 8192,
    obs_char_limit: int = 4000, debug_transcripts: bool = False,
    native_tools: bool | None = None, allow_floating_model: bool = False,
) -> dict:
    # The agent arms are what we measure (tokens); keep them on `model` (e.g. a
    # clean non-reasoning chat model). The evaluation side — gold reference
    # answers + the judge — can use a stronger `judge_model` for more reliable
    # scoring without polluting the measured token counts. Falls back to the
    # agent model when --judge-model isn't given.
    # 2048, not 512: the Jun-25 sonnet probes truncated EVERY completion at
    # exactly the 512 budget (completion_tokens = turns x 512), cutting ACTION
    # JSON and final answers mid-line — a confound on any grounding probe.
    # Reasoning models are still floored to REASONING_MAX_COMPLETION_TOKENS by
    # AgentLLM._budget regardless of this value.
    agent_llm = AgentLLM(model=model, temperature=0.0, max_tokens=2048)
    eval_model = judge_model or model
    judge_llm = AgentLLM(model=eval_model, temperature=0.0, max_tokens=256)
    gold_llm = AgentLLM(model=eval_model, temperature=0.0, max_tokens=512)

    # Floating-alias hygiene (Phase 3): checked on the RESOLVED names so a
    # tier-default `deepseek-chat` (no --model given) is caught too. Refuses
    # (SystemExit 2) before ANY indexing or API work unless explicitly allowed.
    floating_model_warning = enforce_floating_model_policy(
        agent_llm.model, judge_llm.model, allow=allow_floating_model
    )

    # Agent protocol: explicit --native-tools/--no-native-tools override wins;
    # a plain run gets the per-vendor default from resolve_model_route.
    use_native = _effective_native_tools(native_tools, agent_llm)
    agent_protocol = "native_tools" if use_native else "react"

    # Report the resolved model/endpoint + warn on misconfig, so a run never
    # silently uses the wrong (e.g. weak local) model.
    from treeloom.adapters.benchmark.llm_client import describe_config

    line, warnings = describe_config(
        {"model": agent_llm.model, "url": agent_llm.url, "key": agent_llm.key,
         "timeout": agent_llm.timeout}
    )
    print(f"[agentic] {line} arms={arms} max_turns={max_turns} queries={len(queries)}")
    print(f"[agentic] agent protocol: {agent_protocol}"
          + (" (per-vendor default)" if native_tools is None else " (CLI override)"))
    print(f"[agentic] eval model (gold + judge): {judge_llm.model}")
    print("[agentic] treeloom search scoped to repo: "
          + (f"True (path_prefix={repo})" if scope_treeloom else "False (whole index)"))
    if vector_only:
        mode = "VECTOR-ONLY (no graph rescoring, lean payload — no neighbors/community)"
    elif not graph_scoring:
        mode = "NO-GRAPH-SCORING (reranker-only ranking, full payload kept)"
    else:
        mode = "full (graph rescoring + neighbors + community)"
    print(f"[agentic] treeloom mode: {mode}")
    for w in warnings:
        print(f"[agentic]   WARNING: {w}")

    # The repomap arm's map is built once and re-sent every turn via the
    # system prompt (that recurring cost is the point of the arm).
    repomap_text = ""
    if "repomap" in arms:
        print(f"[agentic] repomap: building aider repo map for {repo} "
              f"(map_tokens={repomap_tokens}) ...")
        repomap_text = await asyncio.to_thread(build_repo_map, repo, repomap_tokens)
        print(f"[agentic] repomap: built ({len(repomap_text)} chars)")

    # The claude-context arm needs its MCP server up and the repo indexed in
    # ITS index (Milvus collection it owns) before the first query.
    cc_client: ClaudeContextClient | None = None
    if "claude-context" in arms:
        cc_client = ClaudeContextClient()
        await cc_client.start()
        print(f"[agentic] claude-context: ensuring index for {repo} ...")
        status = await cc_client.ensure_indexed(repo)
        print(f"[agentic] claude-context: {status.splitlines()[0] if status else 'indexed'}")

    # Ditto for code-graph-rag: its MCP server + a Memgraph knowledge graph.
    cgr_client: CodeGraphRagClient | None = None
    if "code-graph-rag" in arms:
        cgr_client = CodeGraphRagClient(repo)
        await cgr_client.start()
        print(f"[agentic] code-graph-rag: ensuring graph for {repo} ...")
        status = await cgr_client.ensure_indexed()
        print(f"[agentic] code-graph-rag: {status.splitlines()[0] if status else 'indexed'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    suffix = f"_k{search_top_k}" if search_top_k is not None else ""
    if lean_search:
        suffix += "_lean"
    if context_window is not None:
        suffix += f"_cw{context_window}"
    if not scope_treeloom:
        suffix += "_unscoped"
    if vector_only:
        suffix += "_vec"
    elif not graph_scoring:
        suffix += "_nogs"
    rows_path, summary_path = _claim_result_paths(RESULTS_DIR, f"{ts}_{repo_name}{suffix}")
    print(f"[agentic] writing per-query rows incrementally to {rows_path}")

    try:
        gold = await build_gold(queries, repo_name, gold_llm, regen=regen_gold)

        rows: list[dict] = []
        # Rows are appended + flushed per query so a crash mid-run (or a ^C)
        # keeps everything already paid for.
        # "a", never "w": the file was claimed empty by _claim_result_paths and
        # is ours alone; append can never truncate rows already written.
        with open(rows_path, "a") as rows_file:
            for i, q in enumerate(queries, 1):
                t0 = time.monotonic()
                row = await run_one_query(
                    q, gold.get(q["id"], {}), repo=repo, search_url=search_url,
                    agent_llm=agent_llm, judge_llm=judge_llm, arms=arms, max_turns=max_turns,
                    search_top_k=search_top_k, lean_search=lean_search,
                    context_window=context_window, scope_treeloom=scope_treeloom,
                    vector_only=vector_only, graph_scoring=graph_scoring,
                    cc_client=cc_client, repomap_text=repomap_text,
                    cgr_client=cgr_client, obs_char_limit=obs_char_limit,
                    debug_transcripts=debug_transcripts,
                    native_tools=use_native,
                )
                rows.append(row)
                rows_file.write(json.dumps(row) + "\n")
                rows_file.flush()
                summary = {a: {"tok": row["arms"][a]["total_tokens"],
                               "turns": row["arms"][a]["turns"],
                               "r@5": row["arms"][a]["recall@5"],
                               "corr": row["arms"][a]["judge"]["correctness"]}
                           for a in arms}
                print(f"  [{i}/{len(queries)}] {q['id']} ({time.monotonic()-t0:.1f}s) {summary}")
    finally:
        if cc_client is not None:
            await cc_client.close()
        if cgr_client is not None:
            await cgr_client.close()

    agg = aggregate(rows, arms, repo=repo_name, model=agent_llm.model)
    agg["judge_model"] = judge_llm.model
    # What the vendors say they actually served (vs the possibly-floating
    # `model`/`judge_model` names above) — the drift canary for alias remaps.
    agg["served_models"] = sorted(agent_llm.served_models)
    agg["judge_served_models"] = sorted(judge_llm.served_models | gold_llm.served_models)
    # The effective protocol this run actually used (protocol change = new
    # baseline epoch — never compare native_tools rows against react rows).
    agg["agent_protocol"] = agent_protocol
    if floating_model_warning:
        # A floating alias ran under the explicit override — mark the summary
        # so the number is never trusted without re-running the smoke fixture.
        agg["floating_model_warning"] = True
    agg["debug_transcripts"] = debug_transcripts
    agg["search_top_k"] = search_top_k  # None = agent-chosen
    agg["lean_search"] = lean_search
    agg["context_window"] = context_window
    agg["obs_char_limit"] = obs_char_limit
    agg["treeloom_scoped"] = scope_treeloom
    agg["treeloom_vector_only"] = vector_only
    agg["treeloom_graph_scoring"] = graph_scoring and not vector_only
    if "repomap" in arms:
        agg["repomap_tokens"] = repomap_tokens

    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)
    agg["_rows_path"] = str(rows_path)
    agg["_summary_path"] = str(summary_path)

    # Opt-in Opik experiment tracking — best-effort, post-hoc, no-op unless
    # OPIK_TRACK / OPIK_URL is set. Built entirely from the finished `rows`, so it
    # cannot affect a single measured token; any failure only prints a warning.
    from treeloom.adapters.benchmark import opik_tracing

    opik_tracing.log_run(
        run_name=f"{ts}_{repo_name}{suffix}",
        repo_name=repo_name, queries=queries, gold=gold, rows=rows, arms=arms,
        config={
            "search_url": search_url, "agent_model": agent_llm.model,
            "judge_model": judge_llm.model, "max_turns": max_turns,
            "agent_protocol": agent_protocol,
            "obs_char_limit": obs_char_limit, "search_top_k": search_top_k,
            "lean_search": lean_search, "context_window": context_window,
            "treeloom_scoped": scope_treeloom, "treeloom_vector_only": vector_only,
            "treeloom_graph_scoring": graph_scoring and not vector_only,
        },
    )
    return agg
