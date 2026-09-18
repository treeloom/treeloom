"""Opik experiment tracking for the agentic benchmark.

Opt-in, best-effort, post-hoc. After a run, `log_run()` mirrors each per-query /
per-arm result into Opik as:

  * a **dataset** per repo (the query set: query + gold + relevant_files),
  * one **experiment** per arm with the full run *config* as metadata — the
    thing that would have caught the v2-benchmark-hit-v1 bug (search_url is
    logged, not implicit in a filename),
  * one **trace** per (query, arm) carrying input/output, an `agent` span with
    token usage (incl. cached/reasoning, already summed by TokenAccount), and
    feedback scores (correctness/completeness/recall@1/recall@5/mrr),
  * **experiment items** linking dataset items → traces for the comparison UI.

Design constraints (learned the hard way):
  * ENTIRELY off unless `OPIK_TRACK=1` (or `OPIK_URL_OVERRIDE`/`OPIK_URL` set).
    No hard dependency — `opik` is imported lazily; if missing it disables + warns.
  * NEVER on the hot path. Built from the finished `rows`, so a logging failure
    can't change a single token of the measured run. The whole thing is wrapped
    so it can only ever print a warning.
  * Traces are created COMPLETE (start_time + end_time + content in one shot);
    never create-then-`.end()` — with batching that races the create against the
    content update and silently drops content (Opik's documented data-loss case).

Self-hosted config (no auth): `OPIK_URL_OVERRIDE=https://<host>/api`,
`OPIK_WORKSPACE=default`. Traces land in the `OPIK_PROJECT_NAME` project
(default `treeloom-benchmark`).
"""
from __future__ import annotations

import datetime as dt
import os

_DISABLED: bool | None = None
_CLIENT = None


def enabled() -> bool:
    """True iff tracking is requested AND the opik SDK imports. Cached; warns once
    if requested-but-unavailable so a typo'd env var fails visibly, not silently."""
    global _DISABLED
    if _DISABLED is None:
        requested = (
            os.environ.get("OPIK_TRACK", "").lower() in ("1", "true", "yes")
            or bool(os.environ.get("OPIK_URL_OVERRIDE"))
            or bool(os.environ.get("OPIK_URL"))
        )
        if not requested:
            _DISABLED = True
        else:
            try:
                import opik  # noqa: F401
                _DISABLED = False
            except Exception as e:  # SDK absent / broken env
                print(f"[opik] tracking requested but unavailable ({e}); disabled. "
                      "Install the 'opik' extra to enable.")
                _DISABLED = True
    return not _DISABLED


def _client():
    global _CLIENT
    if _CLIENT is None:
        import opik
        # Self-hosted has no auth; the SDK reads OPIK_URL_OVERRIDE/OPIK_WORKSPACE.
        os.environ.setdefault("OPIK_PROJECT_NAME", "treeloom-benchmark")
        _CLIENT = opik.Opik()
    return _CLIENT


def _usage(arm_row: dict) -> dict:
    """OpenAI-shaped usage from a per-arm result row. cached/reasoning are passed
    through (Opik retains them); they're already summed by TokenAccount.add."""
    u = {
        "prompt_tokens": int(arm_row.get("prompt_tokens") or 0),
        "completion_tokens": int(arm_row.get("completion_tokens") or 0),
        "total_tokens": int(arm_row.get("total_tokens") or 0),
    }
    for k in ("cached_input_tokens", "cache_write_tokens", "reasoning_tokens"):
        if arm_row.get(k):
            u[k] = int(arm_row[k])
    return u


def _scores(arm_row: dict) -> list[dict]:
    judge = arm_row.get("judge") or {}
    out = []
    for name, val in (
        ("correctness", judge.get("correctness")),
        ("completeness", judge.get("completeness")),
        ("recall@1", arm_row.get("recall@1")),
        ("recall@5", arm_row.get("recall@5")),
        ("mrr", arm_row.get("mrr")),
    ):
        if val is not None:
            out.append({"name": name, "value": float(val)})
    return out


def _tags(arm: str, repo_name: str, run_name: str, config: dict) -> list[str]:
    tags = [f"arm:{arm}", f"repo:{repo_name}", f"run:{run_name}"]
    # The reranker/search target is the comparison axis we most want to filter on.
    su = config.get("search_url")
    if su:
        tags.append(f"search_url:{su}")
    return tags


def log_run(
    *,
    run_name: str,
    repo_name: str,
    queries: list[dict],
    gold: dict,
    rows: list[dict],
    arms: list[str],
    config: dict,
) -> None:
    """Mirror a finished agentic run into Opik. Best-effort: any failure prints a
    warning and returns — it can never affect the benchmark result."""
    if not enabled():
        return
    try:
        _log_run_inner(run_name, repo_name, queries, gold, rows, arms, config)
    except Exception as e:  # never break a benchmark over telemetry
        print(f"[opik] log_run failed (non-fatal): {e}")


def _log_run_inner(run_name, repo_name, queries, gold, rows, arms, config) -> None:
    client = _client()
    project = os.environ.get("OPIK_PROJECT_NAME", "treeloom-benchmark")
    now = dt.datetime.now(dt.timezone.utc)

    # 1) Traces (the primary value) — one per (query, arm), created COMPLETE.
    trace_ids: dict[tuple[str, str], str] = {}
    for row in rows:
        qid = row.get("id")
        for arm in arms:
            a = (row.get("arms") or {}).get(arm)
            if a is None:
                continue
            dur = float(a.get("latency_s") or 0.0)
            start = now - dt.timedelta(seconds=dur)
            metadata = {**config, "arm": arm, "turns": a.get("turns"),
                        "tool_calls": a.get("tool_calls"), "hit_cap": a.get("hit_cap"),
                        **_usage(a)}
            tr = client.trace(
                name=f"{arm}:{qid}", project_name=project,
                input={"query": row.get("query")},
                output={"answer": a.get("final_answer")},
                start_time=start, end_time=now,
                metadata=metadata, tags=_tags(arm, repo_name, run_name, config),
                feedback_scores=_scores(a),
                error_info={"exception_type": "AgentError", "message": a["error"]}
                if a.get("error") else None,
            )
            tr.span(name="agent", type="llm", start_time=start, end_time=now,
                    input={"query": row.get("query")},
                    output={"answer": a.get("final_answer")},
                    usage=_usage(a), model=config.get("agent_model"))
            trace_ids[(qid, arm)] = tr.id
    client.flush()

    # 2) Dataset + per-arm experiments + item links (the comparison UI) — secondary.
    try:
        _log_experiments(client, repo_name, queries, gold, rows, arms, config,
                         run_name, trace_ids)
    except Exception as e:
        print(f"[opik] experiment/dataset linking failed (traces still logged): {e}")

    print(f"[opik] logged {len(trace_ids)} traces to project '{project}' "
          f"(run '{run_name}') at {os.environ.get('OPIK_URL_OVERRIDE','opik')}")


def _log_experiments(client, repo_name, queries, gold, rows, arms, config,
                     run_name, trace_ids) -> None:
    from opik.api_objects.experiment.experiment_item import ExperimentItemReferences

    ds_name = f"treeloom-bench:{repo_name}"
    dataset = client.get_or_create_dataset(name=ds_name)
    # Insert query-set items (Opik dedups by content; query id kept as a field
    # since the `id` column requires a UUID).
    dataset.insert([
        {"query_id": q["id"], "query": q["query"],
         "gold_answer": (gold.get(q["id"]) or {}).get("gold_answer", ""),
         "relevant_files": q.get("relevant_files") or q.get("ground_truth") or []}
        for q in queries
    ])
    # Map query_id -> dataset_item_id for the experiment-item references.
    item_id: dict[str, str] = {}
    for it in dataset.get_items():
        qid = it.get("query_id")
        if qid and it.get("id"):
            item_id[qid] = it["id"]

    for arm in arms:
        exp = client.create_experiment(
            dataset_name=ds_name, name=f"{run_name}:{arm}",
            experiment_config={**config, "arm": arm, "repo": repo_name},
        )
        refs = [
            ExperimentItemReferences(dataset_item_id=item_id[row["id"]],
                                     trace_id=trace_ids[(row["id"], arm)])
            for row in rows
            if row.get("id") in item_id and (row["id"], arm) in trace_ids
        ]
        if refs:
            exp.insert(refs)
