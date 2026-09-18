"""Sufficiency-proxy screen.

A cheap pre-screen for response-shaping configs (verbosity tiers, response_modes)
that does NOT run a full agentic loop. Per query, ONE LLM call decides whether
the search payload is self-sufficient to answer, or whether the agent would need
to read more files (NEED_MORE). That directly captures the signal that sank
summary_tail — terse summaries trigger compensatory reads — at ~1–2 LLM
calls/query instead of a 12-turn agentic loop (minutes, not an hour).

Per config it reports:
- need_more_rate  — fraction of queries the payload was NOT self-sufficient for
                    (the compensatory-fetch proxy; lower is better)
- mean_correctness_answered — judge correctness on the queries it DID answer
- mean_payload_tokens — tiktoken size of the rendered search response (the cut)

The funnel: screen many (verbosity × model) combos here, then run a
full agentic cell only on the 1–2 Pareto survivors.
"""
from __future__ import annotations

import asyncio

import httpx
import tiktoken

from treeloom.adapters.benchmark.agent_llm import AgentLLM
from treeloom.application.benchmark.gold import build_gold
from treeloom.application.benchmark.judge import judge_answer
from treeloom.application.mcp_server import _search_response_to_markdown
from treeloom.domain.benchmark.agent_protocol import strip_thinking

_enc = tiktoken.get_encoding("cl100k_base")

_SCREEN_SYSTEM = (
    "You answer a question about a codebase using ONLY the provided search "
    "results. If they contain enough to answer correctly, give a concise answer. "
    "If you would need to open or read additional files to answer, reply with "
    "exactly NEED_MORE and nothing else. Do not guess."
)


async def _screen_one(q, gold_rec, *, search_url, path_prefix, llm, judge_llm, body_extra):
    body = {"query": q["query"], "top_k": 5, "path_prefix": path_prefix}
    body.update(body_extra)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{search_url.rstrip('/')}/search", json=body)
        resp.raise_for_status()
        data = resp.json()
    rendered = _search_response_to_markdown(data)
    payload_tokens = len(_enc.encode(rendered))
    content, _usage, _tc = await llm.chat(
        [
            {"role": "system", "content": _SCREEN_SYSTEM},
            {"role": "user", "content": f"Question: {q['query']}\n\nSearch results:\n{rendered}"},
        ],
        max_tokens=400,
    )
    ans = strip_thinking(content or "").strip()
    need_more = "NEED_MORE" in ans.upper()[:24]
    if need_more:
        return {"need_more": True, "payload_tokens": payload_tokens, "correctness": None}
    js = await judge_answer(q["query"], gold_rec.get("gold_answer", ""), ans, judge_llm)
    return {"need_more": False, "payload_tokens": payload_tokens, "correctness": js.correctness}


async def run_screen(
    queries, repo_name, *, search_url, path_prefix, configs,
    model=None, judge_model=None, regen_gold=False, concurrency=8,
):
    """`configs` = [{"name": str, "body": dict}] — each `body` merges into the
    /search request (e.g. {"response_mode": "summary_tail"}). `concurrency` caps
    in-flight queries so we don't overwhelm the HyDE/reranker backends (40
    concurrent /search ReadTimeouts them)."""
    llm = AgentLLM(model=model) if model else AgentLLM()
    judge_llm = AgentLLM(model=judge_model) if judge_model else llm
    gold = await build_gold(queries, repo_name, judge_llm, regen=regen_gold)
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(q, body_extra):
        async with sem:
            # A transient LLM/search error on one query must not abort the whole
            # gather — record it as an error row and keep going.
            try:
                return await _screen_one(
                    q, gold.get(q["id"], {}), search_url=search_url,
                    path_prefix=path_prefix, llm=llm, judge_llm=judge_llm,
                    body_extra=body_extra,
                )
            except Exception as exc:
                return {"error": str(exc)}

    out = {}
    for cfg in configs:
        rows = await asyncio.gather(*[_bounded(q, cfg["body"]) for q in queries])
        ok = [r for r in rows if "error" not in r]
        errors = len(rows) - len(ok)
        nm = sum(1 for r in ok if r["need_more"])
        corrs = [r["correctness"] for r in ok if r["correctness"] is not None]
        out[cfg["name"]] = {
            "need_more_rate": round(nm / len(ok), 3) if ok else None,
            "mean_correctness_answered": round(sum(corrs) / len(corrs), 3) if corrs else 0.0,
            "answered": len(corrs),
            "mean_payload_tokens": round(sum(r["payload_tokens"] for r in ok) / len(ok), 1) if ok else 0.0,
            "scored": len(ok),
            "errors": errors,
        }
    return out
