"""Benchmark application: gold (reference) answer generation + disk cache.

Generated ONCE per query from the ground-truth source context (relevant_files
focused on the target entity), cached to benchmarks/gold/<repo>.jsonl, and
reused across runs. The judge later scores each arm's answer against the gold
answer. Cache is plain JSONL so it can be reviewed/edited by hand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from treeloom.adapters.benchmark.agent_llm import AgentLLM

GOLD_DIR = Path("benchmarks/gold")
_PER_FILE_CHARS = 6000
_TOTAL_CHARS = 12000

_GOLD_SYSTEM = (
    "You are writing a reference (gold) answer for a code-search benchmark. "
    "Given the exact ground-truth source context, write the ideal answer to the "
    "user's question: name the file(s), name the target entity, and explain what "
    "it does and how. Be specific and correct. 4-8 sentences. Output prose only — "
    "no markdown, no code fences. /no_think"
)

_GOLD_USER = (
    "Question: {query}\n"
    "Target entity: {entity}\n\n"
    "Ground-truth source context:\n{context}"
)


def gold_path(repo: str) -> Path:
    return GOLD_DIR / f"{repo}.jsonl"


def _load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["id"]] = rec
    return cache


def _write_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in cache.values():
            f.write(json.dumps(rec) + "\n")


def _build_context(query: dict) -> str:
    """Read relevant files (focused near the entity line when possible)."""
    files = list(query.get("relevant_files", [])) + list(
        query.get("additional_relevant_files", [])
    )
    parts: list[str] = []
    used = 0
    entity_name = (query.get("entity") or {}).get("name", "")
    for fp in files:
        if used >= _TOTAL_CHARS:
            break
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        # Focus near the entity definition if we can find it; else file head.
        snippet = text
        if entity_name and entity_name in text and len(text) > _PER_FILE_CHARS:
            idx = text.index(entity_name)
            start = max(0, idx - _PER_FILE_CHARS // 3)
            snippet = text[start : start + _PER_FILE_CHARS]
        else:
            snippet = text[:_PER_FILE_CHARS]
        budget = min(_PER_FILE_CHARS, _TOTAL_CHARS - used)
        snippet = snippet[:budget]
        parts.append(f"=== FILE: {fp} ===\n{snippet}")
        used += len(snippet)
    return "\n\n".join(parts)


async def build_gold(
    queries: list[dict],
    repo: str,
    llm: AgentLLM | None = None,
    *,
    regen: bool = False,
) -> dict[str, dict]:
    """Build/refresh gold answers for `queries`, returning {id: gold_record}."""
    path = gold_path(repo)
    cache = {} if regen else _load_cache(path)
    answerer = llm or AgentLLM(max_tokens=512)

    for q in queries:
        qid = q["id"]
        if qid in cache and not regen and cache[qid].get("gold_answer"):
            continue
        context = _build_context(q)
        if not context.strip():
            cache[qid] = {
                "id": qid,
                "query": q["query"],
                "gold_answer": "",
                "relevant_files": q.get("relevant_files", []),
                "entity": q.get("entity", {}),
                "error": "no readable ground-truth files",
            }
            continue
        content, _, _ = await answerer.chat(
            [
                {"role": "system", "content": _GOLD_SYSTEM},
                {
                    "role": "user",
                    "content": _GOLD_USER.format(
                        query=q["query"],
                        entity=q.get("entity", {}),
                        context=context,
                    ),
                },
            ],
            max_tokens=512,
        )
        from treeloom.domain.benchmark.agent_protocol import strip_thinking

        cache[qid] = {
            "id": qid,
            "query": q["query"],
            "gold_answer": strip_thinking(content),
            "relevant_files": q.get("relevant_files", []),
            "entity": q.get("entity", {}),
        }

    _write_cache(path, cache)
    return cache
