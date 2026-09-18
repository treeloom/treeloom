"""Benchmark application: LLM-paraphrase symbol-free query generation.

Deterministic strategies can't produce queries that describe an entity WITHOUT
naming it (you need paraphrase). This uses an LLM to write a behavioral query
per entity — "what does this code do?" phrased so it shares no identifier token
with the entity name — yielding a genuinely grep-resistant benchmark set. The
entity's file remains the ground truth.
"""
from __future__ import annotations

import json
import os

from treeloom.adapters.benchmark.agent_llm import AgentLLM
from treeloom.application.benchmark.enhanced_query_gen import (
    _SAMPLE_FETCH_CAP,
    DEFAULT_EXCLUDES,
    _ENTITY_TYPES,
    _assemble,
    _is_symbol_free,
    _name_tokens,
    sample_entities,
)

_SRC_CHARS = 1500

_SYSTEM = (
    "You write ONE natural-language code-search query a developer would type to "
    "FIND a specific function/class/method by WHAT IT DOES — never by its name. "
    "Rules: do NOT use the entity's name or any of its word-parts; describe its "
    "behavior, purpose, inputs/outputs; be specific enough that this code is the "
    "answer (not a generic question). Output only the question, no quotes, no "
    "preamble. /no_think"
)


def _read_source(entity: dict) -> str:
    fp = entity.get("file_path", "")
    sl = entity.get("start_line") or 1
    el = entity.get("end_line") or (sl + 40)
    if not fp or not os.path.isfile(fp):
        return ""
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[max(0, sl - 1): el])[:_SRC_CHARS]


def _looks_truncated(q: str) -> bool:
    """True when the model's answer was cut off by the token budget.

    A truncated query is an UNANSWERABLE benchmark row -- e.g. "Given a rule
    identifier and two versions of" -- and silently poisons a query set the
    same way non-discriminating names do. Cheap structural test: a finished
    question ends in terminal punctuation.
    """
    q = (q or "").strip()
    if not q:
        return True
    return q[-1] not in ".?!"


async def _behavioral_query(llm: AgentLLM, entity: dict, src: str, *, strict: bool) -> str:
    tokens = sorted(_name_tokens(entity.get("name", "")))
    forbidden = f"\nForbidden words (do not use any of these): {tokens}" if tokens else ""
    if strict:
        forbidden += "\nYour previous attempt used a forbidden word. Rephrase using only behavior."
    user = (
        f"Entity type: {entity.get('type', '')}\n"
        f"Signature: {entity.get('signature', '') or '(none)'}\n"
        f"Code:\n{src}{forbidden}"
    )
    content, _, _ = await llm.chat(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        max_tokens=400,
    )
    from treeloom.domain.benchmark.agent_protocol import strip_thinking

    return strip_thinking(content).strip().strip('"').strip()


def _load_checkpoint(path: str | None) -> tuple[list[dict], set[str]]:
    """Rows already produced by an interrupted run, and their entity ids.

    Generation is sequential and one entity costs an LLM round trip, so a
    3,500-entity run is ~an hour of paid calls held in memory until the very
    end. A process-level kill used to lose all of it.

    A run killed mid-write leaves a TORN final line; skip it rather than
    letting a resume die on the wreckage of the crash it exists to recover
    from.
    """
    rows: list[dict] = []
    done: set[str] = set()
    if not path or not os.path.exists(path):
        return rows, done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            eid = (row.get("entity") or {}).get("id")
            if eid:
                done.add(eid)
    return rows, done

async def generate_symbol_free_queries(
    *,
    repo: str,
    llm: AgentLLM,
    source_id: str | None = None,
    path_prefix: str | None = None,
    exclude: list[str] | None = None,
    max_entities: int = 100,
    sample_seed: int | None = None,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """One LLM-paraphrased, symbol-free query per entity (validated)."""
    from treeloom import graph_store

    entities = await graph_store.list_entities(
        _ENTITY_TYPES, source_id=source_id, path_prefix=path_prefix,
        exclude=exclude if exclude is not None else DEFAULT_EXCLUDES,
        limit=_SAMPLE_FETCH_CAP if sample_seed is not None else max_entities,
    )
    if sample_seed is not None:
        entities = sample_entities(entities, max_entities, sample_seed)
    rows, _done = _load_checkpoint(checkpoint_path)
    if rows:
        print(f"  llm-symbol-free: resuming from {len(rows)} checkpointed row(s)")
    n = len(rows)
    resumed = len(rows)
    skipped = 0
    leaked = 0
    errored = 0
    truncated = 0
    ckpt = open(checkpoint_path, "a") if checkpoint_path else None
    try:
      for ent in entities:
        if ent.get("id") in _done:
            skipped += 1
            continue
        src = _read_source(ent)
        if not src:
            continue
        # A transient LLM error (e.g. httpx.ReadError on the streamed response)
        # must skip this entity, not abort the whole multi-hundred-entity run.
        try:
            q = await _behavioral_query(llm, ent, src, strict=False)
            if q and not _is_symbol_free(q, ent.get("name", "")):
                q = await _behavioral_query(llm, ent, src, strict=True)  # one retry
        except Exception:
            errored += 1
            continue
        if not q or not _is_symbol_free(q, ent.get("name", "")):
            leaked += 1
            continue
        if _looks_truncated(q):
            truncated += 1
            continue
        n += 1
        variant = {
            "query": q,
            "relevant_files": [ent.get("file_path", "")],
            "strategy": "llm_symbol_free",
            "difficulty": "hard",
        }
        row = _assemble(variant, ent, repo, n)
        rows.append(row)
        if ckpt is not None:
            # Flush per row: the whole point is surviving a kill, and an
            # unflushed buffer is exactly what a kill discards.
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
    finally:
        if ckpt is not None:
            ckpt.close()
    print(f"  llm-symbol-free: kept {len(rows)}, dropped {leaked} (couldn't avoid the symbol)"
          f", truncated {truncated} (cut off by the token budget)"
          f", errored {errored} (transient LLM failures, skipped)"
          + (f", resumed {resumed} + skipped {skipped} already-done entities"
             if resumed or skipped else ""))
    return rows
