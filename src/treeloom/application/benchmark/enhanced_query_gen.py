"""Benchmark application: enhanced (strategy-based) query generation.

Re-wires the deterministic prompt-enhancer (`domain/prompt_enhancer.py`) to the
indexed code graph: pull Class/Function/Method entities from Neo4j, run each
through the 6 enhancer strategies, and emit the rich benchmark-query schema
(id, query, relevant_files, entity, search_strategy, strategy, difficulty).

Ground truth is entity-anchored: each query's `relevant_files` is the entity's
defining file (the `Enhancer` sets this), so no human labeling is needed.
"""
from __future__ import annotations

import os
import random
import re

from treeloom.domain.prompt_enhancer import ALL_STRATEGIES, Enhancer, PromptEnhancerConfig

_ENTITY_TYPES = ["Class", "Function", "Method"]

# Entity fetch cap when sampling: large enough to span any indexed repo so the
# seeded sample sees the whole population, small enough to stay a cheap read.
_SAMPLE_FETCH_CAP = 50_000


def sample_entities(entities: list[dict], k: int, seed: int) -> list[dict]:
    """Seeded uniform sample of k entities, in seeded RANDOM order.

    `list_entities` returns entities ORDER BY file_path — a bare LIMIT slice
    therefore clusters in the alphabetically-first subtree (on one large
    TypeScript repo, 200/16k entities were all under cli/). Sampling over the
    full population
    spreads ground truth across the repo; the seed keeps generation
    reproducible for a release bundle.

    The ORDER matters as much as the set. This used to `sorted()` the picked
    indices, restoring file_path order, so generation swept the tree
    alphabetically and any PREFIX of a run was a biased subset. On featbit
    v5.4.9 a symbol-free run stopped at 64% would have held zero TypeScript
    queries: all 1,018 front-end entities sorted after the C# back-end. With
    generation checkpointed, stopping early is a real option -- so every
    prefix must itself be a uniform sample.

    `rng.sample` already yields its picks in random order; the sort was the
    only thing undoing that. Dropping it keeps the identical draw, so a
    COMPLETE run covers exactly the same entities as before -- only the order,
    and therefore the sequential query ids, change.
    """
    rng = random.Random(seed)
    if len(entities) <= k:
        order = list(range(len(entities)))
        rng.shuffle(order)
        return [entities[i] for i in order]
    return [entities[i] for i in rng.sample(range(len(entities)), k)]


def _name_tokens(name: str) -> set[str]:
    """Identifier tokens from an entity name (camelCase + snake_case split)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    parts = re.split(r"[^A-Za-z0-9]+", spaced)
    return {p.lower() for p in parts if len(p) >= 3}


def _is_symbol_free(query: str, name: str) -> bool:
    """True if the query shares no identifier token with the entity name.

    Symbol-free queries can't be answered by simply grepping the symbol, so
    they actually stress semantic retrieval instead of favoring grep.
    """
    q = (query or "").lower()
    if name and name.lower() in q:
        return False
    return not any(re.search(rf"\b{re.escape(tok)}\b", q) for tok in _name_tokens(name))

# Path substrings excluded by default so generated queries target core code,
# not docs/tests/examples/vendored deps. Override or disable via --exclude.
DEFAULT_EXCLUDES = [
    "/docs/", "/doc/", "/docs_src/", "/test/", "/tests/", "/__tests__/",
    "/examples/", "/example/", "/samples/", "/vendor/", "/third_party/",
    "/node_modules/", "/site-packages/", "/.venv/",
    # Generated/minified build artifacts: anchoring ground truth to a bundle
    # like build/three.core.js makes a query unanswerable (no agent reads a
    # minified bundle), which silently zeroes recall for every arm.
    "/build/", "/dist/", ".min.",
]


def _grep_patterns(name: str) -> list[str]:
    """The entity identifier is the highest-signal grep pattern."""
    return [name] if name else []


def _glob_patterns(file_path: str) -> list[str]:
    """Directory-scoped globs by the entity's file extension."""
    ext = os.path.splitext(file_path)[1] or ""
    d = os.path.dirname(file_path)
    if not ext or not d:
        return []
    return [f"{d}/*{ext}", f"{d}/**/*{ext}"]


def _assemble(variant: dict, entity: dict, repo: str, n: int) -> dict:
    fp = entity.get("file_path", "")
    name = entity.get("name", "")
    return {
        "id": f"{repo}-{n:04d}",
        "query": variant["query"],
        "relevant_files": variant.get("relevant_files", [fp]),
        "additional_relevant_files": variant.get("additional_relevant_files", []),
        "repo": repo,
        "entity": {
            "id": entity.get("id", ""),
            "name": name,
            "type": entity.get("type", ""),
        },
        "search_strategy": {
            "grep_patterns": _grep_patterns(name),
            "glob_patterns": _glob_patterns(fp),
            "max_read_lines": 200,
        },
        "strategy": variant.get("strategy", ""),
        "difficulty": variant.get("difficulty", ""),
    }


def resolve_strategies(spec: str) -> list[str]:
    """Resolve a --strategy spec ('all' or csv) to a validated strategy list."""
    if not spec or spec == "all":
        return list(ALL_STRATEGIES)
    chosen = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [s for s in chosen if s not in ALL_STRATEGIES]
    if bad:
        raise ValueError(
            f"unknown strategies {bad}; valid: {', '.join(ALL_STRATEGIES)}"
        )
    return chosen


async def available_repo_prefixes(limit: int = 500) -> list[tuple[str, int]]:
    """Distinct indexed repo prefixes (first 3 path segments) with entity counts,
    most-populated first. Used to suggest valid targets when a prefix matches
    nothing."""
    from treeloom import graph_store

    async with graph_store._get_session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE e.file_path STARTS WITH '/'
            WITH split(e.file_path, '/') AS parts WHERE size(parts) >= 4
            WITH '/' + parts[1] + '/' + parts[2] + '/' + parts[3] AS prefix, count(*) AS c
            RETURN prefix, c ORDER BY c DESC LIMIT $limit
            """,
            limit=limit,
        )
        records = await result.fetch(limit)
        return [(r["prefix"], r["c"]) for r in records]


async def generate_enhanced_queries(
    *,
    repo: str,
    strategies: list[str],
    source_id: str | None = None,
    path_prefix: str | None = None,
    exclude: list[str] | None = None,
    symbol_free: bool = False,
    max_entities: int = 200,
    max_per_entity: int = 3,
    sample_seed: int | None = None,
) -> tuple[list[dict], int]:
    """Pull entities from the graph and expand them into enhanced queries.

    When `symbol_free` is set, only queries that share no identifier token with
    their target entity name are kept (so grep can't trivially win by grepping
    the symbol). In practice this keeps behavioral/docstring-style queries.

    Returns ``(rows, dropped)`` where ``dropped`` is the number of generated
    queries the symbol-free filter removed (0 when ``symbol_free`` is off). The
    caller uses ``len(rows) + dropped`` as the pre-filter total to detect the
    low-yield case that should steer toward ``--llm-symbol-free``.
    """
    from treeloom import graph_store

    entities = await graph_store.list_entities(
        _ENTITY_TYPES, source_id=source_id, path_prefix=path_prefix,
        exclude=exclude,
        limit=_SAMPLE_FETCH_CAP if sample_seed is not None else max_entities,
    )
    if sample_seed is not None:
        entities = sample_entities(entities, max_entities, sample_seed)
    config = PromptEnhancerConfig(
        strategies=strategies, max_queries_per_entity=max_per_entity
    )
    enhancer = Enhancer(config, graph_store_module=graph_store)

    rows: list[dict] = []
    dropped = 0
    n = 0
    for ent in entities:
        try:
            variants = await enhancer.enhance(ent)
        except Exception:
            continue
        for v in variants:
            if not v.get("query"):
                continue
            if symbol_free and not _is_symbol_free(v["query"], ent.get("name", "")):
                dropped += 1
                continue
            n += 1
            rows.append(_assemble(v, ent, repo, n))
    if symbol_free:
        print(f"  symbol-free filter: kept {len(rows)}, dropped {dropped} "
              "(queries containing the target's identifier)")
    return rows, dropped
