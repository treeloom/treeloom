"""Store-agnostic retrieval orchestration.

Everything here used to live in `adapters/milvus/vector_store.py` but is
vector-store-neutral: it consumes Milvus-SHAPED hit dicts
(`{id, distance, entity: {chunk_text, file_path, ...}}`) from whichever
store `treeloom.retriever` dispatches to (VECTOR_STORE env), reranks them,
fuses graph signals, and emits the ChunkHit wire contract. The milvus
adapter re-imports these names for backward compatibility.
"""
import json
import logging
import math
import os
import re

from treeloom import community, embedder, graph_store, llm
from treeloom import reranker as _reranker
from treeloom.domain.shared import chunk_hit_from_milvus, make_hit_id

logger = logging.getLogger(__name__)

TOP_K_PRE_RERANK = int(os.environ.get("MILVUS_TOP_K_PRE_RERANK", "50"))
MILVUS_TOP_K_PRE_RERANK = TOP_K_PRE_RERANK  # legacy alias (pre-extraction name)
USE_HYBRID = os.environ.get("USE_HYBRID", "1") == "1"
USE_HYDE = os.environ.get("USE_HYDE", "1") == "1"
USE_SUMMARY_VECTOR = os.environ.get("USE_SUMMARY_VECTOR", "1") == "1"
# Graph RESCORING default. The graph layer contributes value in
# two independent ways: (1) the neighbor/community PAYLOAD fed to the agent
# (always on; not governed by this flag) and (2) post-rerank RESCORING that
# fuses community-cosine + centrality + name signals into final_score via
# GRAPH_ALPHA/BETA/GAMMA/DELTA. Ablation D3 showed (2) adds ~0 over a
# STRONG reranker (Cohere rerank-v4.0-pro) while (1) carries the value — so
# payload-without-rescoring (`use_graph_scoring=false`) is a first-class mode.
# BUT a later retrieval re-validation under the *production* reranker
# (Qwen3-Reranker-0.6B) found rescoring still materially helps rank-1 there —
# guava symbol-free recall@1 0.16→0.49, MRR 0.448→0.621 — because a weaker
# reranker leaves room for graph symbol-promotion to fix rank-1. So rescoring
# stays ON by default; flip to off only with a Cohere-class reranker. The
# PAYLOAD is unaffected either way: graph_search always fetches
# neighbors/community summaries; only the rescore branch is gated by this flag.
USE_GRAPH_SCORING = os.environ.get("USE_GRAPH_SCORING", "1") == "1"
GRAPH_ALPHA = float(os.environ.get("GRAPH_ALPHA", "1.0"))
GRAPH_BETA = float(os.environ.get("GRAPH_BETA", "0.3"))
GRAPH_GAMMA = float(os.environ.get("GRAPH_GAMMA", "0.2"))
GRAPH_DELTA = float(os.environ.get("GRAPH_DELTA", "0.4"))
# The reranker returns raw (unbounded) cross-encoder logits, while the graph
# signals (comm/cent/name) live in [0,1]. Summing them directly lets the raw
# logit dominate and swamps every graph signal — so min-max normalize the
# rerank scores to [0,1] within the candidate set before fusion, putting all
# four terms on the same scale. Toggle off to restore the raw-logit behavior.
RERANK_NORMALIZE = os.environ.get("RERANK_NORMALIZE", "1") == "1"
# When on, graph_rescore logs the per-signal contribution breakdown + how much
# the graph signals reordered vs the rerank-only order, for one query at a time.
GRAPH_RESCORE_DEBUG = os.environ.get("GRAPH_RESCORE_DEBUG", "0") == "1"


def _store():
    """The active vector-store module (env-dispatched in treeloom.retriever).

    Imported lazily per call: `treeloom.retriever` imports this module's
    search entry points, so a module-level import would be circular.
    """
    import treeloom.retriever as r

    return r


async def rerank(
    query: str,
    results: list[dict],
    top_k: int = 5,
    batch_size: int | None = None,
) -> list[dict]:
    if not results:
        return []
    texts = [r["entity"]["chunk_text"] for r in results]
    pairs = await _reranker.rerank_texts(query, texts, batch_size=batch_size)
    scored: list[tuple[float, dict]] = []
    for idx, score in pairs:
        # `idx` comes back from the reranker — for the cloud providers that is
        # a remote service's response. Python accepts a negative index
        # silently, so -1 would score the LAST chunk as if it were the first
        # and quietly corrupt the ranking; an over-range one raises IndexError
        # and fails the whole search. Neither belongs in a hot path fed by an
        # external system (CWE-129).
        if not isinstance(idx, int) or not (0 <= idx < len(results)):
            logger.warning(
                "reranker returned out-of-range index %r for %d candidates; "
                "skipping that score",
                idx, len(results),
            )
            continue
        results[idx]["relevance_score"] = score
        scored.append((score, results[idx]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


async def _maybe_hyde_embedding(
    query_text: str, language: str | None, use_hyde: bool
) -> list[float] | None:
    if not use_hyde:
        return None
    hyde_text = await llm.generate_hyde(query_text, language=language)
    if not hyde_text:
        return None
    embs = await embedder.embed([hyde_text])
    return embs[0] if embs else None


_community_emb_cache: dict[int, list[float]] | None = None


async def _get_community_embeddings() -> dict[int, list[float]]:
    global _community_emb_cache
    if _community_emb_cache is None:
        try:
            _community_emb_cache = await community.load_community_embeddings()
        except Exception:
            _community_emb_cache = {}
    return _community_emb_cache


def invalidate_graph_caches():
    global _community_emb_cache
    _community_emb_cache = None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Identifier length at which a name-in-query match earns the full signal.
# Longer = more specific: a file defining `set_physics_ticks_per_second` should
# outweigh one that merely defines a short, common name like `Engine` (which
# appears across headers/callers/tests). This is the recall@1 lever — promote
# the file that DEFINES the specific symbol over files that just reference it.
_NAME_FULL_WEIGHT_LEN = float(os.environ.get("GRAPH_NAME_FULL_WEIGHT_LEN", "16"))

# Extra file-level boost for a file that DEFINES (Function/Class/…) a symbol
# whose name is in the query — added on top of the general name term so the
# definition file outranks files that merely match a name (e.g. a filename
# `Module` match). 0.0 = off (default; identical to pre-boost). A/B via env.
# NB: the per-CHUNK def/ref split was reverted — on the FILE-level
# recall@1 metric it demoted reference chunks of the *correct* file and let
# wrong-file chunks win rank 1 (won 0 / lost 2 on a large C++ repo). This boost works
# at the right granularity: it promotes the whole defining file.
GRAPH_EPSILON = float(os.environ.get("GRAPH_EPSILON", "0.0"))
# Entity types that count as a real definition for the file-level boost. A
# whole-file `Module` (name = filename) is excluded — it's a filename match,
# not a symbol definition.
_DEFINE_TYPES = frozenset({"Function", "Class", "Method", "Interface", "Struct", "Enum"})


def _name_in_query(query_text: str, entities: list[dict]) -> float:
    """Graded file-level name-in-query signal in [0, 1].

    Max specificity weight over the file's entities whose name appears in the
    query; weight scales with identifier length (capped at 1.0) so a long,
    specific symbol outweighs a short common one. `entities_by_file` already
    holds only entities DEFINED in the file, so this is inherently a file-level
    "defines a symbol named in the query" signal.
    """
    q = query_text.lower()
    best = 0.0
    for e in entities:
        n = (e.get("name") or "").lower()
        if n and len(n) >= 3 and n in q:
            w = min(len(n) / _NAME_FULL_WEIGHT_LEN, 1.0)
            if w > best:
                best = w
    return best


def _file_defines_query_symbol(query_text: str, entities: list[dict]) -> float:
    """Graded [0,1]: does the file DEFINE a Function/Class/… named in the query?

    Like `_name_in_query` but restricted to real definition types (excludes the
    whole-file `Module`/filename match). Used as a separate, additive file-level
    boost so a true definition file is promoted above mere name matches.
    """
    q = query_text.lower()
    best = 0.0
    for e in entities:
        if e.get("type") not in _DEFINE_TYPES:
            continue
        n = (e.get("name") or "").lower()
        if n and len(n) >= 3 and n in q:
            w = min(len(n) / _NAME_FULL_WEIGHT_LEN, 1.0)
            if w > best:
                best = w
    return best


def _minmax(values: list[float]) -> dict[int, float]:
    """Map each value to [0,1] by min-max over the set (index-keyed).

    All-equal (incl. single-element) sets carry no discriminative signal, so
    every element maps to 1.0 — a constant that cancels out of the ranking and
    lets the graph signals fully decide the order.
    """
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return {i: 1.0 for i in range(len(values))}
    return {i: (v - lo) / span for i, v in enumerate(values)}


async def graph_rescore(
    reranked: list[dict],
    query_embedding: list[float],
    query_text: str,
    entities_by_file: dict[str, list[dict]],
    community_embs: dict[int, list[float]],
) -> list[dict]:
    # Normalize the raw rerank logits to [0,1] within this candidate set so the
    # fused score isn't dominated by the unbounded logit scale (see
    # RERANK_NORMALIZE note). With normalization off, rel stays the raw logit.
    raw_rels = [float(r.get("relevance_score", 0.0)) for r in reranked]
    norm_map = _minmax(raw_rels) if RERANK_NORMALIZE else {}

    out = []
    for i, r in enumerate(reranked):
        raw_rel = raw_rels[i]
        rel = norm_map.get(i, raw_rel) if RERANK_NORMALIZE else raw_rel
        fp = r["entity"].get("file_path", "")
        ents = entities_by_file.get(fp, [])
        cids = {e["community_id"] for e in ents if e.get("community_id") is not None}
        comm = max(
            (_cosine(query_embedding, community_embs[c]) for c in cids if c in community_embs),
            default=0.0,
        )
        cent = max((float(e.get("centrality") or 0.0) for e in ents), default=0.0)
        name = _name_in_query(query_text, ents)
        file_def = _file_defines_query_symbol(query_text, ents) if GRAPH_EPSILON else 0.0
        r["graph_signals"] = {
            "community": comm, "centrality": cent, "name": name,
            "file_def": file_def, "rel_raw": raw_rel, "rel_norm": rel,
        }
        r["final_score"] = (
            GRAPH_ALPHA * rel
            + GRAPH_BETA * comm
            + GRAPH_GAMMA * cent
            + GRAPH_DELTA * name
            + GRAPH_EPSILON * file_def
        )
        out.append(r)

    if GRAPH_RESCORE_DEBUG:
        _log_rescore_debug(query_text, out)

    out.sort(key=lambda x: x["final_score"], reverse=True)
    return out


def _log_rescore_debug(query_text: str, scored: list[dict], top: int = 8) -> None:
    """Log per-signal contributions + how far graph signals moved each candidate.

    `scored` is in rerank order (pre final-sort). We compute the post-fusion
    order and report, for the top candidates, the raw vs normalized rel, each
    graph signal, the weighted contribution of each term, and the rank delta
    (rerank rank -> fused rank) so it's obvious whether graph signals reorder
    anything and which term did the work.
    """
    rerank_rank = {id(r): i for i, r in enumerate(scored)}
    fused = sorted(scored, key=lambda x: x["final_score"], reverse=True)
    moved = sum(1 for fr, r in enumerate(fused) if rerank_rank[id(r)] != fr)
    logger.info(
        "graph_rescore[norm=%s] q=%r n=%d reordered=%d/%d  weights a/b/g/d=%.2f/%.2f/%.2f/%.2f",
        RERANK_NORMALIZE, query_text[:80], len(scored), moved, len(scored),
        GRAPH_ALPHA, GRAPH_BETA, GRAPH_GAMMA, GRAPH_DELTA,
    )
    for fr, r in enumerate(fused[:top]):
        s = r["graph_signals"]
        fp = r["entity"].get("file_path", "")
        name = fp.rsplit("/", 1)[-1] if fp else "?"
        logger.info(
            "  #%d (was #%d) %-28s final=%.3f | rel raw=%+.3f norm=%.3f "
            "(a=%.3f) name=%.2f(d=%.3f) comm=%.3f(b=%.3f) cent=%.3f(g=%.3f)",
            fr, rerank_rank[id(r)], name[-28:], r["final_score"],
            s["rel_raw"], s["rel_norm"], GRAPH_ALPHA * s["rel_norm"],
            s["name"], GRAPH_DELTA * s["name"],
            s["community"], GRAPH_BETA * s["community"],
            s["centrality"], GRAPH_GAMMA * s["centrality"],
        )


# Exact-symbol definition short-circuit (the recall@1 lever, attacked
# deterministically). When the query contains a code-shaped identifier and the
# graph knows exactly where it's defined, the defining file's chunk is moved
# to rank 1 outright — no fusion weight to tune, no way for a verbose README
# chunk to outscore the definition. Benchmarked driver: on symbol-containing
# queries quality ties grep but recall@1 0.45 vs 0.90 costs 2.66x tokens in
# extra search/read rounds (docs/benchmark-findings.md #6).
SYMBOL_PROMOTE = os.environ.get("SYMBOL_PROMOTE", "1") == "1"
# camelCase / PascalCase (>=2 humps) / snake_case tokens — plain words don't
# trigger. Minimum 5 chars so short generics like `getX` stay out.
_IDENTIFIER_RE = re.compile(
    r"\b(?:"
    r"[a-z][a-z0-9]*_[a-z0-9_]+"          # snake_case
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"  # PascalCase
    r"|[a-z]+(?:[A-Z][a-z0-9]*)+"          # camelCase
    r")\b"
)


def _query_symbols(query_text: str, limit: int = 3) -> list[str]:
    """Code-shaped identifiers from the query, longest (most specific) first."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _IDENTIFIER_RE.finditer(query_text):
        tok = m.group(0)
        if len(tok) >= 5 and tok not in seen:
            seen.add(tok)
            out.append(tok)
    out.sort(key=len, reverse=True)
    return out[:limit]


# Query-class-dependent payload. On a SYMBOL-BEARING query the
# right file is already promoted to rank 1 by _promote_definition, so the
# neighbor/community aux payload is low-value there — drop it. SYMBOL-FREE
# queries keep the rich payload (that's where treeloom's value lives, D1).
# Opt-in, default off; the default flip is benchmark-gated.
QUERY_CLASS_PAYLOAD = os.environ.get("QUERY_CLASS_PAYLOAD", "0") == "1"


def _query_class_payload(query_text, neighbors, summaries, enabled):
    """Drop neighbor/community payload for symbol-bearing queries when enabled.

    Pure: returns the (possibly emptied) (neighbors, summaries). Symbol-free
    queries (no code identifier ≥5 chars in the text) keep the full payload.
    """
    if enabled and _query_symbols(query_text):
        return [], {}
    return neighbors, summaries


def _display_score(r: dict) -> float:
    """The score chunk_hit_from_milvus will surface (same precedence)."""
    if "final_score" in r:
        return float(r.get("final_score") or 0)
    return float(r.get("relevance_score", r.get("distance", 0)) or 0)


async def _promote_definition(
    reranked: list[dict],
    vec_results: list[dict],
    query_text: str,
    source_id: str | None,
    path_prefix: str | None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Move the chunk from a query-symbol's defining file to rank 1.

    For each extracted identifier (most specific first): look up its defining
    entities in Neo4j (index-backed name lookup, scoped like the search). If a
    defining file is already among the ranked chunks, its best-ranked chunk
    moves to the front. If not, but exactly one defining file exists and it
    has a chunk in the pre-rerank pool, that chunk replaces the last result
    (ambiguous multi-file definitions are left to the ranker). At most one
    promotion per query.
    """
    for sym in _query_symbols(query_text):
        ents = await graph_store.find_entities_by_name(
            sym, source_id=source_id, exclude_source_ids=exclude_source_ids,
        )
        def_files = sorted({
            e["file_path"] for e in ents
            if e.get("type") in _DEFINE_TYPES and e.get("file_path")
        })
        if path_prefix:
            def_files = [f for f in def_files if f.startswith(path_prefix)]
        if not def_files:
            continue
        for i, r in enumerate(reranked):
            if r["entity"].get("file_path") in def_files:
                if i > 0:
                    reranked.insert(0, reranked.pop(i))
                    reranked[0]["final_score"] = round(
                        max(_display_score(x) for x in reranked) + 0.001, 3
                    )
                return reranked
        if len(def_files) == 1:
            for r in vec_results:
                if r["entity"].get("file_path") == def_files[0]:
                    r["final_score"] = round(
                        max(_display_score(x) for x in reranked) + 0.001, 3
                    )
                    reranked.insert(0, r)
                    reranked.pop()
                    return reranked
    return reranked


# Response token budget. A hard ceiling on the serialized search payload,
# filled in priority order — chunks (rank order, top hit always kept), then
# neighbors, then community summaries — dropping WHOLE items when the budget
# runs out. Never truncates inside a snippet: the lean-mode benchmark showed
# a partial result that LOOKS complete starves the agent into extra rounds,
# while a missing tail item is just a shorter list. Sized as a guardrail for
# pathological payloads (huge files x top_k=20), not a diet for normal ones:
# a default k=5 featbit response runs ~2.5k tokens. 0 disables.
SEARCH_RESPONSE_MAX_TOKENS = int(os.environ.get("SEARCH_RESPONSE_MAX_TOKENS", "8000"))


def _apply_response_budget(
    chunks: list[dict], neighbors: list[dict], summaries: dict[str, str]
) -> tuple[list[dict], list[dict], dict[str, str]]:
    if SEARCH_RESPONSE_MAX_TOKENS <= 0:
        return chunks, neighbors, summaries
    from treeloom.domain.token_budget import TokenBudget, TokenBudgetConfig

    budget = TokenBudget(TokenBudgetConfig(total_tokens=SEARCH_RESPONSE_MAX_TOKENS))

    kept_chunks: list[dict] = []
    for i, c in enumerate(chunks):
        text = json.dumps(c)
        # Keep a rank prefix: stop at the first chunk that doesn't fit rather
        # than skipping ahead to smaller, lower-ranked ones. The top hit is
        # kept unconditionally — an over-budget rank-1 result still beats an
        # empty response.
        if i > 0 and budget.count(text) > budget.remaining():
            break
        budget.reserve(f"chunk[{i}]", text)
        kept_chunks.append(c)

    kept_neighbors: list[dict] = []
    for i, n in enumerate(neighbors):
        text = json.dumps(n)
        if budget.count(text) > budget.remaining():
            break
        budget.reserve(f"neighbor[{i}]", text)
        kept_neighbors.append(n)

    kept_summaries: dict[str, str] = {}
    for k, v in summaries.items():
        text = f"{k}: {v}"
        if budget.count(text) > budget.remaining():
            break
        budget.reserve(f"community[{k}]", text)
        kept_summaries[k] = v

    dropped = (
        len(chunks) - len(kept_chunks),
        len(neighbors) - len(kept_neighbors),
        len(summaries) - len(kept_summaries),
    )
    if any(dropped):
        logger.info(
            "response budget %d tokens (used %d): dropped %d chunks, "
            "%d neighbors, %d community summaries",
            SEARCH_RESPONSE_MAX_TOKENS, budget.used(), *dropped,
        )
    return kept_chunks, kept_neighbors, kept_summaries


# Adaptive result-count cutoff. When the score curve has one decisive cliff —
# a consecutive gap of at least ADAPTIVE_TOPK_GAP of the whole top-to-bottom
# spread — everything below the cliff is dropped (never below
# ADAPTIVE_TOPK_MIN results). Flat, ambiguous score curves return the full
# top_k untouched, so the agent is only given less when the ranker is
# actually confident — the lean-mode benchmark showed blanket payload cuts
# starve the agent into extra rounds (docs/benchmark-findings.md #2).
# Default OFF until the agentic benchmark validates the turns tradeoff.
ADAPTIVE_TOPK = os.environ.get("ADAPTIVE_TOPK", "0") == "1"
ADAPTIVE_TOPK_MIN = int(os.environ.get("ADAPTIVE_TOPK_MIN", "3"))
ADAPTIVE_TOPK_GAP = float(os.environ.get("ADAPTIVE_TOPK_GAP", "0.5"))


def _adaptive_cut(reranked: list[dict], gap: float | None = None) -> list[dict]:
    gap = ADAPTIVE_TOPK_GAP if gap is None else gap
    if len(reranked) <= ADAPTIVE_TOPK_MIN:
        return reranked
    scores = [_display_score(r) for r in reranked]
    spread = scores[0] - scores[-1]
    if spread <= 0:
        return reranked
    for i in range(ADAPTIVE_TOPK_MIN, len(scores)):
        if scores[i - 1] - scores[i] >= gap * spread:
            return reranked[:i]
    return reranked


def _splice_snippets(members) -> str | None:
    """Merge rank-hit snippets covering contiguous/overlapping line ranges.

    `members` are ChunkHits from one file, sorted by start_line, whose ranges
    form one contiguous run. Snippets are spliced by absolute line number so
    overlaps don't duplicate text. Returns None when any member's line count
    doesn't match its declared range (don't risk corrupting code).
    """
    lines_by_no: dict[int, str] = {}
    for h in members:
        lines = h.snippet.split("\n")
        if len(lines) != h.end_line - h.start_line + 1:
            return None
        for off, line in enumerate(lines):
            lines_by_no[h.start_line + off] = line
    lo = members[0].start_line
    hi = max(h.end_line for h in members)
    return "\n".join(lines_by_no[n] for n in range(lo, hi + 1))


def _merge_adjacent_hits(hits: list) -> list:
    """Collapse same-file hits with adjacent/overlapping line ranges.

    Fixed-size and structural chunks both tile a file, so two hits from the
    same region arrive as separate entries with a seam. Each contiguous run
    becomes one ChunkHit at the best member's rank, spanning the union range,
    carrying the best member's score — the agent gets one continuous block
    instead of N entries re-stating the same neighborhood.
    """
    by_file: dict[str, list[int]] = {}
    for i, h in enumerate(hits):
        by_file.setdefault(h.file_path, []).append(i)

    replaced: dict[int, object] = {}
    consumed: set[int] = set()
    for idxs in by_file.values():
        if len(idxs) < 2:
            continue
        order = sorted(idxs, key=lambda i: (hits[i].start_line, hits[i].end_line))
        runs, run = [], [order[0]]
        for i in order[1:]:
            if hits[i].start_line <= hits[run[-1]].end_line + 1:
                run.append(i)
            else:
                runs.append(run)
                run = [i]
        runs.append(run)
        for run in runs:
            if len(run) < 2:
                continue
            members = sorted((hits[i] for i in run), key=lambda h: h.start_line)
            spliced = _splice_snippets(members)
            if spliced is None:
                continue
            best = min(run)  # hits is rank-ordered; earliest index = best rank
            merged = hits[best].model_copy(update={
                "start_line": members[0].start_line,
                "end_line": max(h.end_line for h in members),
                "snippet": spliced,
                "score": max(h.score for h in members),
                "graph_enhanced": any(h.graph_enhanced for h in members),
            })
            replaced[best] = merged
            consumed.update(i for i in run if i != best)

    return [replaced.get(i, h) for i, h in enumerate(hits) if i not in consumed]


# Per-chunk context header. One comment line prepended to each returned
# snippet naming the enclosing entity and the symbols defined in the chunk's
# line range, so an agent can judge relevance without a follow-up file read.
# Built from the post-rerank entity fetch — costs ~10-20 tokens per chunk.
_COMMENT_MARKERS = {"python": "#", "bash": "#", "yaml": "#", "ruby": "#"}
_HEADER_MAX_DEFINES = 3
_HEADER_MAX_SIGNATURE = 80


def _chunk_header(hit, entities: list[dict]) -> str | None:
    """Compose `# in Class Foo · defines bar(sig), baz` for a ChunkHit.

    `entities` are the chunk's file's graph entities (absolute line numbers,
    same coordinate space as the chunk). Whole-file Module nodes are skipped —
    their name is just the file path. Returns None when the file has no usable
    entities (e.g. graph pass not run yet).
    """
    cs, ce = hit.start_line, hit.end_line
    enclosing = None
    defined = []
    for e in entities:
        if e.get("type") == "Module" or not e.get("name"):
            continue
        es = int(e.get("start_line") or 0)
        ee = int(e.get("end_line") or 0)
        if es <= cs and ee >= ce and es > 0:
            # innermost enclosing entity = the one starting latest
            if enclosing is None or es > int(enclosing.get("start_line") or 0):
                enclosing = e
        elif cs <= es <= ce:
            defined.append(e)
    if enclosing is None and not defined:
        return None

    parts = []
    if enclosing is not None:
        parts.append(f"in {enclosing.get('type')} {enclosing.get('name')}")
    if defined:
        names = []
        for e in defined[:_HEADER_MAX_DEFINES]:
            names.append(e.get("name", ""))
        listed = ", ".join(n for n in names if n)
        more = len(defined) - _HEADER_MAX_DEFINES
        if more > 0:
            listed += f" (+{more} more)"
        parts.append(f"defines {listed}")
        sig = (defined[0].get("signature") or "").strip()
        if len(defined) == 1 and sig:
            parts.append(sig[:_HEADER_MAX_SIGNATURE])
    marker = _COMMENT_MARKERS.get(hit.language, "//")
    return f"{marker} {' · '.join(parts)}"


# ── Snippet import stripping ─────────────────────────────────
# Drop import/using/#include directive lines from snippet BODIES — that
# dependency signal is already carried by the IMPORTS graph edges, so it's the
# lowest-value-per-token content in a snippet (heaviest on Java/C#). Opt-in,
# default off (STRIP_SNIPPET_IMPORTS); whole-line drops, never truncation, and
# line ranges live in start_line/end_line metadata so nothing positional is
# lost (same contract as the existing blank-line strip).
STRIP_SNIPPET_IMPORTS = os.environ.get("STRIP_SNIPPET_IMPORTS", "0") == "1"

# Per-language matchers for a single-line import directive (tested against the
# left-stripped line). Conservative on purpose: C#'s `^using\s+[\w.]` can't
# match a `using (var x = ...)` resource statement (next char is `(`), and JS
# only strips `import` (never `export`, which is real code). Go is intentionally
# omitted — its `import ( ... )` block can't be matched line-by-line safely.
_IMPORT_LINE_RES: dict[str, list[re.Pattern]] = {
    "python": [re.compile(r"^(import|from)\s")],
    "java": [re.compile(r"^(import|package)\s")],
    "csharp": [re.compile(r"^using\s+[\w.]")],
    "javascript": [re.compile(r"^import\s"),
                   re.compile(r"^(const|let|var)\s+.*\brequire\(")],
    "c": [re.compile(r"^#\s*include\b")],
    "cpp": [re.compile(r"^#\s*include\b")],
    "rust": [re.compile(r"^use\s")],
    "php": [re.compile(r"^(use|namespace|require|require_once|include|include_once)\s")],
    "ruby": [re.compile(r"^(require|require_relative)\s")],
}
for _js in ("typescript", "tsx", "jsx"):
    _IMPORT_LINE_RES[_js] = _IMPORT_LINE_RES["javascript"]

# A matched line that ends with one of these opens a MULTI-line import (JS
# destructured `import {`, Python `from x import (`, a `\` continuation, a
# trailing comma). Keep those whole rather than strip the opener and leave
# dangling members — stripping is only safe on self-contained single-line
# directives.
_IMPORT_CONTINUATION_CHARS = ("(", "{", "[", "\\", ",")


def _strip_import_lines(text: str, language: str | None) -> str:
    """Drop self-contained import/using/#include lines from a snippet body."""
    pats = _IMPORT_LINE_RES.get(language or "")
    if not pats or not text:
        return text
    kept: list[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        if (s and not s.endswith(_IMPORT_CONTINUATION_CHARS)
                and any(p.match(s) for p in pats)):
            continue
        kept.append(ln)
    return "\n".join(kept)


# ── Response modes ──────────────────────────────────────────
# Opt-in, NEVER-default server-side body-shaping. `full` is byte-identical to
# the earlier behavior. The two new modes trade snippet body for a cheaper
# agent-facing payload, validated by mean-turn cost in the agentic benchmark
# (a token drop with a turn rise is a NET LOSS).
RESPONSE_MODES = ("full", "facet", "summary_tail")
# In summary_tail the top SUMMARY_TAIL_HEAD ranks keep their full snippets;
# ranks below get the indexed LLM summary (falling back to the snippet if no
# summary is cached — NEVER an empty body, which reads as "complete" to the
# agent = starvation).
SUMMARY_TAIL_HEAD = 2


def _apply_response_mode(
    chunks: list[dict],
    raw_texts: list[str | None],
    response_mode: str,
    summary_by_sha: dict[str, str] | None = None,
) -> list[dict]:
    """Pure body-shaping transform over already-built chunk dicts.

    `chunks` are the full-mode dicts (post model_dump). `raw_texts` is a parallel
    list of the ORIGINAL Milvus chunk_text for each chunk (captured before the
    header-prepend / blank-line-strip transforms), or None for un-summarizable
    hits (e.g. merged adjacent runs). `summary_by_sha` maps SHA1(chunk_text) ->
    cached summary, used only by summary_tail.

    - full: returned unchanged.
    - facet: drop `snippet`; surface the chunk header (already carried on each
      dict as `header`) and NO code body.
    - summary_tail: keep snippets for the top SUMMARY_TAIL_HEAD ranks; for the
      tail, replace `snippet` with the cached summary, falling back to the
      existing snippet when no summary is available (NEVER empty).
    """
    if response_mode not in RESPONSE_MODES:
        raise ValueError(f"unknown response_mode: {response_mode!r}")
    if response_mode == "full":
        return chunks
    summary_by_sha = summary_by_sha or {}

    out: list[dict] = []
    for i, c in enumerate(chunks):
        d = dict(c)
        if response_mode == "facet":
            # Metadata + header only — the agent fetches the dropped body via
            # the hydrate_chunks tool or read_file. hit_id is the handle.
            d.pop("snippet", None)
            d["hit_id"] = make_hit_id(
                d.get("file_path", ""), d.get("start_line", 0),
                d.get("end_line", 0),
            )
        elif response_mode == "summary_tail":
            if i >= SUMMARY_TAIL_HEAD:
                raw = raw_texts[i] if i < len(raw_texts) else None
                summary = None
                if raw is not None:
                    summary = summary_by_sha.get(llm.chunk_cache_key(raw))
                # Fall back to the full snippet when no summary is cached or the
                # hit is un-summarizable (merged run) — never emit an empty body.
                if summary:
                    d["snippet"] = summary
            # header is a full-mode internal field; only facet surfaces it.
            d.pop("header", None)
        out.append(d)
    return out


# Neighbor payload cap and ranking. Neighbors arrive as raw Neo4j node dicts;
# on a typical hit most are ExternalModule import targets (bare npm/pip names
# the agent can't open). Rank internal entities (they have a file_path) and
# code relationships above imports, then better-connected neighbors first,
# and emit only the agent-facing fields — node bookkeeping (updated_at,
# community_id, centrality) and empty values are dropped.
NEIGHBOR_LIMIT = int(os.environ.get("NEIGHBOR_LIMIT", "20"))
_NEIGHBOR_FIELDS = (
    "id", "type", "name", "signature", "file_path",
    "start_line", "end_line", "language",
)
_REL_PRIORITY = {"CALLS": 0, "INHERITS": 0, "DEFINES": 1, "IMPORTS": 2}


def _rank_neighbors(neighbor_index: dict[str, dict]) -> list[dict]:
    """Order deduped neighbors by usefulness and shape them for the response.

    `neighbor_index` maps neighbor id -> {"nb": node dict, "links": count of
    matched entities it connects to}. Sort is fully deterministic (id is the
    final tie-break).
    """
    ranked = sorted(
        neighbor_index.values(),
        key=lambda rec: (
            0 if rec["nb"].get("file_path") else 1,
            _REL_PRIORITY.get(rec["nb"].get("_rel_type", ""), 3),
            -rec["links"],
            rec["nb"]["id"],
        ),
    )
    shaped = []
    for rec in ranked[:NEIGHBOR_LIMIT]:
        nb = rec["nb"]
        out = {}
        for k in _NEIGHBOR_FIELDS:
            v = nb.get(k)
            if v in (None, "", 0):
                continue
            if k == "name" and v == nb.get("file_path"):
                continue  # whole-file Module nodes use the path as their name
            out[k] = v
        for k in ("_rel_type", "_rel_direction", "_target_id"):
            v = nb.get(k)
            if v:
                out[k] = v
        shaped.append(out)
    return shaped


async def graph_search(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 10,
    traverse_depth: int = 1,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
    use_hybrid: bool | None = None,
    use_hyde: bool | None = None,
    use_summary_vector: bool | None = None,
    use_graph_scoring: bool | None = None,
    rerank_pool: int | None = None,
    response_mode: str = "full",
    include_community_summaries: bool = True,
    adaptive_topk: bool | None = None,
    adaptive_topk_gap: float | None = None,
    strip_imports: bool | None = None,
    query_class_payload: bool | None = None,
    summary_prompt_version: int | None = None,
    return_pool: bool = False,
) -> dict:
    if response_mode not in RESPONSE_MODES:
        raise ValueError(f"unknown response_mode: {response_mode!r}")
    pool = rerank_pool or MILVUS_TOP_K_PRE_RERANK
    use_h = USE_HYBRID if use_hybrid is None else use_hybrid
    use_hy = USE_HYDE if use_hyde is None else use_hyde
    use_sv = USE_SUMMARY_VECTOR if use_summary_vector is None else use_summary_vector
    use_gs = USE_GRAPH_SCORING if use_graph_scoring is None else use_graph_scoring
    strip_imp = STRIP_SNIPPET_IMPORTS if strip_imports is None else strip_imports
    qcp = QUERY_CLASS_PAYLOAD if query_class_payload is None else query_class_payload
    # Adaptive-topk: cut low-confidence tail when the score curve
    # has a cliff. Built env-only (ADAPTIVE_TOPK); these per-request overrides
    # (default None = env value) let the agentic benchmark A/B it + sweep the
    # gap without restarting the indexer. Default flip stays gated on that bench.
    use_atk = ADAPTIVE_TOPK if adaptive_topk is None else adaptive_topk
    atk_gap = ADAPTIVE_TOPK_GAP if adaptive_topk_gap is None else adaptive_topk_gap

    if use_h:
        hyde_dense = await _maybe_hyde_embedding(query_text, language, use_hy)
        vec_results = _store().hybrid_search(
            query_text=query_text,
            query_dense=query_embedding,
            query_summary_dense=query_embedding if use_sv else None,
            hyde_dense=hyde_dense,
            top_k=pool,
            language=language,
            path_prefix=path_prefix,
            source_id=source_id,
            exclude_source_ids=exclude_source_ids,
        )
    else:
        vec_results = _store().search(
            query_embedding,
            top_k=pool,
            language=language,
            path_prefix=path_prefix,
            source_id=source_id,
            exclude_source_ids=exclude_source_ids,
        )

    if not vec_results:
        return {"chunks": [], "neighbors": [], "community_summaries": {}}

    if return_pool:
        # Rerank-input mining: return the raw pre-rerank pool exactly as the
        # reranker receives it (raw chunk_text, pre-merge), skipping rerank/graph.
        return {"pool": [
            {"chunk_text": r["entity"].get("chunk_text", ""),
             "file_path": r["entity"].get("file_path", ""),
             "start_line": r["entity"].get("start_line"),
             "end_line": r["entity"].get("end_line"),
             "distance": r.get("distance")}
            for r in vec_results
        ]}

    try:
        reranked = await rerank(query_text, vec_results, top_k=top_k)
    except Exception:
        # Degrade to raw vector order rather than 500, but make it loud:
        # the fallback dicts have no relevance_score, so graph_rescore
        # min-max normalizes the all-equal set to a flat 1.0 on every chunk
        # (the telltale signature of this fallback). Measured cost on
        # featbit: recall@1 0.54 -> 0.32. The counter exists because this
        # warning went unseen through an entire reranker outage.
        from treeloom.infrastructure import metrics
        metrics.rerank_fallbacks.inc()
        logger.warning(
            "rerank failed (RERANKER_PROVIDER=%s); falling back to raw vector "
            "order — scores will be degraded",
            os.environ.get("RERANKER_PROVIDER", "http"),
            exc_info=True,
        )
        reranked = vec_results[:top_k]

    # Graph context is derived from the post-rerank results only.
    # Previously entities/neighbors/communities were fetched for every file
    # in the pre-rerank pool (~150 files, one Cypher query each), then
    # graph_rescore consulted only the reranked files and the response
    # carried an arbitrary [:20] slice of pool neighbors — wasted Neo4j
    # round-trips and an agent-facing payload unrelated to the returned
    # chunks.
    file_paths = set()
    for r in reranked:
        fp = r["entity"].get("file_path", "")
        if fp:
            file_paths.add(fp)

    entities_by_file = await graph_store.get_entities_by_files_batch(
        list(file_paths), source_id=source_id,
        exclude_source_ids=exclude_source_ids,
    )
    matched_entity_ids = set()
    for entities in entities_by_file.values():
        for e in entities:
            matched_entity_ids.add(e["id"])

    # Fetch depth-1 neighbors for every matched entity in a single
    # round-trip instead of one traverse() (= many queries) per entity.
    # Scope the neighbor lookup so a chunk from an authorized source can't pull
    # in a neighbor from a source the caller isn't allowed to see.
    neighbors_by_entity = await graph_store.get_neighbors_batch(
        list(matched_entity_ids),
        source_id=source_id,
        exclude_source_ids=exclude_source_ids,
    )
    neighbor_index: dict[str, dict] = {}  # id -> {"nb": node dict, "links": n}
    for nbs in neighbors_by_entity.values():
        for nb in nbs:
            nid = nb["id"]
            if nid in matched_entity_ids:
                continue
            rec = neighbor_index.get(nid)
            if rec is None:
                neighbor_index[nid] = {"nb": nb, "links": 1}
            else:
                rec["links"] += 1
    neighbor_entities = _rank_neighbors(neighbor_index)

    # Community summaries are verbose Louvain prose returned ONLY in the agent
    # payload — graph_rescore consumes community *embeddings* (community_embs
    # below), not this text. Gating it off drops ~5-15% of payload
    # and skips two graph round-trips, without touching rescoring. Opt-in via
    # include_community_summaries; default True preserves current behavior.
    if include_community_summaries:
        community_ids = await graph_store.get_community_ids(
            list(matched_entity_ids | neighbor_index.keys())
        )
        summaries = await community.get_community_summaries(community_ids)
    else:
        summaries = {}

    if use_gs and reranked:
        community_embs = await _get_community_embeddings()
        reranked = await graph_rescore(
            reranked, query_embedding, query_text, entities_by_file, community_embs
        )
    if SYMBOL_PROMOTE and reranked:
        reranked = await _promote_definition(
            reranked, vec_results, query_text, source_id, path_prefix,
            exclude_source_ids=exclude_source_ids,
        )
    if use_atk:
        reranked = _adaptive_cut(reranked, gap=atk_gap)

    pre_merge_hits = [chunk_hit_from_milvus(r) for r in reranked]
    # Map a ChunkHit's identity to its ORIGINAL Milvus chunk_text (the value
    # summary_tail's SHA1 must be computed over), captured BEFORE the
    # header-prepend / blank-line-strip transforms below mutate the snippet.
    # _merge_adjacent_hits returns the same object for un-merged hits (so the
    # id stays resolvable) and a fresh model_copy for merged runs (id absent ->
    # un-summarizable -> snippet fallback in summary_tail).
    raw_text_by_id = {
        id(h): (reranked[i]["entity"].get("chunk_text") or "")
        for i, h in enumerate(pre_merge_hits)
    }
    hits = _merge_adjacent_hits(pre_merge_hits)
    # Hoist source_id to the response top level when every hit shares it
    # (always true for repo-scoped searches) — no point repeating it per chunk.
    hit_sources = {h.source_id for h in hits if h.source_id}
    shared_source = next(iter(hit_sources)) if len(hit_sources) == 1 else None
    chunks = []
    raw_texts: list[str | None] = []
    for hit in hits:
        raw_texts.append(raw_text_by_id.get(id(hit)))
        # Strip import/using/#include lines from the BODY before the header is
        # prepended — the header is a comment line, never matched.
        if strip_imp and hit.snippet:
            hit.snippet = _strip_import_lines(hit.snippet, hit.language)
        header = _chunk_header(hit, entities_by_file.get(hit.file_path, []))
        if header:
            # facet mode surfaces the header as its own field instead of
            # prepending it to a (now-absent) snippet.
            if response_mode == "facet":
                hit.header = header
            else:
                hit.snippet = f"{header}\n{hit.snippet}"
        # Drop fully-blank lines and trailing whitespace to save tokens; line
        # ranges live in start_line/end_line metadata, not by counting lines,
        # so this loses no positional information. Leading indentation (code
        # structure) is preserved — only trailing whitespace is stripped.
        hit.snippet = "\n".join(
            ln.rstrip() for ln in hit.snippet.splitlines() if ln.strip()
        )
        hit.score = round(hit.score, 3)
        if shared_source:
            hit.source_id = None
        # exclude_defaults drops empty/False fields (language, graph_enhanced,
        # the nulled source_id); file_path has no default so the ChunkHit wire
        # contract keeps its one required key. score is re-added: 0.0 equals
        # the field default but is a real value (the lowest of the
        # minmax-normalized candidate set).
        d = hit.model_dump(exclude_defaults=True)
        d["score"] = hit.score
        chunks.append(d)

    # Opt-in body-shaping. full = no-op (byte-identical to today).
    if response_mode != "full":
        summary_by_sha: dict[str, str] = {}
        if response_mode == "summary_tail":
            tail_shas = sorted({
                llm.chunk_cache_key(raw)
                for raw in raw_texts[SUMMARY_TAIL_HEAD:]
                if raw
            })
            if tail_shas:
                try:
                    summary_by_sha = await llm.cache_get_many(
                        tail_shas, prompt_version=summary_prompt_version)
                except Exception:
                    # Summary cache unavailable -> every tail chunk falls back
                    # to its full snippet (never an empty body).
                    logger.warning(
                        "summary_tail: summary cache lookup failed; "
                        "tail chunks fall back to full snippets", exc_info=True,
                    )
        chunks = _apply_response_mode(
            chunks, raw_texts, response_mode, summary_by_sha
        )

    # Query-class payload: on symbol-bearing queries the answer is
    # already at rank 1 (via _promote_definition), so drop the aux payload;
    # symbol-free queries keep it. Applied before the budget so the budget only
    # shapes what survives.
    neighbor_entities, summaries = _query_class_payload(
        query_text, neighbor_entities, summaries, qcp
    )

    chunks, neighbor_entities, summaries = _apply_response_budget(
        chunks, neighbor_entities, summaries
    )

    result = {
        "chunks": chunks,
        "neighbors": neighbor_entities,
        "community_summaries": summaries,
    }
    if shared_source:
        result["source_id"] = shared_source
    return result


async def graph_enhanced_search(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 10,
    traverse_depth: int = 1,
    language: str | None = None,
    path_prefix: str | None = None,
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
    use_hybrid: bool | None = None,
    use_hyde: bool | None = None,
    use_summary_vector: bool | None = None,
    use_graph_scoring: bool | None = None,
    rerank_pool: int | None = None,
    response_mode: str = "full",
    include_community_summaries: bool = True,
    adaptive_topk: bool | None = None,
    adaptive_topk_gap: float | None = None,
    strip_imports: bool | None = None,
    query_class_payload: bool | None = None,
    summary_prompt_version: int | None = None,
) -> dict:
    # Currently identical to graph_search — the second-pass neighbor-file
    # Milvus search this entry point was named for was never ported to this
    # adapter (the old body was a byte-for-byte copy of graph_search with a
    # dead `neighbor_file_paths` set). Kept as a separate entry point because
    # the MCP tool `search_code_enhanced` routes here; reintroduce the
    # neighbor-expansion pass in this function if/when it comes back.
    return await graph_search(
        query_embedding,
        query_text,
        top_k=top_k,
        traverse_depth=traverse_depth,
        language=language,
        path_prefix=path_prefix,
        source_id=source_id,
        exclude_source_ids=exclude_source_ids,
        use_hybrid=use_hybrid,
        use_hyde=use_hyde,
        use_summary_vector=use_summary_vector,
        use_graph_scoring=use_graph_scoring,
        rerank_pool=rerank_pool,
        response_mode=response_mode,
        include_community_summaries=include_community_summaries,
        adaptive_topk=adaptive_topk,
        adaptive_topk_gap=adaptive_topk_gap,
        strip_imports=strip_imports,
        query_class_payload=query_class_payload,
        summary_prompt_version=summary_prompt_version,
    )


async def hydrate_chunks(
    hit_ids: list[str],
    source_id: str | None = None,
    exclude_source_ids: list[str] | None = None,
) -> list[dict]:
    """Reconstruct full code bodies for facet `hit_id`s.

    The companion to `response_mode="facet"`: the agent gets a body-less catalog
    plus a `hit_id` per result, then calls this in ONE batch to fetch the exact
    dropped bodies (vs N `read_file` round-trips). Returns ChunkHit-shaped dicts
    (same wire contract as a search response's `chunks`). Bad/stale ids are
    skipped; a backend without `get_chunk_bodies` (lancedb/chromadb) yields [].

    `exclude_source_ids` carries the caller's denied sources and MUST be passed
    through from the endpoint's authorization decision. A `hit_id` is only a
    file path plus a line range, so it is guessable for any repository whose
    layout the caller knows; without the exclusion this endpoint hands back
    bodies from sources the caller's grants deny. A backend whose
    `get_chunk_bodies` cannot express the exclusion fails CLOSED here rather
    than returning unfiltered rows — the same stance `VECTOR_STORE=chromadb`
    already takes on shared-index exclusions.
    """
    from treeloom.domain.shared import parse_hit_id

    locations = [p for hid in hit_ids if (p := parse_hit_id(hid))]
    if not locations:
        return []
    getter = getattr(_store(), "get_chunk_bodies", None)
    if getter is None:
        return []
    if exclude_source_ids:
        import inspect

        if "exclude_source_ids" not in inspect.signature(getter).parameters:
            raise RuntimeError(
                "hydrate_chunks: the active vector store cannot apply "
                "source exclusions; refusing rather than returning rows the "
                "caller's grants deny"
            )
        bodies = await getter(locations, source_id, exclude_source_ids)
    else:
        bodies = await getter(locations, source_id)
    out: list[dict] = []
    for b in bodies:
        hit = chunk_hit_from_milvus({"entity": b, "distance": 0.0})
        # Strip blank/trailing lines for token parity with graph_search snippets.
        if hit.snippet:
            hit.snippet = "\n".join(
                ln.rstrip() for ln in hit.snippet.splitlines() if ln.strip()
            )
        d = hit.model_dump(exclude_defaults=True)
        d["hit_id"] = make_hit_id(
            b.get("file_path", ""), b.get("start_line", 0), b.get("end_line", 0)
        )
        out.append(d)
    return out
