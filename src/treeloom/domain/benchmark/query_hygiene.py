"""Benchmark domain: answerable-query hygiene filter — pure, no I/O.

A query row is "answerable" when:
  - it has an entity name,
  - the query text mentions that name (verbatim or all camelCase tokens), AND
  - either the entity name is discriminating (maps to ≤3 distinct files across
    the corpus) or the query carries a path/namespace hint from a relevant file.

This mirrors the logic from benchmarks/make_clean_queries.py but is generalised:
  - BASE_STOP_PARTS covers generic code-structure tokens; repo-specific tokens
    (e.g. "featbit", "src") are passed by the caller as stop_parts.
  - All functions are pure (no file I/O, no randomness).
"""
from __future__ import annotations

import collections
import re

# Generic path-part tokens that don't discriminate one file from another.
# Callers should extend with repo-specific stop parts (project name, user name,
# machine-specific path components, etc.).
BASE_STOP_PARTS: frozenset[str] = frozenset({
    "",
    "src", "app", "modules",
    "ts", "cs", "py", "html", "java", "go", "rs", "cpp", "h",
})


def camel_tokens(name: str) -> list[str]:
    """Split a camelCase / PascalCase name into lowercased word tokens (len>1)."""
    return [t.lower() for t in
            re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name or "")
            if len(t) > 1]


def mentions(query_lower: str, name: str) -> bool:
    """Return True if the lowercased query text mentions *name*.

    Matches either as a literal substring or when every camelCase token of
    *name* appears somewhere in the query.
    """
    if name.lower() in query_lower:
        return True
    toks = camel_tokens(name)
    return bool(toks) and all(t in query_lower for t in toks)


def path_hint(row: dict, stop_parts) -> bool:
    """Return True if any relevant-file path segment (not a stop part) appears in the query.

    *stop_parts* is the union of BASE_STOP_PARTS and any caller-supplied set.
    """
    effective_stop = BASE_STOP_PARTS | set(stop_parts)
    q = row["query"].lower()
    for f in row.get("relevant_files") or []:
        parts = re.split(r"[/.]", f.lower())
        good = [p for p in parts if p not in effective_stop and len(p) > 3]
        if any(p in q for p in good):
            return True
    return False


def name_to_files(rows: list[dict]) -> dict[str, set[str]]:
    """Build a lowercase-entity-name → set(relevant_files) map across *rows*."""
    name_files: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        n = ((r.get("entity") or {}).get("name") or "").lower()
        if n:
            name_files[n].update(r.get("relevant_files") or [])
    return dict(name_files)


def is_answerable(row: dict, name_files: dict[str, set[str]], stop_parts) -> bool:
    """Return True when *row* passes the answerability filter.

    Rules:
      - Must have a non-empty entity name.
      - Query must mention the entity name (verbatim or all camelCase tokens).
      - The entity name must be discriminating (maps to ≤3 distinct files in
        the corpus) OR the query must carry a path/namespace hint.
    """
    name = (row.get("entity") or {}).get("name") or ""
    if not name:
        return False
    q = row["query"].lower()
    if not mentions(q, name):
        return False
    files_for_name = name_files.get(name.lower(), set())
    return len(files_for_name) <= 3 or path_hint(row, stop_parts)


def filter_answerable(rows: list[dict], stop_parts=()) -> list[dict]:
    """Return the subset of *rows* that pass the answerability filter.

    *stop_parts* is merged with BASE_STOP_PARTS internally; callers supply
    repo-specific tokens (project name, username, machine-specific dirs).
    """
    effective_stop = set(stop_parts)
    nf = name_to_files(rows)
    return [r for r in rows if is_answerable(r, nf, effective_stop)]


def dedup(rows: list[dict]) -> list[dict]:
    """Drop rows whose stripped, lowercased query text was already seen; keep first."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = r["query"].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def apply_quota(rows: list[dict], quota: dict[str, int] | None) -> list[dict]:
    """Apply per-strategy caps to *rows*, preserving input order.

    If *quota* is None, return *rows* unchanged (no capping).
    Missing strategy values default to the label ``"base"``.  The loop stops
    early once every quota bucket is exhausted.
    """
    if quota is None:
        return list(rows)
    remaining = dict(quota)
    picked: list[dict] = []
    for r in rows:
        s = r.get("strategy") or "base"
        if remaining.get(s, 0) <= 0:
            continue
        remaining[s] -= 1
        picked.append(r)
        if not any(remaining.values()):
            break
    return picked


# ── off-domain pre-screen (v1) ────────────────────────────────────────────
# Flags query content-words absent from the target repo's vocabulary — the
# sense-drift tell ("insurance rules" generated against featbit, which
# has no insurance domain). v0 (scripts/curation_load_argilla.py) checked raw
# words with naive s-stripping and flagged 38/120 on featbit, mostly gerund/
# nominalization false positives ("wrapping", "listing"). v1 normalizes both
# the query words and the vocabulary through the same cheap suffix lemmatizer
# and widens the generic-English stoplist. Pure: caller supplies the vocab.

GENERIC_QUERY_WORDS: frozenset[str] = frozenset("""
how what where which when whose does specific using retrieve retrieves list
lists return returns returning associated response paginated endpoint unique
identifier method function class file implement implementation handle handles
handled handler create created creating creates update updated updates delete
deleted deletes deleting their there would should could these those against
based include includes including provide provides provided given without
within ensure ensures ensuring convert converts converting structured generic
value values object objects string strings number numbers boolean multiple
single optional default defaults existing available currently correctly
properly attempt attempts trying tries process processes processing result
results contain contains containing perform performs performing check checks
checking verify verifies verifying determine determines determining
""".split())

_SUFFIXES = ("ings", "ing", "edly", "ed", "ies", "es", "s", "ly", "tion", "tions")


def lemma(word: str) -> str:
    """Cheap suffix-stripping normal form (NOT linguistic lemmatization).

    Good enough to match "wrapping"→"wrap" against a vocab containing
    "wrap"/"wrapped"; both sides of the comparison must go through it.
    """
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf == "ies":
                return base + "y"
            # collapse the doubling -ing/-ed introduce: wrapping→wrapp→wrap
            # (only those suffixes double the final consonant — "falls" must
            # stay "fall", not collapse to "fal")
            if (suf in ("ings", "ing", "ed", "edly")
                    and len(base) >= 4 and base[-1] == base[-2]
                    and base[-1] not in "aeiou"):
                base = base[:-1]
            return base
    return w


def off_domain_terms(
    query: str,
    vocab_lemmas: frozenset[str] | set[str],
    *,
    min_len: int = 5,
    extra_stop: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """Content-words of *query* whose lemma is absent from the repo vocabulary.

    `vocab_lemmas` must already be lemma-normalized (see `build_vocab_lemmas`).
    Returns the offending words sorted, original surface form, for display.
    """
    flagged = []
    for w in set(re.findall(r"[a-zA-Z]{%d,}" % min_len, query.lower())):
        if w in GENERIC_QUERY_WORDS or w in extra_stop:
            continue
        lw = lemma(w)
        if lw in GENERIC_QUERY_WORDS:
            continue
        if lw not in vocab_lemmas and w not in vocab_lemmas:
            flagged.append(w)
    return sorted(flagged)


def build_vocab_lemmas(words) -> frozenset[str]:
    """Lemma-normalized vocabulary set from an iterable of raw words."""
    out = set()
    for w in words:
        wl = w.lower()
        out.add(wl)
        out.add(lemma(wl))
    return frozenset(out)
