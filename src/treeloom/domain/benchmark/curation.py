"""Benchmark domain: apply curation verdicts to a query set — pure, no I/O.

The Argilla review workflow produces per-row verdicts
(approve / edit / reject). This module folds them back into the canonical
query JSONL as a *versioned* export: every surviving row carries a
`curation` block so the review outcome survives set regeneration, and
rejected rows are dropped from the canonical set.

Row status semantics in the v2 output:
  - approved:  kept as-is
  - edited:    `query` replaced by the reviewer's corrected_query; the
               original text is preserved in curation.original_query
  - unreviewed / flagged: kept, awaiting review (flagged = the off-domain
               pre-screen marked it and no human has ruled yet) — the
               benchmark-side gate warns on these
  - rejected:  dropped (counted in stats, not emitted)

An `edit` verdict with no corrected_query is kept unchanged and counted as
`edit_missing_correction` — a review-workflow error to surface, not hide.
"""
from __future__ import annotations

import collections
from typing import Any


def apply_verdicts(
    rows: list[dict],
    verdicts: dict[str, dict[str, Any]],
    *,
    dataset: str | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Fold reviewer verdicts into query rows.

    `verdicts` maps query id -> {verdict, corrected_query, review_reason,
    reviewer, reviewed_at, prior_status}; ids absent from the mapping (or
    with no verdict) are treated as unreviewed, inheriting `prior_status`
    ("flagged" from the pre-screen, else "unreviewed").

    Returns (v2_rows, stats). Input rows are not mutated.
    """
    v2: list[dict] = []
    stats: collections.Counter = collections.Counter()
    for row in rows:
        qid = row.get("id")
        v = verdicts.get(qid) or {}
        verdict = (v.get("verdict") or "").strip().lower()
        curation: dict[str, Any] = {}
        if dataset:
            curation["dataset"] = dataset
        for key in ("reviewer", "reviewed_at"):
            if v.get(key):
                curation[key] = v[key]
        if v.get("review_reason"):
            curation["reason"] = v["review_reason"]

        if verdict == "reject":
            stats["rejected_dropped"] += 1
            continue

        out = dict(row)
        if verdict == "approve":
            curation["status"] = "approved"
            stats["approved"] += 1
        elif verdict == "edit":
            corrected = (v.get("corrected_query") or "").strip()
            if corrected:
                curation["status"] = "edited"
                curation["original_query"] = row.get("query")
                out["query"] = corrected
                stats["edited"] += 1
            else:
                curation["status"] = "edit_missing_correction"
                stats["edit_missing_correction"] += 1
        else:
            prior = (v.get("prior_status") or "unreviewed").strip().lower()
            status = "flagged" if prior == "flagged" else "unreviewed"
            curation["status"] = status
            stats[status] += 1

        out["curation"] = curation
        v2.append(out)
    return v2, dict(stats)


def curation_summary(rows: list[dict]) -> dict[str, int]:
    """Count rows per curation status; rows with no curation block count as
    "uncurated" (the set never went through a review pass at all)."""
    counts: collections.Counter = collections.Counter()
    for row in rows:
        curation = row.get("curation")
        if not isinstance(curation, dict) or not curation.get("status"):
            counts["uncurated"] += 1
        else:
            counts[curation["status"]] += 1
    return dict(counts)
