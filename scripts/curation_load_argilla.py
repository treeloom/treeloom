#!/usr/bin/env python3
"""load a query(+gold) set into Argilla for curation review.

Run: uv run --with argilla python scripts/curation_load_argilla.py \
        --queries benchmarks/queries/<set>.jsonl --repo /path/to/checkout \
        [--gold benchmarks/gold/<repo>.jsonl] [--dataset <name>]

Required env:
  ARGILLA_API_URL   e.g. https://argilla.example.com
  ARGILLA_API_KEY   an owner/admin key for that instance

Pre-screen: v1 off-domain-term check (lemma-normalized repo-vocabulary
lookup, treeloom.domain.benchmark.query_hygiene) — flags query content-words
absent from the target repo's vocabulary, the sense-drift tell.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import argilla as rg

from treeloom.domain.benchmark.query_hygiene import (
    build_vocab_lemmas,
    off_domain_terms,
)

VOCAB_INCLUDES = ["*.cs", "*.ts", "*.tsx", "*.py", "*.md", "*.js", "*.jsx",
                  "*.rs", "*.go", "*.java", "*.html", "*.sql", "*.json"]


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        sys.exit(f"error: {name} must be set (see module docstring)")
    return value


def repo_vocab_lemmas(repo: str) -> frozenset:
    out = subprocess.run(
        ["grep", "-rhoiE", "[a-zA-Z]{4,}", repo]
        + [f"--include={g}" for g in VOCAB_INCLUDES],
        capture_output=True, text=True)
    return build_vocab_lemmas(out.stdout.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True, help="query JSONL to load")
    ap.add_argument("--gold", default=None, help="gold JSONL (optional; ids must match)")
    ap.add_argument("--repo", required=True, help="repo checkout for the pre-screen vocabulary")
    ap.add_argument("--dataset", default=None,
                    help="Argilla dataset name (default: <queries-stem>-curation)")
    args = ap.parse_args()

    api_url = require_env("ARGILLA_API_URL").rstrip("/")
    client = rg.Argilla(api_url=api_url, api_key=require_env("ARGILLA_API_KEY"))

    dataset_name = args.dataset or (
        os.path.splitext(os.path.basename(args.queries))[0] + "-curation")
    repo_name = os.path.basename(args.repo.rstrip("/"))

    vocab = repo_vocab_lemmas(args.repo)
    qrows = {json.loads(l)["id"]: json.loads(l) for l in open(args.queries)}
    grows = ({json.loads(l)["id"]: json.loads(l) for l in open(args.gold)}
             if args.gold else {})

    settings = rg.Settings(
        guidelines=(
            f"Curation review for the {dataset_name} benchmark set "
            f"(treeloom curation, repo: {repo_name}).\n\n"
            "For each record: read the QUERY, the GOLD ANSWER (if present), and "
            "the relevant files. Judge whether the query is answerable, "
            f"on-domain for {repo_name}, and faithful to the ground truth.\n\n"
            "- APPROVE: query is answerable and on-domain as written.\n"
            "- EDIT: salvageable but wrong wording (e.g. sense drift: a term "
            "resolved to a domain the repo doesn't have). Put the corrected "
            "wording in 'corrected_query'.\n"
            "- REJECT: unanswerable, off-domain beyond repair, or ground truth "
            "wrong.\n\n"
            "Records with pre-screen flags (lemma-normalized off-domain terms "
            "absent from the repo vocabulary) are marked review_status=flagged "
            "in metadata — review those first; flags are a heuristic, not a "
            "verdict."
        ),
        fields=[
            rg.TextField(name="query", title="Benchmark query"),
            rg.TextField(name="gold_answer", title="Gold answer", use_markdown=True),
            rg.TextField(name="relevant_files", title="Ground-truth files"),
            rg.TextField(name="prescreen_flags", title="Pre-screen flags (v1, lemma-normalized)"),
        ],
        questions=[
            rg.LabelQuestion(name="verdict", title="Review verdict",
                             labels=["approve", "edit", "reject"], required=True),
            rg.TextQuestion(name="corrected_query", title="Corrected query (if verdict=edit)",
                            required=False),
            rg.TextQuestion(name="review_reason", title="Reason (esp. for edit/reject)",
                            required=False),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="review_status"),
            rg.TermsMetadataProperty(name="strategy"),
            rg.TermsMetadataProperty(name="difficulty"),
        ],
    )

    existing = client.datasets(name=dataset_name)
    if existing:
        print(f"dataset '{dataset_name}' already exists — deleting and recreating")
        existing.delete()
    dataset = rg.Dataset(name=dataset_name, settings=settings)
    dataset.create()

    records, n_flagged = [], 0
    for qid, q in sorted(qrows.items()):
        g = grows.get(qid, {})
        fl = off_domain_terms(q["query"], vocab)
        n_flagged += bool(fl)
        # Argilla 2.x's dict-record mapper ignores a nested `metadata` key —
        # metadata fields go flat at the record top level.
        records.append({
            "id": qid,
            "query": q["query"],
            "gold_answer": g.get("gold_answer", "(no gold answer)"),
            "relevant_files": "\n".join(q.get("relevant_files", [])),
            "prescreen_flags": ", ".join(fl) if fl else "(none)",
            "review_status": "flagged" if fl else "unreviewed",
            "strategy": str(q.get("strategy")),
            "difficulty": str(q.get("difficulty")),
        })

    dataset.records.log(records)
    print(f"loaded {len(records)} records ({n_flagged} pre-screen flagged) into "
          f"Argilla dataset '{dataset_name}' at {api_url}")


if __name__ == "__main__":
    main()
