#!/usr/bin/env python3
"""export Argilla curation verdicts back to a versioned query JSONL.

The other half of scripts/curation_load_argilla.py: pull reviewer responses
from an Argilla dataset, fold them into the original query set via the pure
treeloom.domain.benchmark.curation.apply_verdicts, and write a v2 JSONL where
every row carries a `curation` block (rejected rows are dropped).

Run: env -u PYTHONPATH python scripts/curation_export_argilla.py \
        --dataset <argilla-dataset-name> \
        --queries benchmarks/queries/<set>.jsonl \
        [--out benchmarks/queries/<set>.v2.jsonl]

Required env:
  ARGILLA_API_URL   e.g. https://argilla.example.com
  ARGILLA_API_KEY   an owner/admin key for that instance

Uses the plain REST API (no argilla package needed). Multiple responses per
record: the latest *submitted* response wins.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

from treeloom.domain.benchmark.curation import apply_verdicts


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        sys.exit(f"error: {name} must be set (see module docstring)")
    return value


API_URL = ""
API_KEY = ""


def get(path: str):
    req = urllib.request.Request(
        API_URL + path, headers={"X-Argilla-Api-Key": API_KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_all_records(dataset_id: str) -> list[dict]:
    records, offset = [], 0
    while True:
        page = get(f"/api/v1/datasets/{dataset_id}/records"
                   f"?include=responses&limit=500&offset={offset}")["items"]
        records.extend(page)
        if len(page) < 500:
            return records
        offset += len(page)


def username_map() -> dict[str, str]:
    try:
        return {u["id"]: u["username"] for u in get("/api/v1/users")["items"]}
    except Exception:
        return {}  # non-owner key: fall back to raw user ids


def latest_submitted(responses: list[dict]) -> dict | None:
    submitted = [r for r in responses or [] if r.get("status") == "submitted"]
    if not submitted:
        return None
    return max(submitted, key=lambda r: r.get("updated_at") or r.get("inserted_at") or "")


def main() -> None:
    global API_URL, API_KEY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Argilla dataset name")
    ap.add_argument("--queries", required=True, help="original query JSONL")
    ap.add_argument("--out", default=None,
                    help="output path (default: <queries-stem>.v2.jsonl)")
    args = ap.parse_args()

    API_URL = require_env("ARGILLA_API_URL").rstrip("/")
    API_KEY = require_env("ARGILLA_API_KEY")

    workspaces = get("/api/v1/me/workspaces")["items"]
    dataset = None
    for ws in workspaces:
        for d in get(f"/api/v1/me/datasets?workspace_id={ws['id']}")["items"]:
            if d["name"] == args.dataset:
                dataset = d
    if not dataset:
        sys.exit(f"error: dataset '{args.dataset}' not found on {API_URL}")

    users = username_map()
    verdicts: dict[str, dict] = {}
    for rec in fetch_all_records(dataset["id"]):
        qid = rec.get("external_id")
        if not qid:
            continue
        entry: dict = {
            "prior_status": (rec.get("metadata") or {}).get("review_status"),
        }
        resp = latest_submitted(rec.get("responses"))
        if resp:
            values = {k: v.get("value") for k, v in (resp.get("values") or {}).items()}
            entry.update({
                "verdict": values.get("verdict"),
                "corrected_query": values.get("corrected_query"),
                "review_reason": values.get("review_reason"),
                "reviewer": users.get(resp.get("user_id"), resp.get("user_id")),
                "reviewed_at": resp.get("updated_at") or resp.get("inserted_at"),
            })
        verdicts[qid] = entry

    rows = [json.loads(l) for l in open(args.queries) if l.strip()]
    v2, stats = apply_verdicts(rows, verdicts, dataset=args.dataset)

    out_path = args.out or (os.path.splitext(args.queries)[0] + ".v2.jsonl")
    with open(out_path, "w") as f:
        for row in v2:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ids_in_argilla = set(verdicts)
    missing = [r["id"] for r in rows if r["id"] not in ids_in_argilla]
    if missing:
        print(f"warning: {len(missing)} query ids absent from the Argilla dataset "
              f"(treated as unreviewed), e.g. {missing[:3]}")
    print(f"wrote {len(v2)} rows to {out_path}")
    print("stats:", json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
