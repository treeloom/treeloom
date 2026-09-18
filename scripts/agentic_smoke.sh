#!/usr/bin/env bash
# Cross-model agentic smoke fixture (internal-reranker-3 Phase 4).
#
# For each supported agent model, runs a 3-query --native-tools agentic probe
# on the featbit reference set and asserts (via scripts/agentic_smoke_check.py):
# termination (hit_cap false >= 2/3), grounding, mean correctness >= 3.0, and
# served_models + agent_protocol recorded in the summary.
#
# Run this before EVERY paid agentic cell (pennies on deepseek, well under a
# dollar on frontier models). A failure means alias drift or protocol rot —
# stop, do not launch the cell.
#
# Usage:
#   scripts/agentic_smoke.sh [MODEL ...]        # default: the supported set
# Env:
#   AGENTIC_SMOKE_MODELS   space-separated model list (overridden by args)
#   SMOKE_REPO             featbit checkout (default the internal path)
#   SMOKE_SEARCH_URL       indexer URL (default http://localhost:8001)
#   SMOKE_DRY_RUN=1        print the commands without running anything
#
# Vendor keys (DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY) must be
# exported in this shell; unset NOMCP_LLM_URL or per-model routing is disabled.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_MODELS="deepseek-chat deepseek-v4-pro gpt-4o-2024-08-06 claude-sonnet-4-6"
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    # shellcheck disable=SC2206
    MODELS=(${AGENTIC_SMOKE_MODELS:-$DEFAULT_MODELS})
fi

SMOKE_REPO="${SMOKE_REPO:-$HOME/source/featbit}"
SMOKE_SEARCH_URL="${SMOKE_SEARCH_URL:-http://localhost:8001}"
# Overridable: a re-pinned corpus needs its OWN regenerated query set, since
# ground-truth paths follow the tree. The old default points at paths that
# do not exist outside the original checkout, which scores 0.0 silently.
QUERIES="${SMOKE_QUERIES:-benchmarks/queries/featbit-clean-100.jsonl}"
JUDGE_MODEL="gpt-4o-2024-08-06"   # dated, pinned — never a floating alias
RESULTS_DIR="benchmarks/results/agentic"

FAIL=0
for M in "${MODELS[@]}"; do
    CMD=(python -m treeloom.benchmark agentic
         --queries "$QUERIES"
         --repo "$SMOKE_REPO"
         --search-url "$SMOKE_SEARCH_URL"
         --arms treeloom
         --model "$M"
         --judge-model "$JUDGE_MODEL"
         --limit 3
         --native-tools
         --allow-floating-model)

    if [ "${SMOKE_DRY_RUN:-0}" = "1" ]; then
        echo "DRY-RUN: ${CMD[*]}"
        echo "DRY-RUN: python scripts/agentic_smoke_check.py <rows.jsonl> --summary <summary.json> --model $M"
        continue
    fi

    echo "=== smoke: $M ==="
    OUT="$("${CMD[@]}" 2>&1)"
    STATUS=$?
    printf '%s\n' "$OUT" | tail -6

    # Locate this run's summary robustly: prefer the _summary_path the CLI
    # prints in its result JSON; fall back to the newest summary on disk.
    SUMMARY="$(printf '%s\n' "$OUT" \
        | grep -o '"_summary_path": *"[^"]*"' | tail -1 \
        | sed 's/.*: *"//; s/"$//')"
    if [ -z "$SUMMARY" ]; then
        SUMMARY="$(ls -t "$RESULTS_DIR"/*_summary.json 2>/dev/null | head -1)"
    fi
    ROWS="${SUMMARY%_summary.json}.jsonl"

    if [ "$STATUS" -ne 0 ] || [ -z "$SUMMARY" ] || [ ! -f "$ROWS" ]; then
        echo "FAIL $M: run failed (exit $STATUS) or rows not found (${ROWS:-?})"
        FAIL=1
        continue
    fi

    if ! python scripts/agentic_smoke_check.py "$ROWS" --summary "$SUMMARY" --model "$M"; then
        FAIL=1
    fi
done

if [ "${SMOKE_DRY_RUN:-0}" = "1" ]; then
    echo "DRY-RUN complete (no runs executed)."
    exit 0
fi
if [ "$FAIL" -ne 0 ]; then
    echo "SMOKE FIXTURE FAILED — do not launch a paid cell."
    exit 1
fi
echo "SMOKE FIXTURE PASSED for: ${MODELS[*]}"
