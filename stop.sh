#!/bin/bash
set -e

if [ ! -f .env ]; then
    echo "ERROR: .env not found in $(pwd). Cannot determine which port to stop." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

: "${INDEXER_PORT:?INDEXER_PORT is not set in .env}"

echo "=== Stopping host indexer (port ${INDEXER_PORT}) ==="
pids=$(lsof -ti:"${INDEXER_PORT}" 2>/dev/null || true)
if [ -n "$pids" ]; then
    kill $pids
    echo "Stopped pids: $pids"
else
    echo "No process listening on port ${INDEXER_PORT}"
fi

echo ""
echo "=== Stopping Docker services ==="
docker compose stop tei-embedding tei-reranker mcp-server
