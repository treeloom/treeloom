#!/bin/bash
# Stop the simple-profile deployment (counterpart of run-simple.sh).
set -e

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

INDEXER_PORT="${INDEXER_PORT:-8001}"
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
docker compose -f docker-compose.simple.yml stop
