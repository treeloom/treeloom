#!/bin/bash
# Simple deployment quickstart — see docs/simple-mode.md.
# One container by default (postgres) and the indexer on the host with
# embedded stores + cloud APIs. Needs: Docker, Python 3.11+, an OpenAI key.
# MCP access is stdio by default (treeloom-mcp, no container); the
# deprecated HTTP+SSE transport is opt-in via COMPOSE_PROFILES=http-mcp,
# which also starts the mcp-server container.
set -e

if [ ! -f .env ]; then
    echo "ERROR: .env not found in $(pwd)." >&2
    echo "  cp .env.simple.example .env   # then put your OPENAI_API_KEY in it" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

if [ "${TREELOOM_PROFILE:-}" != "simple" ]; then
    echo "WARNING: TREELOOM_PROFILE is not 'simple' in .env — this script" >&2
    echo "expects the simple profile (see .env.simple.example)." >&2
fi
if [ -z "${OPENAI_API_KEY:-}${EMBEDDING_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set in .env — the simple profile uses" >&2
    echo "it for embeddings and HyDE/summaries." >&2
    exit 1
fi

echo "=== Installing Python dependencies (with the [simple] extras) ==="
pip install --quiet -e '.[simple]'

echo "=== Starting Docker services (Postgres) ==="
if [[ ",${COMPOSE_PROFILES:-}," == *",http-mcp,"* ]]; then
    echo "COMPOSE_PROFILES includes http-mcp: also starting the deprecated HTTP+SSE mcp-server container."
    echo "Deprecated HTTP+SSE transport: http://localhost:${MCP_SERVER_PORT:-8000}/sse"
    if [[ "${INDEXER_HOST:-127.0.0.1}" == 127.* || "${INDEXER_HOST:-127.0.0.1}" == "localhost" ]]; then
        echo "  WARNING: INDEXER_HOST=${INDEXER_HOST:-127.0.0.1} is loopback-only, so the"
        echo "  mcp-server container cannot reach the indexer via host.docker.internal."
        echo "  Point INDEXER_HOST at the Docker bridge address instead (usually"
        echo "  172.17.0.1 — check: docker network inspect bridge). That is reachable"
        echo "  from the container without publishing the indexer on every interface,"
        echo "  which 0.0.0.0 would do."
    fi
else
    echo "MCP transport: stdio only (default) — mcp-server container not started."
    echo "Want the HTTP+SSE transport too? Add http-mcp to COMPOSE_PROFILES in .env, or run:"
    echo "  COMPOSE_PROFILES=http-mcp docker compose -f docker-compose.simple.yml up -d mcp-server"
fi
docker compose -f docker-compose.simple.yml up -d --build

INDEXER_HOST="${INDEXER_HOST:-127.0.0.1}"
INDEXER_PORT="${INDEXER_PORT:-8001}"
echo ""
echo "=== Starting indexer on host ==="
echo "Indexer:    http://localhost:${INDEXER_PORT}"
echo ""
echo "MCP clients spawn the stdio server (default, no container):"
cat <<MCPJSON
{
  "mcpServers": {
    "treeloom": {
      "command": "treeloom-mcp",
      "env": {
        "INDEXER_URL": "http://localhost:${INDEXER_PORT}",
        "TREELOOM_MCP_TOKEN": "<your treeloom PAT or API key>"
      }
    }
  }
}
MCPJSON
# Mirror _apply_simple_reranker_default: an explicit RERANKER_PROVIDER wins,
# else the first cloud rerank key present, else the local CPU cross-encoder.
RERANK="${RERANKER_PROVIDER:-}"
if [ -z "$RERANK" ]; then
    if [ -n "${COHERE_API_KEY:-}" ]; then RERANK=cohere
    elif [ -n "${VOYAGE_API_KEY:-}" ]; then RERANK=voyage
    elif [ -n "${ZEROENTROPY_API_KEY:-}" ]; then RERANK=zeroentropy
    else RERANK=local
    fi
fi
if [ "$RERANK" = "local" ]; then
    echo "Reranker: local CPU cross-encoder — the first search downloads the model"
    echo "(~150MB by default); subsequent ones are fast."
else
    echo "Reranker: ${RERANK}"
fi
echo "Press Ctrl+C to stop"
echo ""
echo "=== SECURITY NOTE ==="
echo "Simple mode runs WITHOUT authentication (AUTH_ENABLED=false)."
echo "The indexer listens on ${INDEXER_HOST}:${INDEXER_PORT}. Postgres (and mcp-server,"
echo "if enabled) are loopback-only (127.0.0.1)."
if [[ "${INDEXER_HOST}" == 127.* || "${INDEXER_HOST}" == "localhost" ]]; then
    echo "Loopback-only (the default): any process on THIS machine can use the"
    echo "unauthenticated indexer — fine on a single-user laptop, not on a shared host."
else
    echo "INDEXER_HOST is not loopback: the unauthenticated indexer is reachable from"
    echo "other hosts. Block port ${INDEXER_PORT} with a host firewall rule, or set"
    echo "AUTH_ENABLED=true (plus TREELOOM_HMAC_SECRET) in .env."
fi
echo "====================="
echo ""
uvicorn treeloom.indexer_service:app --host "${INDEXER_HOST}" --port "${INDEXER_PORT}"
