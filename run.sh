#!/bin/bash
set -e

# Load .env so port/URI vars are available to docker-compose substitution
# and the host uvicorn process. The .env file is the single source of truth
# for ports and URIs; missing values cause the relevant step to fail loudly.
if [ ! -f .env ]; then
    echo "ERROR: .env not found in $(pwd). Copy .env.example to .env and configure it." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

# Validate the env vars this script directly needs.
: "${INDEXER_HOST:?INDEXER_HOST is not set in .env}"
: "${INDEXER_PORT:?INDEXER_PORT is not set in .env}"

# Operator UI v2 (the `ui` compose service). The SPA logs in with
# username/password, which uses a session cookie — a credentialed cross-origin
# request that the wildcard CORS default CANNOT serve. So default
# TREELOOM_CORS_ORIGINS to the UI origin (explicit origin + allow-credentials).
# Reaching the UI from another origin (127.0.0.1 / a hostname / LAN IP)? Add it
# to this comma-separated list in .env.
UI_PORT="${UI_PORT:-3001}"
if [ -z "${TREELOOM_CORS_ORIGINS:-}" ]; then
    export TREELOOM_CORS_ORIGINS="http://localhost:${UI_PORT}"
fi

echo "=== Installing Python dependencies ==="
pip install --quiet -e .

echo "=== Building MCP server + UI images ==="
docker compose build mcp-server ui

echo ""
echo "=== Starting Docker services (Postgres + TEI) ==="
# Postgres is a hard dependency of the host indexer (job/source/cache state);
# it fails loud on startup if DATABASE_URL is unreachable.
# Profiles (COMPOSE_PROFILES in .env, comma-separated):
#   local-infra  starts Neo4j + Milvus standalone in compose (the quickstart
#                default in .env.example; drop it if you run them externally
#                and point NEO4J_URI / MILVUS_URI at your own instances)
#   qwen3        starts the Qwen3 reranker (services/qwen3_reranker/README.md)
#   cpu          uses tei-embedding-cpu instead of the GPU tei-embedding
#                (no NVIDIA GPU; also set EMBEDDING_URL=http://localhost:8083)
#   http-mcp     ALSO starts the mcp-server container (deprecated HTTP+SSE
#                transport). The default MCP transport is stdio via the
#                `treeloom-mcp` console script, which this script does not
#                start — your MCP client spawns it directly and needs no
#                container. Only add this profile if you specifically want
#                the opt-in HTTP transport; see .env.example.
EXTRA_SERVICES=""
EMBEDDING_SERVICE="tei-embedding"
if [[ ",${COMPOSE_PROFILES:-}," == *",qwen3,"* ]]; then
    EXTRA_SERVICES="$EXTRA_SERVICES qwen3-reranker"
fi
if [[ ",${COMPOSE_PROFILES:-}," == *",local-infra,"* ]]; then
    EXTRA_SERVICES="$EXTRA_SERVICES neo4j milvus"
fi
if [[ ",${COMPOSE_PROFILES:-}," == *",cpu,"* ]]; then
    EMBEDDING_SERVICE="tei-embedding-cpu"
fi
if [[ ",${COMPOSE_PROFILES:-}," == *",http-mcp,"* ]]; then
    EXTRA_SERVICES="$EXTRA_SERVICES mcp-server"
    echo "COMPOSE_PROFILES includes http-mcp: starting the deprecated HTTP+SSE mcp-server container too."
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
    echo "  COMPOSE_PROFILES=http-mcp docker compose up -d mcp-server"
fi
docker compose up -d postgres ${EMBEDDING_SERVICE} tei-reranker ui ${EXTRA_SERVICES}

echo ""
echo "=== Starting indexer on host ==="
echo "Listening on http://${INDEXER_HOST}:${INDEXER_PORT}"
echo "Operator UI:  http://localhost:${UI_PORT}"
echo "Press Ctrl+C to stop"
echo ""
uvicorn treeloom.indexer_service:app --host "${INDEXER_HOST}" --port "${INDEXER_PORT}"