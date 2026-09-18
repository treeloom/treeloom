import logging
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel
from starlette.routing import Mount

from treeloom.infrastructure.tracing import init_tracer
from treeloom.mcp_server import mcp, search_code

# NOTE: graph_store was removed — the MCP server is a pure API consumer
# that proxies to the host indexer.  Neo4j / graph connections are managed
# exclusively by the indexer process on the host.

sse = SseServerTransport("/messages/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracer("treeloom-mcp", app=app)
    # TREELOOM_MCP_SERVICE_ACCOUNT lets the service account stand in for a
    # caller that sent no credential. That is defensible on stdio (the caller
    # IS the operator) but on this HTTP transport it re-creates the confused
    # deputy: any network caller who reaches the port inherits service-account
    # authority over the indexer. This module only ever serves HTTP, so warn
    # unconditionally.
    if os.environ.get("TREELOOM_MCP_SERVICE_ACCOUNT", "").lower() == "true":
        logging.getLogger(__name__).warning(
            "TREELOOM_MCP_SERVICE_ACCOUNT=true on the HTTP transport: callers "
            "that send no bearer will act with the service account's authority "
            "over the indexer. Intended for stdio/single-user only — unset it "
            "and let callers present their own credential."
        )
    yield  # no teardown needed — indexer manages its own connections


app = FastAPI(title="Treeloom MCP Server", lifespan=lifespan)

# ── MCP transport security ─────────────────────────────────────────
# The MCP spec makes Origin validation a MUST for HTTP transports: a browser on
# any website can reach http://localhost:8000 cross-origin, and this transport
# requires no credential, so an unguarded server is drivable by a malicious page
# (DNS rebinding). Reject a present-but-disallowed Origin with 403; absent
# Origin is allowed so non-browser MCP clients keep working.
#
# Pure ASGI, no database and no adapter imports — the MCP server stays a pure
# HTTP proxy to the indexer.
#
# NOTE: this is transport hardening, NOT authentication. Tools still reach the
# indexer with whichever credential the per-tool policy resolves (the
# caller's own bearer, or TREELOOM_MCP_TOKEN, or — write/admin tools only,
# opt-in — the shared TREELOOM_MCP_API_KEY service account); see docs/authz.md.
from treeloom.application.mcp_origin_guard import (
    OriginGuardMiddleware,
    parse_allowed_origins,
)

logger = logging.getLogger(__name__)

app.add_middleware(
    OriginGuardMiddleware,
    allowed_origins=parse_allowed_origins(
        os.environ.get("TREELOOM_MCP_ALLOWED_ORIGINS")
    ),
)


class SearchRequest(BaseModel):
    query: str
    repo: str = ""
    top_k: int = 10


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    """REST API wrapper for MCP search_code tool.

    search_code() itself proxies to POST /search on the host indexer
    (configured via INDEXER_URL).  This endpoint delegates entirely to
    that proxy — no local retrieval, embedding, or graph logic runs here.
    """
    try:
        result = await search_code(
            req.query,
            top_k=req.top_k,
            path_prefix=req.repo if req.repo else None,
        )
        return {
            "chunks": [
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "snippet": c.snippet,
                    "score": c.score,
                }
                for c in result.chunks
            ]
            if result.chunks
            else []
        }
    except Exception as e:
        # The traceback used to go back in the response body: absolute source
        # paths, the dependency tree and its versions, and local variables in
        # the frame summary, handed to an unauthenticated caller of an
        # unauthenticated transport (CWE-209). It belongs in the server log.
        logger.exception("search failed")
        return {"error": str(e)}


@app.get("/sse")
async def handle_sse(request: Request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp._mcp_server.run(
            streams[0], streams[1], mcp._mcp_server.create_initialization_options()
        )


# Mount the SSE POST endpoint as a raw ASGI app rather than a FastAPI
# route. `sse.handle_post_message` is itself a complete ASGI app — it sends
# its own `202 Accepted` response. Wrapping it in `@app.post` made FastAPI
# send a *second* response after it returned, raising "Unexpected ASGI
# message ... after response already completed" and poisoning the keep-alive
# connection, so every client message after the first failed (-32602 /
# "Received request before initialization was complete").
app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))


if __name__ == "__main__":
    uvicorn.run(
        "treeloom.main:app",
        # Loopback by default (MCP spec: bind localhost when running
        # locally). Override with MCP_BIND_HOST for a container that
        # publishes the port itself.
        host=os.environ.get("MCP_BIND_HOST", "127.0.0.1"),
        port=int(os.environ.get("MCP_BIND_PORT", "8000")),
        reload=False,
    )
