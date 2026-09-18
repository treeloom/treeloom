"""Origin validation for the MCP HTTP transport (DNS-rebinding defence).

The MCP specification's transport security rules are normative for HTTP-based
transports:

    1. Servers **MUST** validate the `Origin` header on all incoming
       connections to prevent DNS rebinding attacks. If the `Origin` header is
       present and invalid, servers **MUST** respond with HTTP 403 Forbidden.
    2. When running locally, servers **SHOULD** bind only to localhost
       (127.0.0.1) rather than all network interfaces (0.0.0.0).

Rule 2 is handled by the compose port binding. This module implements rule 1.

Why it matters here: a browser on any website can issue cross-origin requests
to `http://localhost:8000`. Without an Origin check, a malicious page can drive
the MCP transport in the victim's network context — the browser attaches no
credential, but this server needs none. That is the DNS-rebinding scenario the
spec calls out, and it is reachable even when the port is bound to loopback.

Absent `Origin` is ALLOWED: non-browser MCP clients (stdio bridges, CLI tools,
`curl`) send no Origin, and the spec only mandates rejection when the header is
"present and invalid". Browsers always send it for cross-origin requests, which
is exactly the case being defended against.

Configuration — ``TREELOOM_MCP_ALLOWED_ORIGINS`` (comma-separated exact
origins), mirroring ``TREELOOM_CORS_ORIGINS`` on the indexer:

* set   -> only those exact origins are accepted.
* empty -> loopback-only default (``localhost``, ``127.0.0.1``, ``[::1]`` on
  any port/scheme), matching the loopback port binding.

This module is deliberately dependency-free: no database, no adapters, no
credential handling. The MCP server is a pure HTTP proxy to the indexer and
must stay that way, so an origin check is expressed as a pure predicate plus a
minimal ASGI wrapper.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Hostnames that resolve to the loopback interface. Used for the default
# allow-list when no explicit origins are configured.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Parse ``TREELOOM_MCP_ALLOWED_ORIGINS`` into a list of exact origins.

    Comma-separated; surrounding whitespace stripped; empty entries dropped.
    ``None`` or empty -> ``[]`` (meaning: use the loopback-only default).
    """
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def is_loopback_origin(origin: str) -> bool:
    """True when *origin* names the loopback interface on any port/scheme."""
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if not parts.scheme or not parts.netloc:
        return False
    try:
        host = parts.hostname
    except ValueError:
        # Malformed IPv6 literal, bracket mismatch, etc.
        return False
    return host is not None and host.lower() in _LOOPBACK_HOSTS


def is_origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """Decide whether a request carrying *origin* may proceed.

    * ``None`` (header absent) -> allowed; the spec only requires rejecting an
      Origin that is "present and invalid", and non-browser clients send none.
    * *allowed* non-empty -> exact string match against the configured list.
    * *allowed* empty -> loopback origins only.
    """
    if origin is None:
        return True
    if allowed:
        return origin in allowed
    return is_loopback_origin(origin)


class OriginGuardMiddleware:
    """Pure-ASGI middleware rejecting disallowed ``Origin`` headers with 403.

    Implemented as raw ASGI rather than Starlette's ``BaseHTTPMiddleware``
    on purpose: ``BaseHTTPMiddleware`` buffers the response through an
    intermediate queue, which breaks the long-lived SSE stream on ``/sse``.
    This wrapper inspects the request scope and either short-circuits or
    delegates untouched, so streaming is unaffected.
    """

    def __init__(self, app, allowed_origins: list[str] | None = None):
        self.app = app
        self._allowed = list(allowed_origins or [])

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        origin: str | None = None
        for name, value in scope.get("headers") or []:
            if name == b"origin":
                origin = value.decode("latin-1")
                break

        if not is_origin_allowed(origin, self._allowed):
            await _send_403(send)
            return

        await self.app(scope, receive, send)


async def _send_403(send) -> None:
    """Emit a minimal 403. Body is a JSON-RPC error with no ``id``, which the
    spec permits for a rejected connection."""
    body = (
        b'{"jsonrpc":"2.0","error":{"code":-32600,'
        b'"message":"Origin not allowed"},"id":null}'
    )
    await send(
        {
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
