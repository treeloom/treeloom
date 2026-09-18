"""CORS configuration for the indexer FastAPI app.

The operator UI v2 SPA is served from its own origin and calls the indexer
cross-origin. Cross-origin *credentialed* requests (the cookie-login path)
require an EXPLICIT allowed-origin list: browsers reject the wildcard ``"*"``
origin when credentials are included, so ``allow_origins=["*"]`` combined with
``allow_credentials=True`` is a spec violation that silently breaks cookie auth.
The bundled SPA sends ``credentials: "include"`` on EVERY request (bearer-token
mode included — see ``ui/src/api/client.ts``), so it needs an explicit origin
list in every auth mode; the wildcard default serves only non-credentialed
clients (curl, scripts, other origins without cookies).

``TREELOOM_CORS_ORIGINS`` (comma-separated exact origins) drives this:

* set   -> explicit origins + ``allow_credentials=True`` (cookie path works,
  and it is not wide-open).
* empty -> permissive fallback: ``allow_origins=["*"]`` with
  ``allow_credentials=False`` (spec-correct, but the bundled SPA is blocked —
  its credentialed requests fail CORS until origins are configured).

``.env.example`` sets ``http://localhost:3001`` (the UI's default origin) and
``run.sh`` defaults to ``http://localhost:${UI_PORT}`` when it is unset.

Cookie attributes for the secondary cross-origin cookie path
(SameSite=None; Secure) are handled separately in, not here.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_ALLOW_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_ALLOW_HEADERS = ["Authorization", "Content-Type"]


def parse_cors_origins(raw: str | None) -> list[str]:
    """Parse ``TREELOOM_CORS_ORIGINS`` into a list of exact origins.

    Comma-separated; surrounding whitespace stripped; empty entries dropped.
    ``None`` or empty -> ``[]``.
    """
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def add_cors_middleware(app: FastAPI, raw_origins: str | None) -> list[str]:
    """Apply the CORS middleware to *app* based on *raw_origins*.

    Returns the explicit origin list that was configured (``[]`` means the
    permissive default was used). See the module docstring for the
    credentialed-vs-wildcard rationale.
    """
    origins = parse_cors_origins(raw_origins)
    if "*" in origins:
        # A wildcard inside the EXPLICIT list took the credentialed branch
        # below, which pairs allow_origins=["*"] with allow_credentials=True.
        # Starlette then echoes the requesting origin and sets
        # Access-Control-Allow-Credentials: true, so every site on the web
        # gets credentialed cross-origin access to the API with the victim's
        # session cookie attached — precisely what the CORS spec forbids the
        # literal wildcard from doing, reintroduced by going through the
        # echo path (CWE-942).
        #
        # Downgrade to the permissive, credential-less default rather than
        # honouring it: the operator asked for "anyone", and that is the only
        # safe reading of it.
        import logging

        logging.getLogger(__name__).error(
            "TREELOOM_CORS_ORIGINS contains '*'; ignoring the explicit list and "
            "falling back to wildcard WITHOUT credentials. A wildcard cannot be "
            "combined with credentials — list exact origins to allow those."
        )
        origins = []
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=_ALLOW_METHODS,
            allow_headers=_ALLOW_HEADERS,
        )
    else:
        # No explicit origins: permissive dev default, but spec-correct --
        # a wildcard origin must not be combined with credentials.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    return origins
