import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or empty."""


def require_env(name: str, allow_empty: bool = False) -> str:
    val = os.environ.get(name)
    if val is None:
        raise ConfigError(
            f"Required environment variable {name!r} is not set. "
            f"Add it to .env (see .env.example) or export it before starting."
        )
    if not allow_empty and val == "":
        raise ConfigError(
            f"Required environment variable {name!r} is empty. "
            f"Set a non-empty value in .env."
        )
    return val


def require_int_env(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"Env var {name!r}={raw!r} is not a valid integer") from e


def required_settings() -> list[str]:
    """The env vars that must be set, conditioned on the selected backends.

    Backend selectors (VECTOR_STORE, GRAPH_STORE, EMBEDDING_PROVIDER,
    RERANKER_PROVIDER) decide which service credentials are mandatory —
    e.g. NEO4J_* only matter when GRAPH_STORE=neo4j. Evaluated at call
    time so profile defaults applied in treeloom/__init__ are honored.
    """
    req = ["EMBEDDING_MODEL", "VECTOR_DIM", "LLM_URL", "LLM_MODEL"]
    if os.environ.get("VECTOR_STORE", "milvus") == "milvus":
        req += ["MILVUS_HOST", "MILVUS_PORT"]
    if os.environ.get("GRAPH_STORE", "neo4j") == "neo4j":
        req += ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    if os.environ.get("EMBEDDING_PROVIDER", "tei") == "tei":
        req += ["EMBEDDING_URL"]
    elif not os.environ.get("EMBEDDING_API_KEY"):
        # the openai provider takes EMBEDDING_API_KEY or OPENAI_API_KEY
        req += ["OPENAI_API_KEY"]
    reranker_provider = os.environ.get("RERANKER_PROVIDER", "http")
    if reranker_provider == "http":
        req += ["RERANKER_URL"]
    elif reranker_provider in ("voyage", "cohere", "zeroentropy", "jina"):
        # hosted rerank APIs accept RERANKER_API_KEY or a provider-specific key
        if not (
            os.environ.get("RERANKER_API_KEY")
            or os.environ.get("VOYAGE_API_KEY")
            or os.environ.get("COHERE_API_KEY")
            or os.environ.get("ZEROENTROPY_API_KEY")
            or os.environ.get("JINA_API_KEY")
        ):
            req += ["RERANKER_API_KEY"]
    # local-st (sentence-transformers) needs no service/key — model from HF
    if os.environ.get("AUTH_ENABLED", "").lower() == "true":
        # Keys the HMAC that makes a stored credential hash verifiable. Unset,
        # user_store generates a random secret per process, which silently
        # invalidates every API key, PAT and session cookie on restart and
        # makes auth depend on which worker answers. Fail at startup instead
        # of degrading quietly. Auth-off keeps the ephemeral fallback — there
        # is no stored credential to verify.
        req += ["TREELOOM_HMAC_SECRET"]
    return req


# Back-compat snapshot (full-stack defaults at import time). Prefer
# required_settings() — this list does not react to env changes.
REQUIRED_SETTINGS = required_settings()


def validate_config() -> None:
    """Validate all required settings are present. Exits with structured JSON on failure."""
    missing = [s for s in required_settings() if s not in os.environ or os.environ[s] == ""]
    if missing:
        import json, sys
        print(json.dumps({"error": "missing_config", "missing": missing}), file=sys.stderr)
        raise ConfigError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set them in .env or export before starting."
        )


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Searches for ``treeloom.yaml`` relative to the package directory by default,
    or uses the explicitly provided *path*.

    Args:
        path: Optional explicit path to a YAML config file.  If not given,
              ``src/treeloom/treeloom.yaml`` relative to this file's location
              is used.

    Returns:
        Parsed configuration dictionary.  Returns an empty dict if the file
        is not found.

    Raises:
        ConfigError: If the YAML file exists but cannot be parsed.
    """
    if path is None:
        # Default path: src/treeloom/treeloom.yaml (next to this config module)
        default = Path(__file__).resolve().parent.parent / "treeloom.yaml"
        path = default
    else:
        path = Path(path)

    if not path.exists():
        return {}

    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at {path}: {exc}") from exc

    return data or {}


# ── Indexing constants ───────────────────────────────────────────────────────
# These are read by BOTH the service layer (application/indexer_service.py) and
# the runner leaf (application/indexer_runners.py). They lived only in the
# service, which the leaf may not import (the acyclic-layering rule enforced by
# tests/unit/test_indexer_state_wiring.py) and which nothing injected -- so every
# queued `repo` job died on NameError deep inside a worker. Defined here because
# config is the one module both layers are allowed to depend on.

LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "30"))
SKIP_FILENAMES = {"package-lock.json"}
FEATURE_SKIP_PATTERNS = os.environ.get("TREELOOM_FEATURE_SKIP_PATTERNS", "") == "1"
USE_SUMMARY_VECTOR = os.environ.get("USE_SUMMARY_VECTOR", "1") == "1"
OTEL_TRACE_PER_CHUNK = os.environ.get("OTEL_TRACE_PER_CHUNK", "").lower() == "true"
