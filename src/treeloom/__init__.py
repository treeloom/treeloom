"""treeloom package — DDD architecture with backward-compatible legacy module access.

On import, best-effort load of a project-root `.env` so any process that
imports treeloom (uvicorn, REPL, scripts) sees the same vars docker-compose
substitutes from. Existing env vars are NOT overridden — compose-set values
inside the container always win.
"""
import os
from pathlib import Path


def _parse_dotenv_value(raw: str) -> str:
    """Unquote a .env value and drop any trailing comment.

    Order matters, and getting it backwards is what this fixes. The previous
    version tested `startswith(quote) and endswith(quote)` FIRST, so a line
    like

        AUTH_ENABLED="true"   # turn auth on

    failed that test (it ends in `n`, not `"`), fell through to the
    comment-stripping branch, and produced the value `"true"` — quotes
    included. `"true" != "true"` for every `== "true"` check in the codebase,
    so that line reads as auth OFF while looking exactly like auth on. Same
    shape silently breaks a quoted DATABASE_URL, an allow-list, or a key.

    So: find the closing quote first, and only then treat the remainder as a
    comment. An unquoted value keeps the old rule — `#` starts a comment only
    when preceded by whitespace, so `KEY=val#withhash` stays literal.
    """
    val = raw.strip()
    if val[:1] in ('"', "'"):
        quote = val[0]
        end = val.find(quote, 1)
        if end != -1:
            # Anything after the closing quote is comment or stray whitespace.
            return val[1:end]
        # Unbalanced opening quote — treat the whole thing as literal rather
        # than silently truncating to nothing.
        return val
    for sep in (" #", "\t#"):
        if sep in val:
            return val.split(sep, 1)[0].rstrip()
    return val


def _load_dotenv_if_present() -> None:
    # Escape hatch for tests that need a hermetic env (the loader walks up
    # from the package file, so cwd offers no isolation from the repo .env).
    if os.environ.get("TREELOOM_SKIP_DOTENV"):
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        envfile = parent / ".env"
        if envfile.is_file():
            try:
                text = envfile.read_text(encoding="utf-8")
            except OSError:
                return
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), _parse_dotenv_value(val))
            return


_load_dotenv_if_present()


def _apply_profile_defaults() -> None:
    """Apply deployment-profile defaults (TREELOOM_PROFILE=simple).

    The simple profile targets evaluators on limited hardware: embedded
    LanceDB + SQLite stores, one OpenAI key for embeddings + HyDE/summaries,
    and only postgres in Docker by default (docker-compose.simple.yml) — MCP
    access is stdio (no container); the deprecated HTTP+SSE `mcp-server`
    container is opt-in behind `COMPOSE_PROFILES=http-mcp`, same as full mode.
    Reranking defaults to a cloud provider when a rerank key is present (see
    _apply_simple_reranker_default — GPU-parity quality, no GPU), else the
    in-process CPU cross-encoder so the one-OpenAI-key path still works with
    nothing extra. Everything is `setdefault`, so any value set in .env or
    the shell wins. Runs before any adapter import — adapter modules read
    these vars at import time.
    """
    if os.environ.get("TREELOOM_PROFILE", "") != "simple":
        return
    defaults = {
        # backend selection
        "VECTOR_STORE": "lancedb",
        "GRAPH_STORE": "sqlite",
        "EMBEDDING_PROVIDER": "openai",
        # models — one OPENAI_API_KEY covers embeddings + HyDE/summaries
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "VECTOR_DIM": "1536",
        "LLM_URL": "https://api.openai.com/v1",
        "LLM_MODEL": "gpt-4o-mini",
        # api.openai.com 400s on the non-standard chat_template_kwargs param
        "LLM_COMPAT_CHAT_TEMPLATE_KWARGS": "0",
        # per-chunk LLM summaries are the dominant indexing cost — off by
        # default for evaluators (flip USE_SUMMARY_VECTOR=1 to enable; the
        # LanceDB store supports the summary hybrid leg)
        "USE_SUMMARY_VECTOR": "0",
        # single-machine scale + no observability stack
        "OTEL_SDK_DISABLED": "true",
        "AUTH_ENABLED": "false",
        "INDEX_CONCURRENCY": "2",
        "INDEX_FILE_CONCURRENCY": "4",
        # the postgres container from docker-compose.simple.yml
        "DATABASE_URL": "postgresql://treeloom:treeloom_pass@localhost:5432/treeloom",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)
    _apply_simple_reranker_default()


# Cloud rerank providers preferred when their key is present, best-measured
# first (Cohere rerank-v4.0-pro is the only reranker that beat the GPU bge
# control with per-query significance on featbit-clean-100). Each entry:
# (key env var, provider, recommended model). See docs/simple-mode.md.
_SIMPLE_RERANK_PREFERENCE = (
    ("COHERE_API_KEY", "cohere", "rerank-v4.0-pro"),
    ("VOYAGE_API_KEY", "voyage", "rerank-2.5"),
    ("ZEROENTROPY_API_KEY", "zeroentropy", "zerank-2"),
)


def _apply_simple_reranker_default() -> None:
    """Pick the simple-mode reranker: a cloud provider iff its key is set,
    else the local CPU cross-encoder. Preserves the one-OpenAI-key onboarding
    (no rerank key needed) while auto-upgrading to GPU-parity cloud quality
    the moment a rerank key is added. An explicit RERANKER_PROVIDER wins."""
    if os.environ.get("RERANKER_PROVIDER"):
        return
    for key_env, provider, model in _SIMPLE_RERANK_PREFERENCE:
        # a generic RERANKER_API_KEY can't name a provider on its own, so it
        # only takes effect alongside an explicit RERANKER_PROVIDER (handled
        # by the early return above)
        if os.environ.get(key_env):
            os.environ["RERANKER_PROVIDER"] = provider
            os.environ.setdefault("RERANKER_MODEL", model)
            return
    os.environ["RERANKER_PROVIDER"] = "local"


_apply_profile_defaults()

# Lazy re-export for backward compatibility — covers all legacy module paths.
# embedder/graph_store/retriever/reranker point at the root shim modules,
# which dispatch on EMBEDDING_PROVIDER/GRAPH_STORE/VECTOR_STORE/
# RERANKER_PROVIDER — so `from treeloom import X` and
# `from treeloom.X import fn` resolve to the same backend.
_LEGACY_MODULES = {
    "benchmark": "treeloom.application.benchmark",
    "community": "treeloom.adapters.graph.community_adapter",
    "config": "treeloom.infrastructure.config",
    "embedder": "treeloom.embedder",
    "graph_extractor": "treeloom.adapters.tree_sitter.graph_extractor",
    "graph_store": "treeloom.graph_store",
    "indexer": "treeloom.adapters.tree_sitter.indexer",
    "indexer_service": "treeloom.application.indexer_service",
    "llm": "treeloom.adapters.llm_api.llm_adapter",
    "main": "treeloom.application.main",
    "mcp_server": "treeloom.application.mcp_server",
    "models": "treeloom.domain.shared",
    "prompt_enhancer": "treeloom.domain.prompt_enhancer",
    "reranker": "treeloom.reranker",
    "retriever": "treeloom.retriever",
    "sources": "treeloom.domain.shared",
}

_legacy_imports: dict[str, object] = {}


def __getattr__(name: str):
    if name in _LEGACY_MODULES:
        if name not in _legacy_imports:
            import importlib

            mod = importlib.import_module(_LEGACY_MODULES[name])
            _legacy_imports[name] = mod
        return _legacy_imports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return [
        "benchmark",
        "community",
        "config",
        "embedder",
        "graph_extractor",
        "graph_store",
        "indexer",
        "indexer_service",
        "llm",
        "main",
        "mcp_server",
        "models",
        "prompt_enhancer",
        "retriever",
        "sources",
        "domain",
        "adapters",
        "application",
        "infrastructure",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "__path__",
        "__file__",
        "__doc__",
    ]