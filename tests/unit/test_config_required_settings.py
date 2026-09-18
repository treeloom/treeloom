"""required_settings() matrix — which env vars each backend combo demands."""
import pytest

from treeloom.infrastructure.config import required_settings

ALWAYS = {"EMBEDDING_MODEL", "VECTOR_DIM", "LLM_URL", "LLM_MODEL"}


def _req(monkeypatch, **env) -> set[str]:
    for var in (
        "VECTOR_STORE", "GRAPH_STORE", "EMBEDDING_PROVIDER",
        "RERANKER_PROVIDER", "EMBEDDING_API_KEY",
        "RERANKER_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY",
        "ZEROENTROPY_API_KEY", "JINA_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return set(required_settings())


def test_default_full_stack(monkeypatch):
    req = _req(monkeypatch)
    assert ALWAYS <= req
    assert {"MILVUS_HOST", "MILVUS_PORT"} <= req
    assert {"NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"} <= req
    assert "EMBEDDING_URL" in req
    assert "RERANKER_URL" in req
    assert "OPENAI_API_KEY" not in req


def test_simple_mode_combo(monkeypatch):
    req = _req(
        monkeypatch,
        VECTOR_STORE="lancedb",
        GRAPH_STORE="sqlite",
        EMBEDDING_PROVIDER="openai",
        RERANKER_PROVIDER="local",
    )
    assert ALWAYS <= req
    assert "OPENAI_API_KEY" in req
    for var in ("MILVUS_HOST", "MILVUS_PORT", "NEO4J_URI", "NEO4J_USER",
                "NEO4J_PASSWORD", "EMBEDDING_URL", "RERANKER_URL"):
        assert var not in req


def test_voyage_reranker_requires_api_key(monkeypatch):
    req = _req(monkeypatch, RERANKER_PROVIDER="voyage")
    assert "RERANKER_API_KEY" in req
    assert "RERANKER_URL" not in req


def test_voyage_key_satisfied_by_provider_specific_var(monkeypatch):
    req = _req(monkeypatch, RERANKER_PROVIDER="voyage", VOYAGE_API_KEY="vk")
    assert "RERANKER_API_KEY" not in req


def test_embedding_api_key_substitutes_for_openai_key(monkeypatch):
    req = _req(monkeypatch, EMBEDDING_PROVIDER="openai", EMBEDDING_API_KEY="k")
    assert "OPENAI_API_KEY" not in req
    assert "EMBEDDING_URL" not in req


def test_partial_combo_sqlite_graph_only(monkeypatch):
    """Mixing backends only relaxes the swapped-out service's vars."""
    req = _req(monkeypatch, GRAPH_STORE="sqlite")
    assert {"MILVUS_HOST", "MILVUS_PORT", "EMBEDDING_URL", "RERANKER_URL"} <= req
    assert "NEO4J_URI" not in req


def test_validate_config_uses_dynamic_list(monkeypatch):
    """validate_config evaluates required_settings() at call time."""
    from treeloom.infrastructure.config import ConfigError, validate_config

    _req(monkeypatch, GRAPH_STORE="sqlite")  # clears selector vars
    for var in ("EMBEDDING_MODEL", "VECTOR_DIM", "LLM_URL", "LLM_MODEL",
                "MILVUS_HOST", "MILVUS_PORT", "EMBEDDING_URL", "RERANKER_URL"):
        monkeypatch.setenv(var, "x")
    validate_config()  # NEO4J_* not required with GRAPH_STORE=sqlite

    monkeypatch.setenv("GRAPH_STORE", "neo4j")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(ConfigError):
        validate_config()
