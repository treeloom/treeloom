"""Simple profile — env defaults and the import-time landmine guard.

The landmine: any direct import of the milvus/neo4j/tei adapters from
startup-path modules crashes simple mode at import time (`require_env` on
MILVUS_HOST / NEO4J_URI / ...). Guarded here by importing the full indexer
service in a SUBPROCESS whose env contains no Milvus/Neo4j/TEI vars at all.
Subprocess because this pytest process already imported the dispatchers
under full-stack env.
"""
import logging
import os
import subprocess
import sys
import textwrap


def _clean_env(tmp_path) -> dict:
    """Minimal env: simple profile + a key, and explicitly NO service vars."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),  # ~/.treeloom lands in the tmp dir
        "TREELOOM_PROFILE": "simple",
        "OPENAI_API_KEY": "sk-test",
        # hermetic: without this the package dotenv loader pulls the repo
        # .env (full-stack vars) into the subprocess and masks landmines
        "TREELOOM_SKIP_DOTENV": "1",
        "GRAPH_DB_PATH": str(tmp_path / "graph.db"),
        "LANCEDB_PATH": str(tmp_path / "lancedb"),
    }
    return env


_PROFILE_PROBE = textwrap.dedent(
    """
    import json, os
    # simulate "no .env file": the repo .env would inject full-stack vars
    import treeloom
    treeloom  # the import above already ran _apply_profile_defaults()
    print(json.dumps({k: os.environ.get(k, "") for k in (
        "VECTOR_STORE", "GRAPH_STORE", "EMBEDDING_PROVIDER",
        "RERANKER_PROVIDER", "RERANKER_MODEL", "VECTOR_DIM", "LLM_MODEL",
        "AUTH_ENABLED",
    )}))
    """
)

_IMPORT_PROBE = textwrap.dedent(
    """
    import treeloom.application.indexer_service  # the whole startup surface
    import treeloom.retriever
    import treeloom.graph_store
    import treeloom.embedder
    import treeloom.reranker
    from treeloom.infrastructure.config import validate_config
    validate_config()
    print("OK")
    """
)


def _run(code: str, env: dict, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=120,
    )


def test_profile_defaults_applied(tmp_path):
    import json

    res = _run(_PROFILE_PROBE, _clean_env(tmp_path), cwd=str(tmp_path))
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got == {
        "VECTOR_STORE": "lancedb",
        "GRAPH_STORE": "sqlite",
        "EMBEDDING_PROVIDER": "openai",
        "RERANKER_PROVIDER": "local",
        "RERANKER_MODEL": "",
        "VECTOR_DIM": "1536",
        "LLM_MODEL": "gpt-4o-mini",
        "AUTH_ENABLED": "false",
    }


def test_explicit_env_beats_profile_default(tmp_path):
    import json

    env = _clean_env(tmp_path)
    env["VECTOR_STORE"] = "milvus"
    res = _run(_PROFILE_PROBE, env, cwd=str(tmp_path))
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got["VECTOR_STORE"] == "milvus"
    assert got["GRAPH_STORE"] == "sqlite"


def _probe_env(tmp_path) -> dict:
    import json

    def run(env):
        res = _run(_PROFILE_PROBE, env, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout.strip().splitlines()[-1])

    return run


def test_reranker_defaults_to_local_without_a_rerank_key(tmp_path):
    # the one-OpenAI-key path: no rerank key -> CPU cross-encoder, no regression
    got = _probe_env(tmp_path)(_clean_env(tmp_path))
    assert got["RERANKER_PROVIDER"] == "local"
    assert got["RERANKER_MODEL"] == ""


def test_reranker_auto_upgrades_to_cohere_when_key_present(tmp_path):
    env = _clean_env(tmp_path)
    env["COHERE_API_KEY"] = "co-test"
    got = _probe_env(tmp_path)(env)
    assert got["RERANKER_PROVIDER"] == "cohere"
    assert got["RERANKER_MODEL"] == "rerank-v4.0-pro"


def test_reranker_voyage_key_selects_voyage(tmp_path):
    env = _clean_env(tmp_path)
    env["VOYAGE_API_KEY"] = "vk-test"
    got = _probe_env(tmp_path)(env)
    assert got["RERANKER_PROVIDER"] == "voyage"
    assert got["RERANKER_MODEL"] == "rerank-2.5"


def test_cohere_preferred_over_voyage_when_both_keys_present(tmp_path):
    env = _clean_env(tmp_path)
    env["COHERE_API_KEY"] = "co"
    env["VOYAGE_API_KEY"] = "vk"
    got = _probe_env(tmp_path)(env)
    assert got["RERANKER_PROVIDER"] == "cohere"


def test_explicit_reranker_provider_beats_key_autoselect(tmp_path):
    env = _clean_env(tmp_path)
    env["COHERE_API_KEY"] = "co"
    env["RERANKER_PROVIDER"] = "local"
    got = _probe_env(tmp_path)(env)
    assert got["RERANKER_PROVIDER"] == "local"
    assert got["RERANKER_MODEL"] == ""  # not set when provider is explicit


def test_indexer_service_imports_without_milvus_neo4j_tei_env(tmp_path):
    """The landmine guard: full indexer import + validate_config() under a
    simple-mode env with NO MILVUS_*/NEO4J_*/EMBEDDING_URL/RERANKER_URL."""
    res = _run(_IMPORT_PROBE, _clean_env(tmp_path), cwd=str(tmp_path))
    assert res.returncode == 0, (
        f"simple-mode import crashed — a direct adapter import probably "
        f"reintroduced an import-time require_env.\nstderr:\n{res.stderr[-3000:]}"
    )
    assert res.stdout.strip().splitlines()[-1] == "OK"


def test_startup_warns_when_auth_disabled(monkeypatch, caplog):
    """_warn_if_auth_disabled() emits a WARNING when AUTH_ENABLED is not 'true'."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    from treeloom.application.lifecycle import _warn_if_auth_disabled

    with caplog.at_level(logging.WARNING):
        _warn_if_auth_disabled()

    assert "AUTH_ENABLED is false" in caplog.text


def test_startup_no_warning_when_auth_enabled(monkeypatch, caplog):
    """_warn_if_auth_disabled() is silent when AUTH_ENABLED=true."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    from treeloom.application.lifecycle import _warn_if_auth_disabled

    with caplog.at_level(logging.WARNING):
        _warn_if_auth_disabled()

    assert "AUTH_ENABLED is false" not in caplog.text
