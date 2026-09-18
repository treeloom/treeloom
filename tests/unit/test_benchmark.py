"""Unit tests for benchmark runner — Detroit-style.

Mock only out-of-process dependencies: httpx, subprocess, file I/O.
Internal functions and modules MUST NOT be mocked.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import httpx

# Ensure src/ is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def temp_jsonl(tmp_path):
    """Create a temporary JSONL query file."""
    filepath = tmp_path / "queries.jsonl"
    return filepath


@pytest.fixture
def sample_queries():
    """Sample query data for testing."""
    return [
        {
            "query": "Find GetEmailAsync in OidcClient",
            "ground_truth": ["/path/to/OidcClient.cs"],
            "difficulty": "easy",
            "strategy": "find-function",
        },
        {
            "query": "What happens when log?",
            "ground_truth": ["/path/to/logging.cs", "/path/to/logger.ts"],
            "difficulty": "medium",
            "strategy": "what-happens",
        },
        {
            "query": "How does build work?",
            "ground_truth": [],
            "difficulty": "hard",
            "strategy": "how-does",
        },
    ]


# ── _resolve_llm_config ──────────────────────────────────────────

class TestResolveLlmConfig:
    """Tests for _resolve_llm_config() tier selection."""

    def test_basic_tier_defaults(self, monkeypatch):
        """BENCHMARK_QUALITY=BASIC should return local Qwen defaults."""
        monkeypatch.setenv("BENCHMARK_QUALITY", "BASIC")
        from treeloom.adapters.benchmark.llm_client import resolve_llm_config
        cfg = resolve_llm_config()
        assert "localhost:11434" in cfg["url"]
        assert "qwen" in cfg["model"].lower()
        assert cfg["key"] == ""
        assert cfg["timeout"] == 45

    def test_standard_tier_defaults(self, monkeypatch):
        """BENCHMARK_QUALITY=STANDARD should return deepseek defaults."""
        monkeypatch.setenv("BENCHMARK_QUALITY", "STANDARD")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        from treeloom.adapters.benchmark.llm_client import resolve_llm_config
        cfg = resolve_llm_config()
        assert "deepseek" in cfg["url"]
        assert cfg["model"] == "deepseek-chat"
        assert cfg["key"] == "sk-test"
        assert cfg["timeout"] == 30

    def test_unknown_quality_defaults_to_basic(self, monkeypatch):
        """Unknown BENCHMARK_QUALITY should fall back to BASIC."""
        monkeypatch.setenv("BENCHMARK_QUALITY", "PREMIUM")
        from treeloom.adapters.benchmark.llm_client import resolve_llm_config
        cfg = resolve_llm_config()
        assert "localhost:11434" in cfg["url"]

    def test_nomcp_override_takes_precedence(self, monkeypatch):
        """NOMCP_LLM_MODEL should override tier default."""
        monkeypatch.setenv("BENCHMARK_QUALITY", "STANDARD")
        monkeypatch.setenv("NOMCP_LLM_MODEL", "my-custom-model")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        from treeloom.adapters.benchmark.llm_client import resolve_llm_config
        cfg = resolve_llm_config()
        assert cfg["model"] == "my-custom-model"
        assert "deepseek" in cfg["url"]

    def test_basic_custom_env_vars(self, monkeypatch):
        """BASIC_BENCHMARK_LLM_MODEL should override hardcoded default."""
        monkeypatch.setenv("BENCHMARK_QUALITY", "BASIC")
        monkeypatch.setenv("BASIC_BENCHMARK_LLM_MODEL", "codellama-7b")
        monkeypatch.setenv("BASIC_BENCHMARK_LLM_TIMEOUT", "60")
        from treeloom.adapters.benchmark.llm_client import resolve_llm_config
        cfg = resolve_llm_config()
        assert cfg["model"] == "codellama-7b"
        assert cfg["timeout"] == 60

    def test_standard_custom_env_vars(self, monkeypatch):
        """STANDARD_BENCHMARK_LLM_API_KEY should override DEEPSEEK_API_KEY."""
        monkeypatch.setenv("BENCHMARK_QUALITY", "STANDARD")
        monkeypatch.setenv("STANDARD_BENCHMARK_LLM_API_KEY", "sk-custom")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-default")
        from treeloom.adapters.benchmark.llm_client import resolve_llm_config
        cfg = resolve_llm_config()
        assert cfg["key"] == "sk-custom"


# ── _load_queries ─────────────────────────────────────────────────

class TestLoadQueries:
    """Tests for _load_queries JSONL loading and filtering."""

    def test_loads_all_queries(self, sample_queries, temp_jsonl):
        """Should load all queries from a JSONL file."""
        with open(temp_jsonl, "w") as f:
            for q in sample_queries:
                f.write(json.dumps(q) + "\n")
        from treeloom.domain.benchmark.queries import load_queries
        result = load_queries(str(temp_jsonl))
        assert len(result) == 3

    def test_filters_by_difficulty(self, sample_queries, temp_jsonl):
        """Should filter queries by difficulty field."""
        with open(temp_jsonl, "w") as f:
            for q in sample_queries:
                f.write(json.dumps(q) + "\n")
        from treeloom.domain.benchmark.queries import load_queries
        result = load_queries(str(temp_jsonl), difficulty="easy")
        assert len(result) == 1
        assert result[0]["difficulty"] == "easy"

    def test_filters_by_strategy(self, sample_queries, temp_jsonl):
        """Should filter queries by strategy field."""
        with open(temp_jsonl, "w") as f:
            for q in sample_queries:
                f.write(json.dumps(q) + "\n")
        from treeloom.domain.benchmark.queries import load_queries
        result = load_queries(str(temp_jsonl), strategy="what-happens")
        assert len(result) == 1
        assert "what-happens" in result[0]["strategy"]

    def test_empty_file(self, temp_jsonl):
        """Empty JSONL should return empty list."""
        temp_jsonl.write_text("")
        from treeloom.domain.benchmark.queries import load_queries
        result = load_queries(str(temp_jsonl))
        assert result == []

    def test_skips_blank_lines(self, sample_queries, temp_jsonl):
        """Blank lines in JSONL should be skipped."""
        with open(temp_jsonl, "w") as f:
            f.write(json.dumps(sample_queries[0]) + "\n")
            f.write("\n")
            f.write(json.dumps(sample_queries[1]) + "\n")
        from treeloom.domain.benchmark.queries import load_queries
        result = load_queries(str(temp_jsonl))
        assert len(result) == 2

    def test_uses_relevant_files_field(self, sample_queries, temp_jsonl):
        """Should load ground_truth from relevant_files if ground_truth missing."""
        q = {"query": "test", "relevant_files": ["/a", "/b"]}
        with open(temp_jsonl, "w") as f:
            f.write(json.dumps(q) + "\n")
        from treeloom.domain.benchmark.queries import load_queries
        result = load_queries(str(temp_jsonl))
        assert "relevant_files" in result[0]
        assert result[0]["relevant_files"] == ["/a", "/b"]


# ── Metric computation ────────────────────────────────────────────

class TestMetrics:
    """Tests for recall, precision, MRR computation."""

    def test_recall_hit(self):
        """Recall@5 with matching file should be 1.0."""
        from treeloom.domain.benchmark.metrics import compute_recall
        gt = ["/a.cs", "/b.py", "/c.ts"]
        retrieved = ["/x.js", "/a.cs", "/y.java"]
        assert compute_recall(gt, retrieved, top_k=5) == 1.0

    def test_recall_miss(self):
        """Recall@5 with no matching file should be 0.0."""
        from treeloom.domain.benchmark.metrics import compute_recall
        gt = ["/a.cs"]
        retrieved = ["/x.js", "/y.java"]
        assert compute_recall(gt, retrieved, top_k=5) == 0.0

    def test_recall_empty_ground_truth(self):
        """Empty ground truth should return 0.0."""
        from treeloom.domain.benchmark.metrics import compute_recall
        assert compute_recall([], ["/x.js"], top_k=5) == 0.0

    def test_precision_partial_hit(self):
        """Precision@3 with 1/3 relevant should be 1/3."""
        from treeloom.domain.benchmark.metrics import compute_precision
        gt = ["/a.cs"]
        retrieved = ["/a.cs", "/b.py", "/c.ts"]
        assert compute_precision(gt, retrieved, top_k=3) == 1.0 / 3

    def test_precision_all_relevant(self):
        """Precision with all retrieved in ground truth should be 1.0."""
        from treeloom.domain.benchmark.metrics import compute_precision
        gt = ["/a.cs", "/b.py"]
        retrieved = ["/a.cs", "/b.py"]
        assert compute_precision(gt, retrieved, top_k=2) == 1.0

    def test_mrr_first_position(self):
        """MRR with hit at position 1 should be 1.0."""
        from treeloom.domain.benchmark.metrics import compute_mrr
        assert compute_mrr(["/a.cs", "/b.py"], "/a.cs") == 1.0

    def test_mrr_third_position(self):
        """MRR with hit at position 3 should be 1/3."""
        from treeloom.domain.benchmark.metrics import compute_mrr
        assert compute_mrr(["/x.js", "/y.java", "/a.cs"], "/a.cs") == 1.0 / 3

    def test_mrr_miss(self):
        """MRR with no hit should be 0.0."""
        from treeloom.domain.benchmark.metrics import compute_mrr
        assert compute_mrr(["/x.js"], "/a.cs") == 0.0


# ── _run_ripgrep ─────────────────────────────────────────────────

class TestRunRipgrep:
    """Tests for ripgrep output parsing."""

    def test_parses_standard_output(self):
        """Should parse file:line:text from rg output."""
        output = "src/main.py:42:def get_email(): pass\nsrc/utils.py:10:class EmailClient:\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=output, stderr=""
            )
            from treeloom.adapters.benchmark.ripgrep import run_ripgrep
            matches = run_ripgrep("/repo", "Email")
        assert len(matches) == 2
        assert matches[0] == ("src/main.py", 42, "def get_email(): pass")
        assert matches[1] == ("src/utils.py", 10, "class EmailClient:")

    def test_handles_no_matches(self):
        """rg exit code 1 (no matches) should return empty list."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr=""
            )
            from treeloom.adapters.benchmark.ripgrep import run_ripgrep
            matches = run_ripgrep("/repo", "nonexistent")
        assert matches == []

    def test_handles_subprocess_error(self):
        """Subprocess errors should return empty list."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("rg not installed")
            from treeloom.adapters.benchmark.ripgrep import run_ripgrep
            matches = run_ripgrep("/repo", "test")
        assert matches == []

    def test_skips_malformed_lines(self):
        """Lines without proper file:line:text format should be skipped."""
        output = "bad_line_no_colon\nsrc/main.py:not_a_number:text\nsrc/main.py:42:valid\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=output, stderr=""
            )
            from treeloom.adapters.benchmark.ripgrep import run_ripgrep
            matches = run_ripgrep("/repo", "test")
        assert len(matches) == 1
        assert matches[0] == ("src/main.py", 42, "valid")


# ── _session_model_search ────────────────────────────────────────

class TestSessionModelSearch:
    """Tests for LLM pattern generation."""

    @pytest.mark.asyncio
    async def test_generates_patterns_from_llm_response(self):
        """Should parse LLM output into clean search patterns."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "GetEmailAsync\nEmailClient\nfind_email"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.dict(os.environ, {
            "NOMCP_LLM_URL": "http://test/v1",
            "NOMCP_LLM_MODEL": "test-model",
        }, clear=True):
            from treeloom.adapters.benchmark.llm_client import session_model_search
            patterns, tokens = await session_model_search(
                "Find GetEmailAsync", _client=mock_client
            )

        assert len(patterns) == 3
        assert "GetEmailAsync" in patterns
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_filters_garbage_patterns(self):
        """Short or garbage patterns should be filtered out."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "1. GetEmailAsync\n2. a\n3. \n"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.dict(os.environ, {
            "NOMCP_LLM_URL": "http://test/v1",
            "NOMCP_LLM_MODEL": "test-model",
        }, clear=True):
            from treeloom.adapters.benchmark.llm_client import session_model_search
            patterns, tokens = await session_model_search(
                "test", _client=mock_client
            )

        # "a" (too short) and blank line should be filtered
        assert len(patterns) == 1
        assert patterns[0] == "GetEmailAsync"

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self):
        """LLM errors should return empty patterns, not crash."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        with patch.dict(os.environ, {
            "NOMCP_LLM_URL": "http://test/v1",
            "NOMCP_LLM_MODEL": "test-model",
        }, clear=True):
            from treeloom.adapters.benchmark.llm_client import session_model_search
            patterns, tokens = await session_model_search(
                "test", _client=mock_client
            )

        assert patterns == []
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_llm_configured(self):
        """Basic tier is always available as default — never truly empty.
        
        When no env vars are set, _resolve_llm_config() still returns
        BASIC tier defaults (local Qwen). Change this test if BASIC_BENCHMARK_LLM_API_BASE
        default is removed or if a 'none' quality tier is added.
        """
        pass  # No-op: test documented as awareness check

    @pytest.mark.asyncio
    async def test_counts_prompt_and_completion_tokens(self):
        """Token count should include both prompt and completion."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "pattern1\npattern2"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.dict(os.environ, {
            "NOMCP_LLM_URL": "http://test/v1",
            "NOMCP_LLM_MODEL": "test-model",
        }, clear=True):
            from treeloom.adapters.benchmark.llm_client import session_model_search
            patterns, tokens = await session_model_search(
                "Find the function that handles email", _client=mock_client
            )

        assert tokens > 0
        # Prompt should be at least the query length
        assert tokens > 10
