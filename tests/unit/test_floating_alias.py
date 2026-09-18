"""Unit tests for floating-alias hygiene (internal-reranker-3 Phase 3).

Floating aliases (deepseek-chat, bare gpt-4o, *-latest) can be silently
remapped by the vendor — the 2026-06-25 `deepseek-chat` V3→v4-flash remap
invalidated a day of agentic verdicts. The agentic runner must refuse them by
default and stamp `floating_model_warning` into the summary when overridden.

Detroit-style: pure-function assertions, no services, no mocks of internals.
"""
from __future__ import annotations

import pytest

from treeloom.adapters.benchmark.llm_client import FLOATING_ALIASES, is_floating_alias
from treeloom.application.benchmark.agentic_runner import enforce_floating_model_policy


# ---------------------------------------------------------------- alias check

@pytest.mark.parametrize("model", [
    "deepseek-chat",
    "deepseek-reasoner",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-7-sonnet-latest",
    "chatgpt-4o-latest",
])
def test_floating_aliases_are_flagged(model):
    msg = is_floating_alias(model)
    assert msg is not None and model in msg


@pytest.mark.parametrize("model", [
    "gpt-4o-2024-08-06",       # the pinned judge — dated IDs must pass
    "gpt-4o-mini-2024-07-18",
    "deepseek-v4-pro",         # a specific deepseek model, not the alias
    "claude-sonnet-4-6",
    "qwen2.5-coder:7b",        # local tag
])
def test_dated_or_specific_ids_pass(model):
    assert is_floating_alias(model) is None


def test_none_and_empty_are_not_floating():
    assert is_floating_alias(None) is None
    assert is_floating_alias("") is None


def test_alias_check_is_case_and_whitespace_insensitive():
    assert is_floating_alias(" GPT-4o ") is not None
    assert is_floating_alias("DeepSeek-Chat") is not None


def test_message_names_dated_alternative_where_one_exists():
    msg = is_floating_alias("gpt-4o")
    assert "gpt-4o-2024-08-06" in msg
    msg = is_floating_alias("gpt-4o-mini")
    assert "gpt-4o-mini-2024-07-18" in msg


def test_deepseek_message_says_no_dated_ids_and_points_at_smoke_fixture():
    # DeepSeek publishes no dated IDs — the message must say so and direct the
    # user at the smoke fixture (the drift canary) instead of a dated pin.
    for m in ("deepseek-chat", "deepseek-reasoner"):
        msg = is_floating_alias(m)
        assert "no dated IDs" in msg
        assert "smoke fixture" in msg


def test_registry_contains_the_minimum_incident_set():
    for m in ("deepseek-chat", "deepseek-reasoner", "gpt-4o", "gpt-4o-mini"):
        assert m in FLOATING_ALIASES


# ---------------------------------------------------------- runner-level guard

@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("TREELOOM_ALLOW_FLOATING_MODELS", raising=False)


def test_guard_refuses_floating_agent_model(capsys):
    with pytest.raises(SystemExit) as exc:
        enforce_floating_model_policy("deepseek-chat", "gpt-4o-2024-08-06")
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "deepseek-chat" in out
    assert "refusing to start" in out


def test_guard_refuses_floating_judge_model_too(capsys):
    with pytest.raises(SystemExit) as exc:
        enforce_floating_model_policy("gpt-4o-2024-08-06", "gpt-4o")
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "judge model" in out
    assert "'gpt-4o'" in out  # names the offending alias, not the agent model


def test_guard_passes_dated_ids_silently():
    assert enforce_floating_model_policy(
        "gpt-4o-2024-08-06", "claude-sonnet-4-6"
    ) is False


def test_guard_allows_with_flag_and_returns_warning_true(capsys):
    warned = enforce_floating_model_policy(
        "deepseek-chat", "gpt-4o-2024-08-06", allow=True
    )
    assert warned is True  # caller stamps floating_model_warning: true
    out = capsys.readouterr().out
    assert "floating_model_warning" in out


def test_guard_allows_via_env_var(monkeypatch, capsys):
    monkeypatch.setenv("TREELOOM_ALLOW_FLOATING_MODELS", "1")
    assert enforce_floating_model_policy("deepseek-chat", None) is True


def test_guard_no_warning_flag_means_no_stamp():
    # Clean models return False → run_agentic never writes the summary key.
    assert enforce_floating_model_policy("deepseek-v4-pro", None) is False
