"""TREELOOM_HMAC_SECRET must be set when auth is on.

`hash_api_key` keys an HMAC with a process-level secret. Unset, the module
generated a random one per process and logged a warning. That warning
understates it: the secret is what makes a stored credential hash verifiable,
so an ephemeral secret means

  * every API key, PAT and session cookie in Postgres becomes unverifiable
    the moment the process restarts — users are logged out and service keys
    stop working, with nothing in the logs to connect cause to effect; and
  * with more than one indexer process (the documented deployment — workers
    share a Postgres job queue), a credential issued by worker A cannot be
    verified by worker B, so authentication succeeds or fails depending on
    which worker answers.

It fails closed, so it is not a bypass. It is a security-critical setting
that silently degrades instead of announcing itself, which is the class of
defect `validate_config()` exists to prevent. Auth-off (dev) keeps the
ephemeral fallback — there is nothing to verify.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from treeloom.infrastructure.config import ConfigError, required_settings, validate_config


def _hash_in_subprocess(env_extra: dict[str, str]) -> str:
    """Hash a fixed key in a fresh interpreter, so the module-level secret is
    generated from scratch — the thing a second worker process would do."""
    code = textwrap.dedent(
        """
        from treeloom.adapters.authorization.user_store import hash_api_key
        print(hash_api_key("the-same-api-key"))
        """
    )
    env = {**os.environ, **env_extra}
    env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


class TestWhyItMatters:
    """The failure the fix prevents, demonstrated rather than asserted."""

    def test_two_processes_disagree_without_the_secret(self):
        a = _hash_in_subprocess({"TREELOOM_HMAC_SECRET": ""})
        b = _hash_in_subprocess({"TREELOOM_HMAC_SECRET": ""})
        assert a != b, (
            "expected per-process random secrets to produce different hashes"
        )

    def test_two_processes_agree_with_it(self):
        env = {"TREELOOM_HMAC_SECRET": "a-shared-secret-value"}
        assert _hash_in_subprocess(env) == _hash_in_subprocess(env)


class TestItIsRequiredWhenAuthIsOn:
    def test_listed_as_required_when_auth_enabled(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        assert "TREELOOM_HMAC_SECRET" in required_settings()

    @pytest.mark.parametrize("value", ["false", "0", "", "TRUE_ISH"])
    def test_not_required_when_auth_is_off(self, monkeypatch, value):
        """Dev runs with auth off have no stored credential to verify, so the
        ephemeral fallback is correct there and must not become a hard stop."""
        monkeypatch.setenv("AUTH_ENABLED", value)
        assert "TREELOOM_HMAC_SECRET" not in required_settings()

    def test_required_check_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "TRUE")
        assert "TREELOOM_HMAC_SECRET" in required_settings()

    def test_startup_fails_loudly_when_auth_on_and_secret_missing(self, monkeypatch):
        for var in required_settings():
            monkeypatch.setenv(var, "set")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.delenv("TREELOOM_HMAC_SECRET", raising=False)
        with pytest.raises(ConfigError, match="TREELOOM_HMAC_SECRET"):
            validate_config()

    def test_startup_passes_once_it_is_set(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        for var in required_settings():
            monkeypatch.setenv(var, "set")
        validate_config()  # must not raise

    def test_empty_string_does_not_satisfy_it(self, monkeypatch):
        """An exported-but-empty var is the likeliest way to get this wrong."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        for var in required_settings():
            monkeypatch.setenv(var, "set")
        monkeypatch.setenv("TREELOOM_HMAC_SECRET", "")
        with pytest.raises(ConfigError, match="TREELOOM_HMAC_SECRET"):
            validate_config()
