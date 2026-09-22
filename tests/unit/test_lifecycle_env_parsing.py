"""Numeric lifecycle settings tolerate a blank value in .env.

`.env.example` ships ``TREELOOM_LOGIN_PURGE_RETENTION_SECONDS=`` (blank means
"use the derived default"). The dotenv loader in ``treeloom/__init__`` exports
that as an empty string, and ``int("")`` raised at module import — so a
checkout that copied the example env verbatim (CI, a fresh quickstart) could
not import the indexer at all.
"""
import pytest

from treeloom.application import lifecycle
from treeloom.application import routes_auth as rauth


_DERIVED_DEFAULT = max(rauth.LOGIN_WINDOW_SECONDS * 4, 3600)


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_retention_falls_back_to_derived_default(monkeypatch, raw):
    monkeypatch.setenv("TREELOOM_LOGIN_PURGE_RETENTION_SECONDS", raw)
    assert lifecycle._login_purge_retention() == _DERIVED_DEFAULT


def test_unset_retention_falls_back_to_derived_default(monkeypatch):
    monkeypatch.delenv("TREELOOM_LOGIN_PURGE_RETENTION_SECONDS", raising=False)
    assert lifecycle._login_purge_retention() == _DERIVED_DEFAULT


def test_explicit_retention_is_honoured(monkeypatch):
    monkeypatch.setenv("TREELOOM_LOGIN_PURGE_RETENTION_SECONDS", "86400")
    assert lifecycle._login_purge_retention() == 86400


def test_zero_retention_still_means_default(monkeypatch):
    """0 was already documented as 'derive it'; keep that contract."""
    monkeypatch.setenv("TREELOOM_LOGIN_PURGE_RETENTION_SECONDS", "0")
    assert lifecycle._login_purge_retention() == _DERIVED_DEFAULT
