"""PREFLIGHT_ALLOWED_ROOTS scoping on the local indexing endpoints.

`/preflight` already refused a path outside PREFLIGHT_ALLOWED_ROOTS, but
`/index-repo`, `/index-file` and `/index-directory` did not — so a caller who
could reach the indexer could have it read any path the service user can read
and then retrieve the contents back out through `/search` (CWE-22).

The guard is deliberately **opt-in**: an empty allow-list is the shipped
default, and failing closed the way `/preflight` does would 403 every ordinary
"index this repo" call. What keeps these endpoints off the network by default
is the loopback INDEXER_HOST bind plus AUTH_ENABLED; this narrows the blast
radius for operators who configure roots, and closes the asymmetry where one
endpoint validated a path and three did not.

PREFLIGHT_ALLOWED_ROOTS is resolved from the environment at import time, so
these tests patch the module attribute rather than the env var.
"""

import os

import pytest
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException
from fastapi.testclient import TestClient

from treeloom.application import indexer_service as svc


def _set_roots(monkeypatch, roots):
    """Patch the module-level allow-list (computed at import, not per call)."""
    monkeypatch.setattr(
        _authz, "PREFLIGHT_ALLOWED_ROOTS", [os.path.realpath(r) for r in roots]
    )


class TestUnconfiguredIsUnchanged:
    """The default ships with no roots. That path must keep working, or every
    ordinary index call turns into a 403."""

    def test_no_roots_allows_any_path(self, monkeypatch):
        _set_roots(monkeypatch, [])
        # No raise, for a path nobody would ever whitelist.
        assert _authz._require_path_in_scope("/etc/passwd") is None

    def test_no_roots_allows_a_traversal_string_too(self, monkeypatch):
        _set_roots(monkeypatch, [])
        assert _authz._require_path_in_scope("/srv/repos/../../etc/shadow") is None


class TestConfiguredScoping:
    def test_path_inside_a_root_is_allowed(self, monkeypatch, tmp_path):
        _set_roots(monkeypatch, [str(tmp_path)])
        inside = tmp_path / "myrepo" / "src"
        inside.mkdir(parents=True)
        assert _authz._require_path_in_scope(str(inside)) is None

    def test_the_root_itself_is_allowed(self, monkeypatch, tmp_path):
        _set_roots(monkeypatch, [str(tmp_path)])
        assert _authz._require_path_in_scope(str(tmp_path)) is None

    def test_path_outside_every_root_is_refused(self, monkeypatch, tmp_path):
        _set_roots(monkeypatch, [str(tmp_path / "allowed")])
        (tmp_path / "allowed").mkdir()
        with pytest.raises(HTTPException) as exc:
            _authz._require_path_in_scope("/etc/passwd")
        assert exc.value.status_code == 403
        assert "PREFLIGHT_ALLOWED_ROOTS" in str(exc.value.detail)

    def test_dotdot_traversal_out_of_a_root_is_refused(self, monkeypatch, tmp_path):
        """realpath() collapses `..` before the prefix test, so escaping by
        traversal is caught rather than passing a naive startswith()."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        _set_roots(monkeypatch, [str(allowed)])
        with pytest.raises(HTTPException) as exc:
            _authz._require_path_in_scope(str(allowed / ".." / "elsewhere"))
        assert exc.value.status_code == 403

    def test_symlink_escaping_a_root_is_refused(self, monkeypatch, tmp_path):
        """A symlink inside an allowed root pointing out of it must not
        smuggle the target in — realpath() resolves it first."""
        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        link = allowed / "escape"
        link.symlink_to(outside, target_is_directory=True)
        _set_roots(monkeypatch, [str(allowed)])
        with pytest.raises(HTTPException) as exc:
            _authz._require_path_in_scope(str(link))
        assert exc.value.status_code == 403

    def test_sibling_sharing_a_prefix_is_refused(self, monkeypatch, tmp_path):
        """`/srv/allowed-evil` must not pass a check for `/srv/allowed`. This
        is what the `root + os.sep` boundary buys over a bare startswith()."""
        allowed = tmp_path / "allowed"
        sibling = tmp_path / "allowed-evil"
        allowed.mkdir()
        sibling.mkdir()
        _set_roots(monkeypatch, [str(allowed)])
        with pytest.raises(HTTPException) as exc:
            _authz._require_path_in_scope(str(sibling))
        assert exc.value.status_code == 403

    def test_refusal_does_not_depend_on_the_path_existing(self, monkeypatch, tmp_path):
        """403 for an out-of-scope path whether or not it exists — otherwise
        the 403/404 split is itself a filesystem oracle."""
        _set_roots(monkeypatch, [str(tmp_path)])
        with pytest.raises(HTTPException) as exc:
            _authz._require_path_in_scope("/definitely/not/here/at/all")
        assert exc.value.status_code == 403


class TestEndpointsEnforceTheGuard:
    """All three local-path endpoints must refuse, and must refuse BEFORE the
    existence check so the status code leaks nothing."""

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTH_ENABLED", "false")
        _set_roots(monkeypatch, [str(tmp_path / "allowed")])
        (tmp_path / "allowed").mkdir()
        return TestClient(svc.app, raise_server_exceptions=False)

    def test_index_file_refuses_out_of_scope(self, client):
        resp = client.post("/index-file", json={"file_path": "/etc/passwd"})
        assert resp.status_code == 403

    def test_index_directory_refuses_out_of_scope(self, client):
        resp = client.post("/index-directory", json={"directory": "/etc"})
        assert resp.status_code == 403

    def test_index_repo_refuses_out_of_scope(self, client):
        resp = client.post("/index-repo", json={"path": "/etc"})
        assert resp.status_code == 403

    def test_403_not_404_for_a_nonexistent_out_of_scope_path(self, client):
        """The scope check runs first, so a probe cannot distinguish
        'exists but forbidden' from 'does not exist'."""
        resp = client.post(
            "/index-file", json={"file_path": "/etc/definitely-not-here.py"}
        )
        assert resp.status_code == 403

    def test_index_repo_url_mode_reaches_the_url_validator_not_the_path_guard(
        self, client
    ):
        """Only local paths are scoped. A `url` must still be judged by
        _is_safe_git_url, so a remote-helper smuggling attempt returns its 400
        rather than being pre-empted (or masked) by the path guard's 403.

        `ext::` is used because _is_safe_git_url is otherwise very permissive —
        it rejects only empty strings, a leading '-', and '::'. That laxity is
        its own open finding; this test pins the routing, not the URL policy.
        """
        resp = client.post("/index-repo", json={"url": "ext::sh -c id"})
        assert resp.status_code == 400
        assert resp.status_code != 403
