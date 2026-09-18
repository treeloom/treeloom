"""Four findings from the SAST pass, all of them "the handler never asked".

* /jobs, /jobs/{id}, /jobs/{id}/errors and /status returned the whole job
  record to anyone. The auth middleware exempts the `/jobs` PREFIX, so the
  per-file error log — absolute paths paired with raw exception text — was
  the most disclosive endpoint on the service and the least protected
  (CWE-306 / CWE-209).
* POST /index-graph checked only that the caller was authenticated, so any
  principal could drop and rebuild any source's graph (CWE-862).
* POST /build-community took no `request` parameter at all, so any
  authenticated principal could start a fleet-wide rebuild (CWE-862).
* The session cookie's Secure attribute defaulted to off (CWE-614).

The redaction design point: these progress endpoints are *documented* as
reachable without a credential, so closing them would be a breaking change
aimed at the wrong thing. What leaked was the payload, not the reachability —
and only when auth is on at all. An install with AUTH_ENABLED unset behaves
exactly as before.
"""

import pytest
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


@pytest.fixture
def job_record():
    return {
        "id": "job-1",
        "job_id": "job-1",
        "kind": "repo",
        "status": "running",
        "source": "/home/alice/private/secret-repo",
        "source_id": "src-1",
        "current_file": "/home/alice/private/secret-repo/creds.py",
        "total_files": 10,
        "processed_files": 3,
        "total_chunks": 42,
        "error": "Traceback (most recent call last): FileNotFoundError: /etc/shadow",
        "message": "boom",
        "last_log_time": 123.0,
        "start_time": 100.0,
    }


class TestJobRedaction:
    def test_progress_survives_redaction(self, job_record):
        """The endpoints exist to report progress; that must still work."""
        out = _authz._public_job(job_record, redact=True)
        assert out["status"] == "running"
        assert out["processed_files"] == 3
        assert out["total_files"] == 10
        assert out["total_chunks"] == 42
        assert out["job_id"] == "job-1"

    def test_location_and_failure_detail_are_stripped(self, job_record):
        out = _authz._public_job(job_record, redact=True)
        assert out["source"] is None
        assert out["current_file"] is None
        assert out["error"] is None
        assert out["message"] is None

    def test_no_sensitive_value_survives_anywhere_in_the_payload(self, job_record):
        """Field-by-field assertions miss a field added later. This fails if
        any secret string reaches the response by any route."""
        out = _authz._public_job(job_record, redact=True)
        blob = repr(out)
        for leaked in ("/home/alice", "secret-repo", "creds.py", "Traceback", "/etc/shadow"):
            assert leaked not in blob, f"{leaked!r} still present"

    def test_redaction_is_advertised(self, job_record):
        """A client seeing nulls should be able to tell them from real nulls."""
        assert _authz._public_job(job_record, redact=True)["redacted"] is True
        assert "redacted" not in _authz._public_job(job_record, redact=False)

    def test_unredacted_view_is_unchanged(self, job_record):
        out = _authz._public_job(job_record, redact=False)
        assert out["source"] == "/home/alice/private/secret-repo"
        assert out["error"].startswith("Traceback")
        assert "last_log_time" not in out, "still dropped, as before"


class TestWhoGetsRedacted:
    @pytest.mark.asyncio
    async def test_auth_disabled_never_redacts(self, mocker, monkeypatch):
        """An install that has not enabled auth is untouched by this change."""
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        assert await _authz._job_view_is_redacted(mocker.Mock()) is False

    @pytest.mark.asyncio
    async def test_authenticated_caller_sees_everything(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.object(
            _authz, "_require_authenticated_user",
            new=mocker.AsyncMock(return_value=mocker.Mock()),
        )
        assert await _authz._job_view_is_redacted(mocker.Mock()) is False

    @pytest.mark.asyncio
    async def test_anonymous_caller_is_redacted(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.object(
            _authz, "_require_authenticated_user",
            new=mocker.AsyncMock(side_effect=HTTPException(401, "nope")),
        )
        assert await _authz._job_view_is_redacted(mocker.Mock()) is True

    @pytest.mark.asyncio
    async def test_a_401_is_redaction_not_a_500(self, mocker, monkeypatch):
        """_require_authenticated_user raises rather than returning None for a
        missing credential. Letting that propagate would turn a documented
        open endpoint into an error for exactly the callers it is meant for."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.object(
            _authz, "_require_authenticated_user",
            new=mocker.AsyncMock(side_effect=HTTPException(401, "nope")),
        )
        # Must not raise.
        assert await _authz._job_view_is_redacted(mocker.Mock()) is True


class TestCookieSecure:
    """Defaulted to off, so any deployment that did not think about it served
    a session cookie browsers send over plain HTTP."""

    def _req(self, mocker, scheme, forwarded=None):
        r = mocker.Mock()
        r.url.scheme = scheme
        r.headers = {"x-forwarded-proto": forwarded} if forwarded else {}
        return r

    def test_auto_sets_secure_on_https(self, mocker, monkeypatch):
        monkeypatch.setattr(_authz, "_COOKIE_SECURE_MODE", "auto")
        assert _authz._cookie_secure(self._req(mocker, "https")) is True

    def test_auto_leaves_local_http_dev_working(self, mocker, monkeypatch):
        monkeypatch.setattr(_authz, "_COOKIE_SECURE_MODE", "auto")
        assert _authz._cookie_secure(self._req(mocker, "http")) is False

    def test_forwarded_proto_ignored_unless_proxy_is_trusted(self, mocker, monkeypatch):
        """Untrusted, a client could send `X-Forwarded-Proto: http` and strip
        Secure from its own cookie."""
        monkeypatch.setattr(_authz, "_COOKIE_SECURE_MODE", "auto")
        monkeypatch.setattr(_authz, "LOGIN_TRUST_FORWARDED_FOR", False)
        req = self._req(mocker, "https", forwarded="http")
        assert _authz._cookie_secure(req) is True, "header must not downgrade"

    def test_forwarded_proto_honoured_behind_a_trusted_proxy(self, mocker, monkeypatch):
        monkeypatch.setattr(_authz, "_COOKIE_SECURE_MODE", "auto")
        monkeypatch.setattr(_authz, "LOGIN_TRUST_FORWARDED_FOR", True)
        req = self._req(mocker, "http", forwarded="https")
        assert _authz._cookie_secure(req) is True, "TLS terminated at the proxy"

    def test_forwarded_proto_takes_the_first_hop(self, mocker, monkeypatch):
        monkeypatch.setattr(_authz, "_COOKIE_SECURE_MODE", "auto")
        monkeypatch.setattr(_authz, "LOGIN_TRUST_FORWARDED_FOR", True)
        req = self._req(mocker, "http", forwarded="https, http")
        assert _authz._cookie_secure(req) is True

    @pytest.mark.parametrize("mode,expected", [("1", True), ("0", False)])
    def test_explicit_settings_still_win(self, mocker, monkeypatch, mode, expected):
        monkeypatch.setattr(_authz, "_COOKIE_SECURE_MODE", mode)
        assert _authz._cookie_secure(self._req(mocker, "http")) is expected
        assert _authz._cookie_secure(self._req(mocker, "https")) is expected


class TestGraphRebuildAuthorization:
    @pytest.mark.asyncio
    async def test_index_graph_requires_the_index_scope(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"search"}
        authorize = mocker.patch.object(
            _authz, "_authorize_scope", new=mocker.AsyncMock(return_value=None)
        )
        with pytest.raises(HTTPException) as exc:
            await svc.handle_index_graph(
                svc.IndexGraphRequest(source_id="victim"), request
            )
        assert exc.value.status_code == 403
        authorize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_index_graph_consults_the_per_source_acl(self, mocker, monkeypatch):
        """Authentication alone was the entire check, so any principal could
        name any source_id. delete_source() runs before the rebuild, so this
        damages the target's graph whether or not the rebuild succeeds."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = mocker.Mock()
        request.state.scopes = {"index"}
        mocker.patch.object(
            _authz, "_authorize_scope",
            new=mocker.AsyncMock(side_effect=HTTPException(403, "denied")),
        )
        repo = mocker.patch.object(_state, "_source_repo")
        with pytest.raises(HTTPException) as exc:
            await svc.handle_index_graph(
                svc.IndexGraphRequest(source_id="victim"), request
            )
        assert exc.value.status_code == 403
        repo.get_by_id.assert_not_called(), "must refuse before touching the source"

    @pytest.mark.asyncio
    async def test_build_community_is_admin_only(self, mocker, monkeypatch):
        """It took no `request` at all. One unprivileged POST scheduled roughly
        20 seconds of work per source across every source in the install, and
        the single-flight guard let the caller hold the slot against the
        operator."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = mocker.Mock(role=svc.Role.USER)
        request.state.scopes = {"index"}
        create_task = mocker.patch.object(svc.asyncio, "create_task")
        with pytest.raises(HTTPException) as exc:
            await svc.handle_build_community(request)
        assert exc.value.status_code == 403
        create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_community_still_runs_for_an_admin(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        request = mocker.Mock()
        request.state.user = mocker.Mock(role=svc.Role.ADMIN)
        request.state.scopes = {"admin", "index", "search"}
        mocker.patch.object(_state, "_community_build_task", None)
        create_task = mocker.patch.object(svc.asyncio, "create_task")
        out = await svc.handle_build_community(request)
        assert out["status"] == "started"
        create_task.assert_called_once()


class TestJobListingIsScopedToTheCaller:
    """findings 45/46/47 CWE-639. /jobs returned every job in the process to every
    caller, and /job-groups rolled the same jobs up by another route. A
    non-admin could enumerate the absolute paths and clone URLs of every other
    principal's sources, plus their failure messages.

    Jobs carry no owner column, so there is nothing to compare a caller
    against directly. They do carry source_id, and per-source authorization
    already exists — reusing it avoids both a migration and a second
    permission model to keep in step with the first.
    """

    @pytest.fixture
    def two_sources(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.delenv("TREELOOM_SEARCH_OPEN", raising=False)
        mocker.patch.dict(_state._jobs, {
            "j-mine": {"id": "j-mine", "job_id": "j-mine", "status": "done",
                       "source_id": "mine", "source": "/home/me/repo"},
            "j-theirs": {"id": "j-theirs", "job_id": "j-theirs", "status": "done",
                         "source_id": "theirs", "source": "/home/them/secret"},
        }, clear=True)
        user = mocker.Mock(id="u1", role=svc.Role.USER)
        mocker.patch.object(
            _authz, "_require_authenticated_user", new=mocker.AsyncMock(return_value=user)
        )
        authz = mocker.Mock()

        async def _authorize(u, sid):
            return mocker.Mock(allowed=(sid == "mine"))

        authz.authorize_source = _authorize
        mocker.patch.object(_authz, "_authz", return_value=authz)
        return user

    @pytest.mark.asyncio
    async def test_list_jobs_hides_another_principals_job(self, mocker, two_sources):
        out = await svc.list_jobs(mocker.Mock())
        assert [j["job_id"] for j in out] == ["j-mine"]

    @pytest.mark.asyncio
    async def test_get_job_is_404_not_403_for_a_hidden_job(self, mocker, two_sources):
        """403 would confirm the id exists, which is the thing being hidden."""
        with pytest.raises(HTTPException) as exc:
            await svc.get_job("j-theirs", mocker.Mock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_own_job_still_readable(self, mocker, two_sources):
        assert (await svc.get_job("j-mine", mocker.Mock()))["job_id"] == "j-mine"

    @pytest.mark.asyncio
    async def test_a_job_with_no_source_is_not_shown(self, mocker, two_sources):
        """An unattributable job cannot be authorized, so it is not the
        caller's to see — fail closed rather than default-visible."""
        _state._jobs["j-orphan"] = {"id": "j-orphan", "job_id": "j-orphan",
                                 "status": "done", "source_id": None}
        out = await svc.list_jobs(mocker.Mock())
        assert "j-orphan" not in [j["job_id"] for j in out]

    @pytest.mark.asyncio
    async def test_admin_sees_everything(self, mocker, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.dict(_state._jobs, {
            "a": {"id": "a", "job_id": "a", "source_id": "s1", "status": "done"},
            "b": {"id": "b", "job_id": "b", "source_id": "s2", "status": "done"},
        }, clear=True)
        mocker.patch.object(
            _authz, "_require_authenticated_user",
            new=mocker.AsyncMock(return_value=mocker.Mock(role=svc.Role.ADMIN)),
        )
        out = await svc.list_jobs(mocker.Mock())
        assert {j["job_id"] for j in out} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_enforcement_off_is_unchanged(self, mocker, monkeypatch):
        """Open mode must keep listing everything, or this becomes a
        behaviour change for every install that has not enabled auth."""
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        mocker.patch.dict(_state._jobs, {
            "a": {"id": "a", "job_id": "a", "source_id": "s1", "status": "done"},
            "b": {"id": "b", "job_id": "b", "source_id": "s2", "status": "done"},
        }, clear=True)
        out = await svc.list_jobs(mocker.Mock())
        assert {j["job_id"] for j in out} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_source_decisions_are_cached_per_request(self, mocker, monkeypatch):
        """A listing is many jobs over few sources; without the cache this is
        one authorization round-trip per job."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.dict(_state._jobs, {
            f"j{i}": {"id": f"j{i}", "job_id": f"j{i}", "source_id": "same",
                      "status": "done"}
            for i in range(25)
        }, clear=True)
        mocker.patch.object(
            _authz, "_require_authenticated_user",
            new=mocker.AsyncMock(return_value=mocker.Mock(id="u1", role=svc.Role.USER)),
        )
        authz = mocker.Mock()
        authz.authorize_source = mocker.AsyncMock(
            return_value=mocker.Mock(allowed=True)
        )
        mocker.patch.object(_authz, "_authz", return_value=authz)

        out = await svc.list_jobs(mocker.Mock())
        assert len(out) == 25
        assert authz.authorize_source.await_count == 1
