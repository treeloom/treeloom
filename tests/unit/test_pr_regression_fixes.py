"""Defects the re-scan found in THIS PR's own fixes.

Each passed the suite as written, which is the point: the tests added
alongside each original fix asserted the thing that was fixed and nothing
about the paths next door.

* /jobs/{id}/errors and /job-groups/{id} were never scoped, while list_jobs
  and get_job were (CWE-639). The errors endpoint is the one under /jobs
  composed purely of absolute paths and exception text, and get_job_group
  returned RAW task rows with no redaction at all.
* The login timing fix moved the DECOY bcrypt call off-thread and left the
  real one blocking the event loop (CWE-400).
"""

import asyncio

import pytest
from treeloom.application import routes_auth as _rauth
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


@pytest.fixture
def scoped(mocker, monkeypatch):
    """Authenticated non-admin authorized for 'mine' but not 'theirs'."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("TREELOOM_SEARCH_OPEN", raising=False)
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


class TestJobErrorsIsScoped:
    @pytest.fixture(autouse=True)
    def store(self, mocker):
        mocker.patch.object(
            _state._job_file_error_store, "list_for_job",
            new=mocker.AsyncMock(return_value=[]),
        )

    @pytest.mark.asyncio
    async def test_another_principals_job_errors_are_404(self, mocker, scoped):
        mocker.patch.dict(_state._jobs, {
            "j-theirs": {"id": "j-theirs", "source_id": "theirs"},
        }, clear=True)
        with pytest.raises(HTTPException) as exc:
            await svc.get_job_errors("j-theirs", mocker.Mock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_own_job_errors_are_returned(self, mocker, scoped):
        mocker.patch.dict(_state._jobs, {
            "j-mine": {"id": "j-mine", "source_id": "mine"},
        }, clear=True)
        out = await svc.get_job_errors("j-mine", mocker.Mock())
        assert out["job_id"] == "j-mine"

    @pytest.mark.asyncio
    async def test_a_finished_job_resolves_through_the_store(self, mocker, scoped):
        """Errors outlive the in-memory _jobs entry, so the lookup has to fall
        through to the durable store or every finished job 404s."""
        mocker.patch.dict(_state._jobs, {}, clear=True)
        mocker.patch.object(
            _state, "_job_store",
            mocker.Mock(get=mocker.AsyncMock(return_value=mocker.Mock(source_id="mine"))),
        )
        out = await svc.get_job_errors("j-old", mocker.Mock())
        assert out["job_id"] == "j-old"

    @pytest.mark.asyncio
    async def test_unknown_job_is_not_visible(self, mocker, scoped):
        mocker.patch.dict(_state._jobs, {}, clear=True)
        mocker.patch.object(
            _state, "_job_store", mocker.Mock(get=mocker.AsyncMock(return_value=None))
        )
        with pytest.raises(HTTPException) as exc:
            await svc.get_job_errors("nope", mocker.Mock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_open_mode_is_unchanged(self, mocker, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        mocker.patch.dict(_state._jobs, {"j": {"id": "j", "source_id": "any"}}, clear=True)
        out = await svc.get_job_errors("j", mocker.Mock())
        assert out["job_id"] == "j"


class TestJobGroupIsScopedAndRedacted:
    def _group(self, mocker, tasks):
        mocker.patch(
            "treeloom.application.job_groups.summarize_group",
            return_value={"id": "g1", "status": "done"},
        )
        mocker.patch.object(
            _state, "_job_group_store",
            mocker.Mock(get=mocker.AsyncMock(return_value=mocker.Mock(id="g1"))),
        )
        mocker.patch.object(
            _state, "_job_store",
            mocker.Mock(list_by_group=mocker.AsyncMock(return_value=tasks)),
        )

    def _task(self, mocker, sid):
        t = mocker.Mock(source_id=sid)
        t.to_dict.return_value = {
            "job_id": f"j-{sid}", "status": "done", "source_id": sid,
            "source": f"/home/{sid}/repo", "error": "Traceback: boom",
        }
        return t

    @pytest.mark.asyncio
    async def test_group_of_another_principals_tasks_is_404(self, mocker, scoped):
        self._group(mocker, [self._task(mocker, "theirs")])
        with pytest.raises(HTTPException) as exc:
            await svc.get_job_group("g1", mocker.Mock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_foreign_tasks_are_filtered_from_a_mixed_group(self, mocker, scoped):
        self._group(mocker, [self._task(mocker, "mine"), self._task(mocker, "theirs")])
        out = await svc.get_job_group("g1", mocker.Mock())
        assert [t["source_id"] for t in out["tasks"]] == ["mine"]

    @pytest.mark.asyncio
    async def test_tasks_go_through_redaction_not_raw_to_dict(self, mocker, monkeypatch):
        """It returned t.to_dict() directly — the raw row, bypassing the
        redaction every other job endpoint applies."""
        monkeypatch.setenv("AUTH_ENABLED", "true")
        mocker.patch.object(
            _authz, "_require_authenticated_user",
            new=mocker.AsyncMock(side_effect=HTTPException(401, "anon")),
        )
        self._group(mocker, [self._task(mocker, "mine")])
        mocker.patch.object(_authz, "_visible_job_filter", new=mocker.AsyncMock(return_value=None))

        out = await svc.get_job_group("g1", mocker.Mock())

        blob = repr(out["tasks"])
        assert "/home/mine/repo" not in blob
        assert "Traceback" not in blob
        assert out["tasks"][0]["status"] == "done", "progress still visible"


class TestLoginVerifyIsOffThread:
    """Only the DECOY was moved off-thread by the timing fix; the real verify
    kept blocking the event loop for a full bcrypt round (~168ms) on every
    login with a known username — the branch an attacker drives hardest."""

    def test_no_verify_password_is_called_inline(self):
        """Both calls now pass verify_password BY REFERENCE to
        asyncio.to_thread, so a direct `verify_password(` call anywhere in
        this handler means one is back on the event loop.

        Checked this way rather than by scanning a window around the call:
        `user.password_hash` also appears in the guard clause above, so an
        index-based window silently inspects the wrong site and passes.
        """
        import inspect
        import re

        src = inspect.getsource(_rauth.auth_login)
        inline = re.findall(r"(?<!,\s)\bverify_password\(", src)
        assert inline == [], f"inline verify_password call(s): {inline}"

    def test_both_branches_hand_it_to_to_thread(self):
        """Symmetry matters: if one branch regresses to inline, the timing
        oracle comes back with it."""
        import inspect
        import re

        src = inspect.getsource(_rauth.auth_login)
        handed_off = re.findall(r"to_thread\(\s*\n?\s*verify_password", src)
        assert len(handed_off) == 2, (
            f"expected both branches off-thread, found {len(handed_off)}"
        )
