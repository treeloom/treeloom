"""finding 26 path traversal via webhook payload, finding 28 login timing oracle,
finding 49 bcrypt's silent 72-byte truncation.
"""

import os

import pytest
from treeloom.application import indexer_runners as _runners
from treeloom.application import routes_auth as _rauth
from treeloom.application import indexer_authz as _authz
from fastapi import HTTPException

from treeloom.application import indexer_service as svc
from treeloom.application import indexer_state as _state


class TestWebhookPathContainment:
    """finding 26 CWE-22. changed_files[].path comes from the webhook payload and was
    joined onto the clone directory unchecked. The resolved file was then
    chunked, embedded and made searchable — so arbitrary file read became
    durable exfiltration through /search, not a one-shot peek.
    """

    @pytest.fixture
    def clone(self, tmp_path):
        root = tmp_path / "treeloom_inc_abc"
        (root / "src").mkdir(parents=True)
        (root / "src" / "a.py").write_text("x = 1\n")
        return root

    def test_ordinary_relative_path_resolves(self, clone):
        got = _runners._resolve_inside_clone(str(clone), "src/a.py")
        assert got == os.path.realpath(str(clone / "src" / "a.py"))

    @pytest.mark.parametrize(
        "evil",
        [
            "../../etc/passwd",
            "../../../../../../etc/shadow",
            "src/../../../etc/passwd",
            "/etc/passwd",              # absolute: os.path.join discards the root
            "",
        ],
    )
    def test_escaping_paths_are_refused(self, clone, evil):
        assert _runners._resolve_inside_clone(str(clone), evil) is None

    def test_interior_dotdot_that_stays_inside_is_allowed(self, clone):
        """Refusing every '..' would be over-broad — real diffs contain them."""
        got = _runners._resolve_inside_clone(str(clone), "src/../src/a.py")
        assert got == os.path.realpath(str(clone / "src" / "a.py"))

    def test_sibling_sharing_a_prefix_is_refused(self, tmp_path):
        """/tmp/treeloom_inc_abc must not admit /tmp/treeloom_inc_abcDEF —
        this is what the trailing os.sep on the boundary test buys."""
        root = tmp_path / "treeloom_inc_abc"
        root.mkdir()
        (tmp_path / "treeloom_inc_abcDEF").mkdir()
        assert _runners._resolve_inside_clone(
            str(root), "../treeloom_inc_abcDEF/loot.py"
        ) is None

    def test_symlink_planted_in_the_clone_is_refused(self, clone, tmp_path):
        """A repo can ship a symlink. realpath() resolves it before the
        boundary test, so pointing one out of the tree does not smuggle the
        target in."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("SECRET\n")
        (clone / "link.py").symlink_to(outside / "secret.py")
        assert _runners._resolve_inside_clone(str(clone), "link.py") is None


class TestLoginTimingIsEqualised:
    """finding 28 CWE-208. The absent-user branch returned before bcrypt ran, so a
    non-existent username answered in microseconds while a real one paid the
    full KDF cost — two-plus orders of magnitude, measurable on one request.
    The 401 text was already identical; the clock was the tell.
    """

    def test_absent_user_branch_still_runs_a_verify(self):
        import inspect

        src = inspect.getsource(_rauth.auth_login)
        idx = src.index("if user is None or not user.password_hash:")
        branch = src[idx: src.index("raise HTTPException", idx)]
        assert "verify_password" in branch

    def test_the_dummy_hash_is_a_real_bcrypt_hash(self):
        """A cheap stand-in (empty string, a constant) would not cost the same
        as the real path, which is the entire point."""
        from treeloom.adapters.authorization.user_store import _BCRYPT_HASH_RE

        assert _BCRYPT_HASH_RE.match(_authz._dummy_password_hash())

    def test_it_is_computed_once(self):
        assert _authz._dummy_password_hash() is _authz._dummy_password_hash()

    def test_no_password_can_match_it(self):
        from treeloom.adapters.authorization.user_store import verify_password

        for guess in ("", "password", "admin"):
            assert verify_password(guess, _authz._dummy_password_hash()) is False


class TestBcryptLengthLimit:
    """finding 49 CWE-916. bcrypt hashes at most 72 BYTES and discards the rest in
    silence, so two passphrases sharing a 72-byte prefix are one credential
    and nothing says so."""

    def test_over_limit_password_is_rejected_not_truncated(self):
        from treeloom.adapters.authorization.user_store import (
            PasswordTooLongError,
            hash_password,
        )

        with pytest.raises(PasswordTooLongError):
            hash_password("a" * 73)

    def test_the_limit_is_bytes_not_characters(self):
        """A non-ASCII passphrase hits the limit sooner than its character
        count suggests — 30 four-byte characters is already over."""
        from treeloom.adapters.authorization.user_store import (
            PasswordTooLongError,
            hash_password,
        )

        assert len("😀") == 1 and len("😀".encode()) == 4
        with pytest.raises(PasswordTooLongError):
            hash_password("😀" * 19)

    def test_at_the_limit_still_works(self):
        from treeloom.adapters.authorization.user_store import (
            hash_password,
            verify_password,
        )

        pw = "a" * 72
        assert verify_password(pw, hash_password(pw)) is True

    def test_verify_refuses_an_over_long_candidate(self):
        """The read side of the same hazard: an over-long password must not
        authenticate against a hash made from its first 72 bytes."""
        from treeloom.adapters.authorization.user_store import (
            hash_password,
            verify_password,
        )

        stored = hash_password("a" * 72)
        assert verify_password("a" * 72 + "anything-at-all", stored) is False

    @pytest.mark.asyncio
    async def test_endpoint_answers_400_not_500(self, mocker):
        """The caller can fix this, so it is a client error — and silently
        truncating, which is what it replaced, is not an option."""
        mocker.patch.object(_authz, "_acting_as_admin", return_value=True)
        mocker.patch.object(
            _state._user_store, "get_by_id",
            new=mocker.AsyncMock(return_value=mocker.Mock(id="u1", password_hash=None)),
        )
        update = mocker.patch.object(
            _state._user_store, "update_password", new=mocker.AsyncMock(return_value=True)
        )
        request = mocker.Mock()
        request.state.user = mocker.Mock(id="u1")

        with pytest.raises(HTTPException) as exc:
            await _rauth.set_password("u1", _rauth.SetPasswordRequest(password="a" * 200), request)

        assert exc.value.status_code == 400
        assert "72" in exc.value.detail
        update.assert_not_awaited(), "nothing should be stored"

    @pytest.mark.asyncio
    async def test_a_password_within_the_limit_is_still_stored(self, mocker):
        mocker.patch.object(_authz, "_acting_as_admin", return_value=True)
        mocker.patch.object(
            _state._user_store, "get_by_id",
            new=mocker.AsyncMock(return_value=mocker.Mock(id="u1", password_hash=None)),
        )
        update = mocker.patch.object(
            _state._user_store, "update_password", new=mocker.AsyncMock(return_value=True)
        )
        mocker.patch.object(_state, "_session_store", mocker.Mock(
            delete_for_user=mocker.AsyncMock()
        ))
        request = mocker.Mock()
        request.state.user = mocker.Mock(id="u1")

        await _rauth.set_password(
            "u1", _rauth.SetPasswordRequest(password="a-fine-password"), request
        )
        update.assert_awaited_once()
