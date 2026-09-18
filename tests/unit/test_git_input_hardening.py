"""Git URL destination policy and the hardened preflight git invocation.

Two findings, one theme: what reaches `git`.

`_is_safe_git_url` previously rejected only an empty string, a leading '-' and
'::'. GIT_ALLOW_PROTOCOL constrained the *scheme* but nothing constrained the
*host*, so `git://10.0.0.50/secrets` or `http://169.254.169.254/` was accepted,
cloned, embedded into Milvus/Neo4j and then readable via /search — SSRF with
durable exfiltration (CWE-918).

Separately, the preflight scanner runs `git ls-files` inside a caller-supplied
directory. git consults `core.fsmonitor` from that repo's config while
enumerating with `--others` and executes the program it names (CWE-78).

Design note these tests pin: RFC1918 is allowed by DEFAULT. Self-hosted git on
a private network is a first-class use case, so blocking it out of the box
would break the common deployment; it is opt-in via TREELOOM_GIT_BLOCK_PRIVATE.
Loopback and link-local are always refused because no legitimate git remote
lives there.

Third: the host policy above judges the URL the *caller submitted*, but git
connects to wherever that URL redirects, so a public-looking
`https://attacker.example/repo` answering `302 -> http://169.254.169.254/`
walked straight past it. Redirect-following is therefore off by default in
`_git_env()`, which every remote-talking git invocation now shares.
"""

import pytest
from treeloom.application import indexer_runners as _runners

from treeloom.application import indexer_runners as _indexer_service


def _svc(monkeypatch, *, allowed_hosts="", block_private="", allow_loopback=""):
    """Point the module's env-derived policy constants at a test value.

    These are resolved from the environment at import time, so setenv alone
    would not reach them. Patching the attributes is preferred over
    importlib.reload: reloading re-registers the FastAPI event handlers on
    every call, which floods the run with deprecation warnings and leaves
    other test modules holding a stale module object.
    """
    monkeypatch.setattr(
        _indexer_service,
        "GIT_ALLOWED_HOSTS",
        frozenset(h.strip().lower() for h in allowed_hosts.split(",") if h.strip()),
    )
    monkeypatch.setattr(
        _indexer_service, "GIT_BLOCK_PRIVATE", block_private.lower() == "true"
    )
    monkeypatch.setattr(
        _indexer_service, "GIT_ALLOW_LOOPBACK", allow_loopback.lower() == "true"
    )
    return _indexer_service


class TestPreExistingRejections:
    """The original three rules must survive the rewrite."""

    @pytest.mark.parametrize(
        "url", ["", "-upload-pack=evil", "ext::sh -c id", "file::/etc/passwd"]
    )
    def test_still_rejected(self, monkeypatch, url):
        assert _svc(monkeypatch)._is_safe_git_url(url) is False


class TestSchemeValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/acme/widget.git",
            "http://git.example.com/acme/widget.git",
            "ssh://git@github.com/acme/widget.git",
            "git://git.example.com/acme/widget.git",
            "git@github.com:acme/widget.git",  # scp-like, no scheme
        ],
    )
    def test_supported_schemes_accepted(self, monkeypatch, url):
        assert _svc(monkeypatch)._is_safe_git_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://git.example.com/repo.git",
            "data://x/y",
        ],
    )
    def test_unsupported_schemes_refused(self, monkeypatch, url):
        assert _svc(monkeypatch)._is_safe_git_url(url) is False

    def test_colon_without_slashes_parses_as_scp_like_just_as_git_does(
        self, monkeypatch
    ):
        """`javascript:alert(1)//x/y` has no `://`, so it is scp-like syntax
        with host `javascript` — which is exactly how git reads it. It is
        accepted here and then fails at git, because there is no such host.

        Pinned deliberately: this looks like a miss if you read it with
        browser eyes, and someone may be tempted to "fix" it by rejecting all
        colons — which would break the legitimate `git@host:path` form that
        the scp-like tests above depend on.
        """
        svc = _svc(monkeypatch)
        assert svc._git_url_host("javascript:alert(1)//x/y") == "javascript"
        assert svc._is_safe_git_url("javascript:alert(1)//x/y") is True


class TestAlwaysForbiddenDestinations:
    """No legitimate git remote lives at these, so they are refused
    unconditionally — not only under the opt-in private-range flag."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "git://127.0.0.1/repo.git",
            "https://localhost/repo.git",
            "ssh://git@[::1]/repo.git",
            "git://0.0.0.0/repo.git",
        ],
    )
    def test_refused_by_default(self, monkeypatch, url):
        assert _svc(monkeypatch)._is_safe_git_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "git://0.0.0.0/repo.git",
        ],
    )
    def test_loopback_opt_in_does_not_reach_them(self, monkeypatch, url):
        """TREELOOM_GIT_ALLOW_LOOPBACK opens loopback and nothing else. The
        metadata address in particular is the whole reason link-local is on
        this list, and 0.0.0.0 is not an address a remote can live at."""
        svc = _svc(monkeypatch, allow_loopback="true")
        assert svc._is_safe_git_url(url) is False


class TestLoopbackIsOptIn:
    """Refused by default — a request to clone 127.0.0.1 is the shape of an
    SSRF probe. The opt-in exists for an operator-started git daemon, which
    is how scripts/loadtest_merge_searchable.py serves its fixture repos.
    """

    LOCAL = [
        "git://localhost/repo0",
        "git://127.0.0.1/repo.git",
        "https://localhost.localdomain/repo.git",
    ]

    @pytest.mark.parametrize("url", LOCAL)
    def test_refused_without_the_flag(self, monkeypatch, url):
        assert _svc(monkeypatch)._is_safe_git_url(url) is False

    @pytest.mark.parametrize("url", LOCAL)
    def test_allowed_with_the_flag(self, monkeypatch, url):
        svc = _svc(monkeypatch, allow_loopback="true")
        assert svc._is_safe_git_url(url) is True

    def test_the_loadtest_harness_url_shape_works(self, monkeypatch):
        """The concrete URL the harness builds, so a future tightening that
        breaks it fails here rather than in a load run nobody reads."""
        svc = _svc(monkeypatch, allow_loopback="true")
        assert svc._is_safe_git_url("git://localhost/repo3") is True

    def test_the_flag_is_exact_match_not_truthy(self, monkeypatch):
        for value in ("1", "yes", "TRUE ", "on"):
            svc = _svc(monkeypatch, allow_loopback=value)
            assert svc._is_safe_git_url("git://localhost/r") is False

    def test_ipv6_literals_stay_refused_because_of_the_double_colon_rule(
        self, monkeypatch
    ):
        """`[::1]` never reaches the host check at all: `_is_safe_git_url`
        refuses any URL containing '::' as remote-helper smuggling
        (`ext::`, `file::`), and an IPv6 literal contains one.

        So IPv6 remotes are unsupported across the board, not just on
        loopback — a pre-existing limitation of that rule, pinned here
        because it makes the ::1 case in TestAlwaysForbiddenDestinations
        pass for a reason other than the one it names. Narrowing '::' to a
        real transport-prefix match is the change that would lift it, and
        that is a deliberate piece of work, not a tweak.
        """
        svc = _svc(monkeypatch, allow_loopback="true")
        assert svc._is_safe_git_url("ssh://git@[::1]/repo.git") is False
        assert svc._is_safe_git_url("https://[2606:4700::1111]/a.git") is False
        # The host check itself has no objection — the URL rule fires first.
        assert svc._host_is_forbidden("::1") is False


class TestPrivateRangesAreOptIn:
    PRIVATE = "git://10.0.0.5/internal.git"

    def test_allowed_by_default_so_self_hosted_git_keeps_working(self, monkeypatch):
        assert _svc(monkeypatch)._is_safe_git_url(self.PRIVATE) is True

    def test_refused_when_operator_opts_in(self, monkeypatch):
        svc = _svc(monkeypatch, block_private="true")
        assert svc._is_safe_git_url(self.PRIVATE) is False

    def test_public_host_still_allowed_when_opted_in(self, monkeypatch):
        svc = _svc(monkeypatch, block_private="true")
        assert svc._is_safe_git_url("https://github.com/acme/widget.git") is True


class TestHostAllowList:
    def test_unset_allows_any_permitted_host(self, monkeypatch):
        svc = _svc(monkeypatch)
        assert svc._is_safe_git_url("https://gitlab.com/a/b.git") is True

    def test_configured_list_excludes_everything_else(self, monkeypatch):
        svc = _svc(monkeypatch, allowed_hosts="github.com, git.acme.test")
        assert svc._is_safe_git_url("https://github.com/a/b.git") is True
        assert svc._is_safe_git_url("https://git.acme.test/a/b.git") is True
        assert svc._is_safe_git_url("https://gitlab.com/a/b.git") is False

    def test_match_is_case_insensitive(self, monkeypatch):
        svc = _svc(monkeypatch, allowed_hosts="github.com")
        assert svc._is_safe_git_url("https://GitHub.COM/a/b.git") is True

    def test_allow_list_does_not_override_always_forbidden(self, monkeypatch):
        """Explicitly listing a loopback host must not re-enable it — the
        allow-list narrows what is permitted, it never widens it."""
        svc = _svc(monkeypatch, allowed_hosts="localhost")
        assert svc._is_safe_git_url("https://localhost/repo.git") is False

    def test_allow_list_still_applies_under_the_loopback_opt_in(self, monkeypatch):
        """Opting into loopback re-enables the destination, but the host
        allow-list is a separate gate and still has to pass."""
        svc = _svc(monkeypatch, allowed_hosts="github.com", allow_loopback="true")
        assert svc._is_safe_git_url("git://localhost/repo") is False
        svc = _svc(monkeypatch, allowed_hosts="localhost", allow_loopback="true")
        assert svc._is_safe_git_url("git://localhost/repo") is True


class TestHostExtraction:
    def test_scp_like_form(self, monkeypatch):
        svc = _svc(monkeypatch)
        assert svc._git_url_host("git@github.com:acme/widget.git") == "github.com"

    def test_url_form_strips_credentials_and_port(self, monkeypatch):
        svc = _svc(monkeypatch)
        assert svc._git_url_host("https://user:pw@git.example.com:8443/a.git") == (
            "git.example.com"
        )

    def test_no_host_is_none(self, monkeypatch):
        svc = _svc(monkeypatch)
        assert svc._git_url_host("just-a-path/with/slashes") is None


class TestPreflightGitInvocationIsHardened:
    """The scanner must neutralise config-driven execution in a repo it did
    not create. `-c` on the command line overrides the target's .git/config."""

    def test_fsmonitor_and_hooks_are_disabled_and_system_config_ignored(self, mocker):
        from treeloom.preflight import scanner

        run = mocker.patch.object(scanner.subprocess, "run")
        run.return_value = mocker.Mock(stdout="")
        scanner._gitignored_files(scanner.Path("/tmp/some-repo"))

        argv = run.call_args[0][0]
        assert "-c" in argv
        assert "core.fsmonitor=" in argv, "fsmonitor must be blanked"
        assert "core.hooksPath=/dev/null" in argv
        # The overrides must precede -C so they apply to that repo.
        assert argv.index("core.fsmonitor=") < argv.index("-C")
        assert run.call_args[1]["env"]["GIT_CONFIG_NOSYSTEM"] == "1"

    def test_global_config_is_left_alone(self, mocker):
        """core.excludesFile still shapes --exclude-standard, so preflight
        counts keep matching what the operator actually ignores."""
        from treeloom.preflight import scanner

        run = mocker.patch.object(scanner.subprocess, "run")
        run.return_value = mocker.Mock(stdout="")
        scanner._gitignored_files(scanner.Path("/tmp/some-repo"))

        assert "GIT_CONFIG_GLOBAL" not in run.call_args[1]["env"]


class TestRemoteGitEnvironment:
    """`_git_env()` is the single environment every clone and ls-remote uses.

    Verified out-of-band against real git 2.43: with redirect-following at
    git's default, `git ls-remote` on a URL that 302s to a loopback target
    reaches that target; with this environment it does not.
    """

    def _env(self, monkeypatch, follow=False):
        monkeypatch.setattr(_indexer_service, "GIT_FOLLOW_REDIRECTS", follow)
        return _indexer_service._git_env()

    def test_transport_allow_list_is_still_applied(self, monkeypatch):
        env = self._env(monkeypatch)
        assert env["GIT_ALLOW_PROTOCOL"] == "http:https:ssh:git"

    def test_redirects_are_refused_by_default(self, monkeypatch):
        env = self._env(monkeypatch)
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "http.followRedirects"
        assert env["GIT_CONFIG_VALUE_0"] == "false"

    def test_config_is_passed_by_environment_not_by_dash_c(self, monkeypatch):
        """GitPython classes `-c`/`--config` as unsafe options, and unlocking
        them to pass one would also unlock `--upload-pack`. The GIT_CONFIG_*
        triple is how this setting gets in without that trade."""
        env = self._env(monkeypatch)
        assert not any(k.startswith("-") for k in env)

    def test_operator_can_opt_back_in(self, monkeypatch):
        """A moved upstream stops resolving under the default, so there has to
        be a way back."""
        env = self._env(monkeypatch, follow=True)
        assert "GIT_CONFIG_KEY_0" not in env
        assert env["GIT_ALLOW_PROTOCOL"] == "http:https:ssh:git"


class TestEveryRemoteCallUsesThatEnvironment:
    """A clone site that builds its own env dict is a site the redirect
    policy does not reach — which is how this gap existed in the first place.
    """

    def test_no_clone_site_builds_its_own_env_dict(self):
        """Each of the three Repo.clone_from sites used to carry its own
        `{"GIT_ALLOW_PROTOCOL": ...}` literal. Adding a fourth by copy-paste
        is how a clone path would silently opt out of the redirect policy.

        Scans BOTH modules: the runners moved to indexer_runners while the
        /preflight clone stayed on the HTTP side, so checking one module only
        would pass while missing sites in the other.
        """
        import inspect

        from treeloom.application import indexer_service, indexer_runners

        import re

        src = inspect.getsource(indexer_runners) + inspect.getsource(indexer_service)
        assert '"env": {' not in src, "a clone site is building its own env"
        # Two sites live in indexer_runners and call _git_env() directly; the
        # /preflight clone stayed in indexer_service and reaches it as
        # runners._git_env(). Match either spelling — the property under test
        # is that all three go through the shared builder, not how it is named.
        assert len(re.findall(r'"env": (?:\w+\.)?_git_env\(\)', src)) == 3

    def test_ls_remote_uses_it(self, mocker, monkeypatch):
        # _git_remote_sha stayed on the HTTP side (staleness resolution),
        # while _git_env moved with the runners.
        from treeloom.application import indexer_service as _svc

        monkeypatch.setattr(_indexer_service, "GIT_FOLLOW_REDIRECTS", False)
        fake_git = mocker.Mock()
        fake_git.ls_remote.return_value = "abc123\tHEAD"
        mocker.patch("git.cmd.Git", return_value=fake_git)

        sha = _runners._git_remote_sha("https://github.com/acme/w.git")

        assert sha == "abc123"
        env = fake_git.update_environment.call_args.kwargs
        assert env["GIT_CONFIG_VALUE_0"] == "false"
        assert env["GIT_ALLOW_PROTOCOL"] == "http:https:ssh:git"
