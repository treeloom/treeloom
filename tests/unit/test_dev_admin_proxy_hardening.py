"""The dev-mode synthetic admin must not be reachable through a proxy.

`_synthetic_dev_admin` grants ADMIN with scopes {search, index, admin} when
AUTH_ENABLED=false + TREELOOM_DEV_MODE=1 + the client is on loopback. With
auth off everything else is already open, so what this grant adds is the
admin-gated surface: /users, /groups, /embedding-backends, /audit/search.

The loopback test reads `request.client.host`. That is not always the TCP
peer: uvicorn runs ProxyHeadersMiddleware with proxy_headers=True by default
and rewrites `scope["client"]` from X-Forwarded-For whenever the immediate
peer is in forwarded_allow_ips (default 127.0.0.1) — which a reverse proxy on
the same host always is.

This is hardening, not a demonstrated bypass. uvicorn 0.34's
`get_trusted_client_host` walks the list right-to-left and returns the first
untrusted hop, so a proxy that appends (`$proxy_add_x_forwarded_for`) still
yields the real client IP and an attacker cannot force loopback. The case
that does grant admin to every remote caller is a proxy that emits
`X-Forwarded-For: 127.0.0.1` itself — all hops trusted, so uvicorn falls back
to the first entry. That is a misconfiguration rather than an attack, but the
grant should not depend on a header any intermediary can set.

So: a request carrying forwarded headers has provably traversed something,
and dev-mode admin is refused for it. The check can only ever narrow the
grant, never widen it.
"""

import pytest

from treeloom.application.auth_middleware import _is_loopback_client


class _Client:
    def __init__(self, host):
        self.host = host


class _Req:
    def __init__(self, host, headers=None):
        self.client = _Client(host) if host is not None else None
        self.headers = headers or {}


class TestDirectLoopbackStillWorks:
    """The dev-convenience case this exists for must keep working."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "::1", "localhost"])
    def test_loopback_without_proxy_headers_is_accepted(self, host):
        assert _is_loopback_client(_Req(host)) is True

    @pytest.mark.parametrize("host", ["192.0.2.10", "192.168.1.5", "8.8.8.8", ""])
    def test_routable_peer_is_refused(self, host):
        assert _is_loopback_client(_Req(host)) is False

    def test_missing_client_is_refused(self):
        assert _is_loopback_client(_Req(None)) is False


class TestProxyEvidenceRefusesTheGrant:
    @pytest.mark.parametrize(
        "header",
        ["x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "forwarded"],
    )
    def test_forwarded_header_refuses_even_when_host_looks_loopback(self, header):
        """The exact shape of the misconfiguration: uvicorn has rewritten
        client.host to 127.0.0.1 from a header, so the peer check passes on
        evidence supplied by an intermediary."""
        req = _Req("127.0.0.1", {header: "127.0.0.1"})
        assert _is_loopback_client(req) is False

    def test_header_casing_does_not_matter(self):
        req = _Req("127.0.0.1", {"X-Forwarded-For": "127.0.0.1"})
        assert _is_loopback_client(req) is False

    def test_unrelated_headers_do_not_refuse(self):
        """Only forwarding evidence counts — an ordinary local request carries
        plenty of other headers."""
        req = _Req("127.0.0.1", {"user-agent": "curl/8", "accept": "*/*"})
        assert _is_loopback_client(req) is True

    def test_request_without_headers_attribute_is_handled(self):
        """Never let the hardening itself throw on an unexpected request
        shape — that would turn a narrow check into a 500 on every request."""
        class Bare:
            client = _Client("127.0.0.1")

        assert _is_loopback_client(Bare()) is True
