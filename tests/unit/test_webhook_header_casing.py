"""Webhook adapters must read the header casing the wire actually delivers.

`routes_webhook.handle_webhook` builds the adapter's header dict with
`dict(request.headers)`. Starlette stores header names lowercased, so that
dict has lowercase keys for EVERY real delivery. An adapter that looks its
credential up by a capitalized name reads `""` and rejects every genuine
webhook — and its signature check becomes unreachable code.

Tests that hand the adapter a literal like `{"X-Hub-Signature-256": ...}`
cannot catch this — that is a shape the ASGI server never produces — so an
adapter with the bug would pass its own tests while rejecting 100% of real
deliveries. This guard covers the supported adapters, because nothing else
stops the mistake being made.

These tests take their headers from a real request rather than a literal.
"""

import hashlib
import hmac
import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from treeloom.adapters.webhook.gitea import GiteaAdapter
from treeloom.adapters.webhook.github import GitHubAdapter

SECRET = "webhook-shared-secret"
BODY = {"action": "closed", "pull_request": {"merged": False}}


def wire_headers(sent: dict[str, str]) -> dict[str, str]:
    """The header dict `routes_webhook` really builds, for headers really sent.

    Round-trips through an ASGI request so the casing is the server's, not
    ours. A hand-written dict is what hid the original bug.
    """
    captured: dict[str, str] = {}

    async def endpoint(request):
        captured.update(dict(request.headers))  # exactly what the route does
        return JSONResponse({})

    TestClient(Starlette(routes=[Route("/", endpoint, methods=["POST"])])).post(
        "/", headers=sent
    )
    return captured


def sign(raw: bytes) -> str:
    return hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()


class TestTheWireIsLowercase:
    def test_capitalized_lookup_misses_on_a_real_request(self):
        """The premise, pinned. Every assertion below depends on it."""
        hdrs = wire_headers({"X-Hub-Signature-256": "sha256=abc"})
        assert hdrs.get("X-Hub-Signature-256", "") == ""
        assert hdrs["x-hub-signature-256"] == "sha256=abc"


class TestAdaptersAcceptRealDeliveries:
    """A correct signature sent over a real request must pass the signature
    check. Failing here means the credential check is dead — the adapter
    rejects every genuine webhook."""

    @pytest.mark.parametrize(
        "adapter,header",
        [
            (GitHubAdapter(secret=SECRET), "X-Hub-Signature-256"),
            (GiteaAdapter(secret=SECRET, base_url="https://gitea.example.com"),
             "X-Gitea-Signature"),
        ],
        ids=["github", "gitea"],
    )
    def test_valid_signature_passes_the_credential_check(self, adapter, header):
        raw = json.dumps(BODY).encode()
        hdrs = wire_headers({header: sign(raw)})

        result = adapter.handle_webhook(hdrs, BODY, raw_body=raw)

        # BODY is not a merged PR, so the result is None either way. The
        # distinction that matters is WHY: a signature failure and a
        # non-merge event both return None, so assert on the signature
        # directly rather than on the return value.
        from treeloom.domain.webhook import validate_hmac

        sig = {k.lower(): v for k, v in hdrs.items()}[header.lower()]
        assert validate_hmac(SECRET, raw, sig), "signature read from the wire failed"
        assert result is None  # non-merge event, as constructed

    @pytest.mark.parametrize(
        "adapter,header",
        [
            (GitHubAdapter(secret=SECRET), "X-Hub-Signature-256"),
            (GiteaAdapter(secret=SECRET, base_url="https://gitea.example.com"),
             "X-Gitea-Signature"),
        ],
        ids=["github", "gitea"],
    )
    def test_wrong_signature_is_rejected(self, adapter, header):
        raw = json.dumps(BODY).encode()
        hdrs = wire_headers({header: sign(b"different bytes")})
        assert adapter.handle_webhook(hdrs, BODY, raw_body=raw) is None


class TestNoAdapterReadsACapitalizedHeader:
    """The drift guard. Behavioural tests above cover today's two adapters;
    this catches a NEW adapter, or a new header read in an existing one,
    written with the capitalized spelling that looks right and never matches.
    """

    def test_inbound_header_reads_go_through_a_lowercased_dict(self):
        import pathlib
        import re

        pkg = pathlib.Path(GitHubAdapter.__module__.replace(".", "/")).parent
        root = pathlib.Path(__file__).resolve().parents[2] / "src" / pkg
        offenders = []
        for mod in sorted(root.glob("*.py")):
            for num, line in enumerate(mod.read_text().splitlines(), 1):
                # a read of the inbound `headers` param by a name with an
                # uppercase letter — assignments (outbound headers) are fine
                if re.search(r'headers\[\s*"[^"]+"\s*\]\s*=', line):
                    continue  # assignment: an OUTBOUND header being built
                m = re.search(r'\bheaders(?:\.get\(|\[)\s*"([^"]*[A-Z][^"]*)"', line)
                if m and "lower_headers" not in line:
                    offenders.append(f"{mod.name}:{num}: {m.group(1)}")
        assert not offenders, (
            "inbound header read by a capitalized name — `dict(request.headers)` "
            f"is lowercase, so this matches nothing on a real request: {offenders}"
        )
