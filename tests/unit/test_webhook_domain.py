"""Domain tests for webhook — Provider enum, WebhookPayload, detection, HMAC, normalization."""
import hmac
import hashlib
import json
import pytest
from treeloom.domain.webhook import (
    Provider,
    WebhookPayload,
    detect_provider,
    validate_hmac,
    normalize_changed_files,
)


# ── Provider Enum ───────────────────────────────────────────────────────

def test_provider_enum_values():
    """The supported providers, and only those.

    The enum is the contract `routes_webhook` dispatches on, so a member with
    no adapter would route a delivery to nothing.
    """
    assert Provider.GITHUB.value == "github"
    assert Provider.GITEA.value == "gitea"
    assert {p.value for p in Provider} == {"github", "gitea"}


def test_provider_from_string():
    """Provider can be looked up from string value."""
    assert Provider("github") == Provider.GITHUB
    assert Provider("gitea") == Provider.GITEA


def test_provider_invalid():
    """Unknown provider string raises ValueError."""
    with pytest.raises(ValueError):
        Provider("bitbucket")


# ── WebhookPayload Dataclass ────────────────────────────────────────────

def test_webhook_payload_defaults():
    """WebhookPayload can be created with minimal fields."""
    payload = WebhookPayload(
        provider=Provider.GITHUB,
        repo_url="https://github.com/owner/repo.git",
        branch="main",
    )
    assert payload.provider == Provider.GITHUB
    assert payload.repo_url == "https://github.com/owner/repo.git"
    assert payload.branch == "main"
    assert payload.changed_files == []


def test_webhook_payload_with_files():
    """WebhookPayload with changed_files list."""
    files = [
        {"path": "src/main.py", "action": "modified"},
        {"path": "src/utils.py", "action": "added"},
    ]
    payload = WebhookPayload(
        provider=Provider.GITEA,
        repo_url="https://gitea.example.com/team/project.git",
        branch="develop",
        changed_files=files,
    )
    assert len(payload.changed_files) == 2
    assert payload.changed_files[0]["path"] == "src/main.py"
    assert payload.changed_files[0]["action"] == "modified"


def test_webhook_payload_equality():
    """Two payloads with same values are equal."""
    p1 = WebhookPayload(provider=Provider.GITEA, repo_url="r", branch="b")
    p2 = WebhookPayload(provider=Provider.GITEA, repo_url="r", branch="b")
    assert p1 == p2


# ── detect_provider ─────────────────────────────────────────────────────

def test_detect_github():
    """GitHub detected via X-GitHub-Event header."""
    headers = {"X-GitHub-Event": "pull_request", "Content-Type": "application/json"}
    body = {}
    assert detect_provider(headers, body) == Provider.GITHUB



def test_detect_gitea():
    """Gitea detected via X-Gitea-Event header."""
    headers = {"X-Gitea-Event": "pull_request"}
    body = {}
    assert detect_provider(headers, body) == Provider.GITEA



def test_detect_unknown():
    """Unknown provider raises ValueError with descriptive message."""
    headers = {"Content-Type": "application/json"}
    body = {"some": "data"}
    with pytest.raises(ValueError, match="Unknown webhook provider"):
        detect_provider(headers, body)


def test_detect_github_case_insensitive():
    """Header detection is case-insensitive."""
    headers = {"x-github-event": "pull_request"}
    assert detect_provider(headers, {}) == Provider.GITHUB


# ── validate_hmac ───────────────────────────────────────────────────────

def test_validate_hmac_github_style():
    """GitHub-style HMAC-SHA256: signature = sha256=hexdigest."""
    secret = "my-secret-token"
    payload = json.dumps({"action": "merged", "pull_request": {}}).encode()
    signature = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    assert validate_hmac(secret, payload, signature) is True


def test_validate_hmac_bare_hex():
    """A bare hex digest, with no `sha256=` prefix, is accepted."""
    secret = "webhook-token"
    payload = json.dumps({"action": "closed"}).encode()
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert validate_hmac(secret, payload, expected) is True


def test_validate_hmac_wrong_secret():
    """Wrong secret returns False."""
    payload = b'{"test": "data"}'
    signature = hmac.new(b"correct-secret", payload, hashlib.sha256).hexdigest()
    assert validate_hmac("wrong-secret", payload, signature) is False


def test_validate_hmac_wrong_signature():
    """Tampered signature returns False."""
    secret = "my-secret"
    payload = b'{"test": "data"}'
    assert validate_hmac(secret, payload, "bad-signature-value") is False


def test_validate_hmac_empty_secret():
    """Empty secret validates only empty signature."""
    assert validate_hmac("", b"{}", "") is True
    assert validate_hmac("", b"{}", "any-signature") is False


def test_validate_hmac_sha1():
    """Non-SHA256 signatures (e.g., sha1=...) are rejected."""
    secret = "secret123"
    payload = b"body"
    sig = "sha1=" + hmac.new(secret.encode(), payload, hashlib.sha1).hexdigest()
    assert validate_hmac(secret, payload, sig) is False


# ── normalize_changed_files ─────────────────────────────────────────────

def test_normalize_github_files():
    """GitHub PR files normalized to [{path, action}]."""
    raw_files = [
        {"filename": "src/main.py", "status": "modified"},
        {"filename": "tests/test_main.py", "status": "added"},
        {"filename": "old/legacy.py", "status": "removed"},
    ]
    result = normalize_changed_files(Provider.GITHUB, raw_files)
    assert len(result) == 3
    assert {"path": "src/main.py", "action": "modified"} in result
    assert {"path": "tests/test_main.py", "action": "added"} in result
    assert {"path": "old/legacy.py", "action": "removed"} in result



def test_normalize_gitea_files():
    """Gitea PR files normalized (same shape as GitHub)."""
    raw_files = [
        {"Filename": "pkg/handler.go", "Status": "changed"},
    ]
    result = normalize_changed_files(Provider.GITEA, raw_files)
    assert result == [{"path": "pkg/handler.go", "action": "modified"}]



def test_normalize_none_provider():
    """None provider returns empty list (graceful degradation)."""
    result = normalize_changed_files(None, [])
    assert result == []


def test_normalize_empty_files():
    """Empty file list returns empty list for all providers."""
    for provider in Provider:
        assert normalize_changed_files(provider, []) == []


@pytest.mark.parametrize(
    "headers,body,label",
    [
        ({"X-Webhook-Event": "merge_request"}, {}, "event header"),
        ({}, {"eventType": "pull_request.merged"}, "event body field"),
    ],
)
def test_unsupported_provider_events_are_not_detected(headers, body, label):
    """An event-shaped delivery from an unsupported provider gets a clean 400
    (ValueError here) rather than being mistaken for GitHub or Gitea."""
    with pytest.raises(ValueError, match="Unknown webhook provider"):
        detect_provider(headers, body)
