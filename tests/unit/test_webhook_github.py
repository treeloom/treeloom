"""Adapter tests for GitHub webhook — GitHubAdapter: parse, fetch, handle_webhook.

Mock only httpx (out-of-process dependency), per Principle VIII.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock

import pytest

from treeloom.adapters.webhook.github import GitHubAdapter
from treeloom.domain.webhook import Provider, WebhookPayload


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_merge_payload(
    *,
    repo_url: str = "https://github.com/owner/repo",
    branch: str = "main",
    pr_number: int = 42,
) -> dict:
    """Build a minimal GitHub pull_request.merged webhook body."""
    return {
        "action": "closed",
        "pull_request": {
            "number": pr_number,
            "merged": True,
            "base": {
                "ref": branch,
                "repo": {
                    "clone_url": repo_url + ".git",
                    "html_url": repo_url,
                    "full_name": "owner/repo",
                },
            },
        },
    }


def _make_github_files_response(files: list[dict]) -> list[dict]:
    """Return a GitHub /pulls/:n/files response body."""
    return [
        {"filename": f["path"], "status": f["status"], "additions": 5, "deletions": 2}
        for f in files
    ]


def _mock_httpx_response(json_body, status_code=200):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


def _make_httpx_client_mock(mocker, get_return=None):
    """Create a mock httpx.Client that works as a context manager.

    ``httpx.Client()`` is used as ``with httpx.Client() as client:``.
    The mock must support the context-manager protocol so that ``client``
    inside the ``with`` block is the mocked object.
    """
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    if get_return is not None:
        mock_client.get.return_value = get_return
    mocker.patch("httpx.Client", return_value=mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# GitHubAdapter.__init__
# ---------------------------------------------------------------------------

def test_github_adapter_init_with_token():
    """GitHubAdapter accepts secret and optional token."""
    adapter = GitHubAdapter(secret="my-secret", token="ghp_test123")
    assert adapter._secret == "my-secret"
    assert adapter._token == "ghp_test123"


def test_github_adapter_init_without_token():
    """GitHubAdapter works without a token."""
    adapter = GitHubAdapter(secret="my-secret")
    assert adapter._token is None


# ---------------------------------------------------------------------------
# parse_merge_event
# ---------------------------------------------------------------------------

def test_parse_merge_event_extracts_repo_url_branch_pr():
    """parse_merge_event extracts repo_url, branch, pr_number from a merge payload."""
    adapter = GitHubAdapter(secret="secret")
    body = _make_merge_payload(
        repo_url="https://github.com/acme/widget",
        branch="develop",
        pr_number=123,
    )
    repo_url, branch, pr_number = adapter.parse_merge_event(body)

    assert repo_url == "https://github.com/acme/widget.git"
    assert branch == "develop"
    assert pr_number == 123


def test_parse_merge_event_uses_clone_url():
    """parse_merge_event prefers clone_url from base.repo."""
    adapter = GitHubAdapter(secret="secret")
    body = _make_merge_payload(
        repo_url="https://github.com/myorg/myrepo",
    )
    repo_url, branch, pr_number = adapter.parse_merge_event(body)

    assert repo_url == "https://github.com/myorg/myrepo.git"
    assert branch == "main"
    assert pr_number == 42


def test_parse_merge_event_non_merged_returns_none():
    """parse_merge_event returns None for non-merged PR events."""
    adapter = GitHubAdapter(secret="secret")
    body = {
        "action": "opened",
        "pull_request": {
            "number": 1,
            "merged": False,
            "base": {
                "ref": "main",
                "repo": {"clone_url": "https://github.com/a/b.git"},
            },
        },
    }
    result = adapter.parse_merge_event(body)
    assert result is None


def test_parse_merge_event_missing_fields():
    """parse_merge_event returns None when required fields are missing."""
    adapter = GitHubAdapter(secret="secret")

    assert adapter.parse_merge_event({}) is None
    assert adapter.parse_merge_event({"action": "closed"}) is None
    assert adapter.parse_merge_event(
        {"action": "closed", "pull_request": {}}
    ) is None


# ---------------------------------------------------------------------------
# fetch_changed_files
# ---------------------------------------------------------------------------

def test_fetch_changed_files_calls_github_api(mocker):
    """fetch_changed_files calls GET /repos/{owner}/{repo}/pulls/{n}/files via httpx."""
    adapter = GitHubAdapter(secret="secret", token="ghp_test")

    mock_files = _make_github_files_response([
        {"path": "src/main.py", "status": "modified"},
        {"path": "tests/test_main.py", "status": "added"},
    ])

    mock_client = _make_httpx_client_mock(
        mocker, get_return=_mock_httpx_response(mock_files)
    )

    result = adapter.fetch_changed_files(
        repo_url="https://github.com/owner/repo.git",
        pr_number=42,
    )

    # Verify the API call
    mock_client.get.assert_called_once()
    call_args, call_kwargs = mock_client.get.call_args
    assert "owner/repo/pulls/42/files" in call_args[0]
    assert call_kwargs.get("headers", {}).get("Authorization") == "Bearer ghp_test"

    # Verify result (GitHub API returns "filename", not "path")
    assert len(result) == 2
    assert result[0]["filename"] == "src/main.py"
    assert result[0]["status"] == "modified"
    assert result[1]["filename"] == "tests/test_main.py"
    assert result[1]["status"] == "added"


def test_fetch_changed_files_no_token(mocker):
    """fetch_changed_files works without an API token (unauthenticated)."""
    adapter = GitHubAdapter(secret="secret")

    mock_client = _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    result = adapter.fetch_changed_files(
        repo_url="https://github.com/owner/repo.git", pr_number=1
    )

    call_args, call_kwargs = mock_client.get.call_args
    # No Authorization header when token is None
    headers = call_kwargs.get("headers", {})
    assert "Authorization" not in headers

    assert result == []


def test_fetch_changed_files_strips_git_suffix(mocker):
    """fetch_changed_files strips .git from repo_url to build owner/repo."""
    adapter = GitHubAdapter(secret="secret", token="t")
    mock_client = _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    adapter.fetch_changed_files(
        repo_url="https://github.com/myorg/myrepo.git", pr_number=7
    )

    call_args, _ = mock_client.get.call_args
    # The URL should have the slug "myorg/myrepo" (no trailing .git)
    url = call_args[0]
    assert "/repos/myorg/myrepo/pulls/7/files" in url
    assert "myrepo.git" not in url


def test_fetch_changed_files_http_error(mocker):
    """fetch_changed_files raises on HTTP error from GitHub API."""
    adapter = GitHubAdapter(secret="secret", token="t")

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = Exception("404 Not Found")

    _make_httpx_client_mock(mocker, get_return=mock_resp)

    with pytest.raises(Exception, match="404"):
        adapter.fetch_changed_files(
            repo_url="https://github.com/owner/repo.git", pr_number=1
        )


# ---------------------------------------------------------------------------
# handle_webhook — orchestration
# ---------------------------------------------------------------------------

def test_handle_webhook_full_flow(mocker):
    """handle_webhook orchestrates: validate HMAC → parse → fetch → normalize."""
    secret = "my-secret"
    adapter = GitHubAdapter(secret=secret, token="ghp_token")

    body = _make_merge_payload(
        repo_url="https://github.com/owner/repo",
        branch="main",
        pr_number=42,
    )

    # Compute valid HMAC signature
    payload_bytes = json.dumps(body).encode()
    signature = "sha256=" + hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()

    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature,
    }

    mock_files = _make_github_files_response([
        {"path": "src/app.py", "status": "modified"},
        {"path": "src/new.py", "status": "added"},
    ])

    # Mock httpx.Client for API call
    mock_client = _make_httpx_client_mock(
        mocker, get_return=_mock_httpx_response(mock_files)
    )

    result = adapter.handle_webhook(headers, body)

    assert result is not None
    assert isinstance(result, WebhookPayload)
    assert result.provider == Provider.GITHUB
    assert result.repo_url == "https://github.com/owner/repo.git"
    assert result.branch == "main"

    # Check normalized files
    assert len(result.changed_files) == 2
    assert {"path": "src/app.py", "action": "modified"} in result.changed_files
    assert {"path": "src/new.py", "action": "added"} in result.changed_files

    # Verify API was called
    mock_client.get.assert_called_once()


def test_handle_webhook_invalid_hmac(mocker):
    """handle_webhook returns None when HMAC validation fails."""
    adapter = GitHubAdapter(secret="correct-secret")

    body = _make_merge_payload()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": "sha256=wrongsignature",
    }

    # httpx.Client should NOT be called
    mock_client = _make_httpx_client_mock(mocker)

    result = adapter.handle_webhook(headers, body)

    assert result is None
    mock_client.get.assert_not_called()


def test_handle_webhook_non_merge_event(mocker):
    """handle_webhook returns None for non-merge events (parse returns None)."""
    secret = "secret"
    adapter = GitHubAdapter(secret=secret)

    body = {
        "action": "opened",
        "pull_request": {
            "number": 1,
            "merged": False,
            "base": {
                "ref": "main",
                "repo": {"clone_url": "https://github.com/a/b.git"},
            },
        },
    }
    payload_bytes = json.dumps(body).encode()
    signature = "sha256=" + hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature,
    }

    mock_client = _make_httpx_client_mock(mocker)

    result = adapter.handle_webhook(headers, body)

    assert result is None
    mock_client.get.assert_not_called()


def test_handle_webhook_fetch_failure_returns_none(mocker):
    """handle_webhook returns None gracefully when fetch_changed_files fails."""
    secret = "secret"
    adapter = GitHubAdapter(secret=secret, token="t")

    body = _make_merge_payload()
    payload_bytes = json.dumps(body).encode()
    signature = "sha256=" + hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature,
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("500 Server Error")

    _make_httpx_client_mock(mocker, get_return=mock_resp)

    result = adapter.handle_webhook(headers, body)

    assert result is None


def test_handle_webhook_empty_files(mocker):
    """handle_webhook returns payload with empty changed_files when no files changed."""
    secret = "secret"
    adapter = GitHubAdapter(secret=secret, token="t")

    body = _make_merge_payload()
    payload_bytes = json.dumps(body).encode()
    signature = "sha256=" + hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature,
    }

    _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    result = adapter.handle_webhook(headers, body)

    assert result is not None
    assert result.provider == Provider.GITHUB
    assert result.changed_files == []


def test_handle_webhook_no_secret(mocker):
    """handle_webhook with empty secret skips HMAC validation (passes)."""
    adapter = GitHubAdapter(secret="", token="t")

    body = _make_merge_payload()
    headers = {}  # No signature header needed

    _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    result = adapter.handle_webhook(headers, body)

    assert result is not None
    assert result.provider == Provider.GITHUB


# ---------------------------------------------------------------------------
# Regression: HMAC must validate the RAW request body, not a re-serialized
# json.dumps(parsed) — real GitHub signs the raw (compact) bytes.
# ---------------------------------------------------------------------------

def test_handle_webhook_validates_raw_body_compact_json(mocker):
    """A signature over the RAW (compact, separators without spaces) body — as
    real GitHub sends — is accepted when raw_body is supplied. The old code
    HMAC'd json.dumps(parsed) (spaced) and would reject this."""
    secret = "my-secret"
    adapter = GitHubAdapter(secret=secret, token="ghp_token")

    body = _make_merge_payload(
        repo_url="https://github.com/owner/repo", branch="main", pr_number=42
    )
    # Compact wire bytes (what GitHub actually sends / signs) — differs from
    # json.dumps(body) which inserts ", " and ": " separators.
    raw = json.dumps(body, separators=(",", ":")).encode()
    assert raw != json.dumps(body).encode()  # ensure the distinction is real
    signature = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    headers = {"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": signature}

    _make_httpx_client_mock(
        mocker,
        get_return=_mock_httpx_response(
            _make_github_files_response([{"path": "a.py", "status": "modified"}])
        ),
    )

    result = adapter.handle_webhook(headers, body, raw_body=raw)
    assert result is not None
    assert isinstance(result, WebhookPayload)


def test_handle_webhook_rejects_signature_over_reserialized_when_raw_differs(mocker):
    """When raw_body is supplied, a signature computed over json.dumps(parsed)
    (the OLD scheme) is rejected if it differs from the raw bytes."""
    secret = "my-secret"
    adapter = GitHubAdapter(secret=secret, token="ghp_token")

    body = _make_merge_payload(
        repo_url="https://github.com/owner/repo", branch="main", pr_number=42
    )
    raw = json.dumps(body, separators=(",", ":")).encode()       # actual bytes
    reserialized = json.dumps(body).encode()                     # old (spaced)
    bad_sig = "sha256=" + hmac.new(
        secret.encode(), reserialized, hashlib.sha256
    ).hexdigest()
    headers = {"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": bad_sig}

    result = adapter.handle_webhook(headers, body, raw_body=raw)
    assert result is None  # signature didn't match the raw body → rejected


# ---------------------------------------------------------------------------
# Gated inline-files test mode: read changed files from the payload
# instead of the GitHub API (no real PR / no rate limit).
# ---------------------------------------------------------------------------

def test_handle_webhook_inline_files_skips_api(mocker):
    """With trust_inline_files=True, changed files come from pull_request.files
    and NO GitHub API call is made."""
    secret = "my-secret"
    adapter = GitHubAdapter(secret=secret, token=None, trust_inline_files=True)

    body = _make_merge_payload(
        repo_url="https://github.com/owner/repo", branch="main", pr_number=1
    )
    body["pull_request"]["files"] = [
        {"filename": "src/a.py", "status": "modified"},
        {"filename": "src/b.py", "status": "added"},
    ]
    raw = json.dumps(body).encode()
    sig = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    headers = {"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": sig}

    # If the adapter touched the API this would blow up.
    spy = mocker.patch.object(
        adapter, "fetch_changed_files", side_effect=AssertionError("API called")
    )

    result = adapter.handle_webhook(headers, body, raw_body=raw)
    assert result is not None
    spy.assert_not_called()
    paths = {f["path"] for f in result.changed_files}
    assert paths == {"src/a.py", "src/b.py"}


def test_handle_webhook_inline_files_default_off_uses_api(mocker):
    """Without the flag, the adapter still fetches via the API (default)."""
    secret = "my-secret"
    adapter = GitHubAdapter(secret=secret, token="ghp_t")  # trust_inline_files defaults False
    assert adapter._trust_inline_files is False
