"""Adapter tests for Gitea webhook — GiteaAdapter: parse, fetch, handle_webhook.

Mock only httpx (out-of-process dependency), per Principle VIII.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock

import pytest

from treeloom.adapters.webhook.gitea import GiteaAdapter
from treeloom.domain.webhook import Provider, WebhookPayload


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_merge_payload(
    *,
    repo_url: str = "https://gitea.example.com/owner/repo",
    branch: str = "main",
    pr_number: int = 42,
) -> dict:
    """Build a minimal Gitea pull_request.merged webhook body."""
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


def _make_gitea_files_response(files: list[dict]) -> list[dict]:
    """Return a Gitea /pulls/:n/files response body.

    Gitea returns ``Filename`` and ``Status`` (capitalized) — same shape
    the domain normalizer expects for Provider.GITEA.
    """
    return [
        {"Filename": f["path"], "Status": f["status"]}
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
# GiteaAdapter.__init__
# ---------------------------------------------------------------------------

def test_gitea_adapter_init_with_token():
    """GiteaAdapter accepts secret, base_url, and optional token."""
    adapter = GiteaAdapter(
        secret="my-secret",
        base_url="https://gitea.example.com",
        token="gitea-token-123",
    )
    assert adapter._secret == "my-secret"
    assert adapter._base_url == "https://gitea.example.com"
    assert adapter._token == "gitea-token-123"


def test_gitea_adapter_init_without_token():
    """GiteaAdapter works without a token."""
    adapter = GiteaAdapter(secret="my-secret", base_url="https://gitea.example.com")
    assert adapter._token is None


def test_gitea_adapter_init_trailing_slash_cleaned():
    """GiteaAdapter strips trailing slash from base_url."""
    adapter = GiteaAdapter(
        secret="s", base_url="https://gitea.example.com/", token="t"
    )
    assert adapter._base_url == "https://gitea.example.com"


# ---------------------------------------------------------------------------
# parse_merge_event
# ---------------------------------------------------------------------------

def test_parse_merge_event_extracts_repo_url_branch_pr():
    """parse_merge_event extracts repo_url, branch, pr_number from a merge payload."""
    adapter = GiteaAdapter(secret="secret", base_url="https://gitea.example.com")
    body = _make_merge_payload(
        repo_url="https://gitea.example.com/acme/widget",
        branch="develop",
        pr_number=123,
    )
    repo_url, branch, pr_number = adapter.parse_merge_event(body)

    assert repo_url == "https://gitea.example.com/acme/widget.git"
    assert branch == "develop"
    assert pr_number == 123


def test_parse_merge_event_uses_clone_url():
    """parse_merge_event prefers clone_url from base.repo."""
    adapter = GiteaAdapter(secret="secret", base_url="https://gitea.example.com")
    body = _make_merge_payload(
        repo_url="https://gitea.example.com/myorg/myrepo",
    )
    repo_url, branch, pr_number = adapter.parse_merge_event(body)

    assert repo_url == "https://gitea.example.com/myorg/myrepo.git"
    assert branch == "main"
    assert pr_number == 42


def test_parse_merge_event_non_merged_returns_none():
    """parse_merge_event returns None for non-merged PR events."""
    adapter = GiteaAdapter(secret="secret", base_url="https://gitea.example.com")
    body = {
        "action": "opened",
        "pull_request": {
            "number": 1,
            "merged": False,
            "base": {
                "ref": "main",
                "repo": {"clone_url": "https://gitea.example.com/a/b.git"},
            },
        },
    }
    result = adapter.parse_merge_event(body)
    assert result is None


def test_parse_merge_event_missing_fields():
    """parse_merge_event returns None when required fields are missing."""
    adapter = GiteaAdapter(secret="secret", base_url="https://gitea.example.com")

    assert adapter.parse_merge_event({}) is None
    assert adapter.parse_merge_event({"action": "closed"}) is None
    assert adapter.parse_merge_event(
        {"action": "closed", "pull_request": {}}
    ) is None


# ---------------------------------------------------------------------------
# fetch_changed_files
# ---------------------------------------------------------------------------

def test_fetch_changed_files_calls_gitea_api(mocker):
    """fetch_changed_files calls GET /api/v1/repos/{owner}/{repo}/pulls/{n}/files via httpx."""
    adapter = GiteaAdapter(
        secret="secret", base_url="https://gitea.example.com", token="gitea-token"
    )

    mock_files = _make_gitea_files_response([
        {"path": "src/main.py", "status": "changed"},
        {"path": "tests/test_main.py", "status": "added"},
    ])

    mock_client = _make_httpx_client_mock(
        mocker, get_return=_mock_httpx_response(mock_files)
    )

    result = adapter.fetch_changed_files(
        repo_url="https://gitea.example.com/owner/repo.git",
        pr_number=42,
    )

    # Verify the API call
    mock_client.get.assert_called_once()
    call_args, call_kwargs = mock_client.get.call_args
    assert "owner/repo/pulls/42/files" in call_args[0]
    assert "/api/v1/repos/" in call_args[0]
    assert call_kwargs.get("headers", {}).get("Authorization") == "token gitea-token"

    # Verify raw result (Gitea API returns Filename/Status capitalized)
    assert len(result) == 2
    assert result[0]["Filename"] == "src/main.py"
    assert result[0]["Status"] == "changed"
    assert result[1]["Filename"] == "tests/test_main.py"
    assert result[1]["Status"] == "added"


def test_fetch_changed_files_no_token(mocker):
    """fetch_changed_files works without an API token (unauthenticated)."""
    adapter = GiteaAdapter(secret="secret", base_url="https://gitea.example.com")

    mock_client = _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    result = adapter.fetch_changed_files(
        repo_url="https://gitea.example.com/owner/repo.git", pr_number=1
    )

    call_args, call_kwargs = mock_client.get.call_args
    # No Authorization header when token is None
    headers = call_kwargs.get("headers", {})
    assert "Authorization" not in headers

    assert result == []


def test_fetch_changed_files_strips_git_suffix(mocker):
    """fetch_changed_files strips .git from repo_url to build owner/repo."""
    adapter = GiteaAdapter(
        secret="secret", base_url="https://gitea.example.com", token="t"
    )
    mock_client = _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    adapter.fetch_changed_files(
        repo_url="https://gitea.example.com/myorg/myrepo.git", pr_number=7
    )

    call_args, _ = mock_client.get.call_args
    url = call_args[0]
    assert "/repos/myorg/myrepo/pulls/7/files" in url
    assert "myrepo.git" not in url


def test_fetch_changed_files_http_error(mocker):
    """fetch_changed_files raises on HTTP error from Gitea API."""
    adapter = GiteaAdapter(
        secret="secret", base_url="https://gitea.example.com", token="t"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = Exception("404 Not Found")

    _make_httpx_client_mock(mocker, get_return=mock_resp)

    with pytest.raises(Exception, match="404"):
        adapter.fetch_changed_files(
            repo_url="https://gitea.example.com/owner/repo.git", pr_number=1
        )


# ---------------------------------------------------------------------------
# handle_webhook — orchestration
# ---------------------------------------------------------------------------

def test_handle_webhook_full_flow(mocker):
    """handle_webhook orchestrates: validate HMAC → parse → fetch → normalize."""
    secret = "my-secret"
    adapter = GiteaAdapter(
        secret=secret, base_url="https://gitea.example.com", token="gitea-token"
    )

    body = _make_merge_payload(
        repo_url="https://gitea.example.com/owner/repo",
        branch="main",
        pr_number=42,
    )

    # Compute valid HMAC signature (Gitea uses raw hex, no prefix)
    payload_bytes = json.dumps(body).encode()
    signature = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()

    headers = {
        "X-Gitea-Event": "pull_request",
        "X-Gitea-Signature": signature,
    }

    mock_files = _make_gitea_files_response([
        {"path": "src/app.py", "status": "changed"},
        {"path": "src/new.py", "status": "added"},
    ])

    # Mock httpx.Client for API call
    mock_client = _make_httpx_client_mock(
        mocker, get_return=_mock_httpx_response(mock_files)
    )

    result = adapter.handle_webhook(headers, body)

    assert result is not None
    assert isinstance(result, WebhookPayload)
    assert result.provider == Provider.GITEA
    assert result.repo_url == "https://gitea.example.com/owner/repo.git"
    assert result.branch == "main"

    # Check normalized files
    assert len(result.changed_files) == 2
    assert {"path": "src/app.py", "action": "modified"} in result.changed_files
    assert {"path": "src/new.py", "action": "added"} in result.changed_files

    # Verify API was called
    mock_client.get.assert_called_once()


def test_handle_webhook_invalid_hmac(mocker):
    """handle_webhook returns None when HMAC validation fails."""
    adapter = GiteaAdapter(secret="correct-secret", base_url="https://gitea.example.com")

    body = _make_merge_payload()
    headers = {
        "X-Gitea-Event": "pull_request",
        "X-Gitea-Signature": "wrongsignature",
    }

    # httpx.Client should NOT be called
    mock_client = _make_httpx_client_mock(mocker)

    result = adapter.handle_webhook(headers, body)

    assert result is None
    mock_client.get.assert_not_called()


def test_handle_webhook_non_merge_event(mocker):
    """handle_webhook returns None for non-merge events (parse returns None)."""
    secret = "secret"
    adapter = GiteaAdapter(secret=secret, base_url="https://gitea.example.com")

    body = {
        "action": "opened",
        "pull_request": {
            "number": 1,
            "merged": False,
            "base": {
                "ref": "main",
                "repo": {"clone_url": "https://gitea.example.com/a/b.git"},
            },
        },
    }
    payload_bytes = json.dumps(body).encode()
    signature = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-Gitea-Event": "pull_request",
        "X-Gitea-Signature": signature,
    }

    mock_client = _make_httpx_client_mock(mocker)

    result = adapter.handle_webhook(headers, body)

    assert result is None
    mock_client.get.assert_not_called()


def test_handle_webhook_fetch_failure_returns_none(mocker):
    """handle_webhook returns None gracefully when fetch_changed_files fails."""
    secret = "secret"
    adapter = GiteaAdapter(
        secret=secret, base_url="https://gitea.example.com", token="t"
    )

    body = _make_merge_payload()
    payload_bytes = json.dumps(body).encode()
    signature = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-Gitea-Event": "pull_request",
        "X-Gitea-Signature": signature,
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("500 Server Error")

    _make_httpx_client_mock(mocker, get_return=mock_resp)

    result = adapter.handle_webhook(headers, body)

    assert result is None


def test_handle_webhook_empty_files(mocker):
    """handle_webhook returns payload with empty changed_files when no files changed."""
    secret = "secret"
    adapter = GiteaAdapter(
        secret=secret, base_url="https://gitea.example.com", token="t"
    )

    body = _make_merge_payload()
    payload_bytes = json.dumps(body).encode()
    signature = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-Gitea-Event": "pull_request",
        "X-Gitea-Signature": signature,
    }

    _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    result = adapter.handle_webhook(headers, body)

    assert result is not None
    assert result.provider == Provider.GITEA
    assert result.changed_files == []


def test_handle_webhook_no_secret(mocker):
    """handle_webhook with empty secret skips HMAC validation (passes)."""
    adapter = GiteaAdapter(secret="", base_url="https://gitea.example.com", token="t")

    body = _make_merge_payload()
    headers = {}  # No signature header needed

    _make_httpx_client_mock(mocker, get_return=_mock_httpx_response([]))

    result = adapter.handle_webhook(headers, body)

    assert result is not None
    assert result.provider == Provider.GITEA
