"""Adapter tests for webhook endpoint — FastAPI POST /webhook integration.

Tests cover:
- Valid GitHub payload → 202 with job_id
- Invalid HMAC → 401
- Unknown provider → 400
- Unindexed repo → 202 with "full index" message
- Missing secret env var → 503
- Already-indexed repo creates incremental job
- Non-merge event returns 202 but no action
"""
import json
import hmac
import hashlib
import pytest
from treeloom.application import indexer_runners as _runners
from treeloom.application import indexer_state
from treeloom.application import indexer_state as _state
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from treeloom.domain.webhook import Provider, WebhookPayload


@pytest.fixture
def test_client(mock_milvus_collection, mock_neo4j_driver):
    """Create a FastAPI TestClient for the indexer service.

    Requires mock_milvus_collection and mock_neo4j_driver from conftest
    so that the indexer_service module can import cleanly.
    """
    from treeloom.indexer_service import app
    return TestClient(app)


@pytest.fixture
def github_webhook_body():
    """Minimal valid GitHub pull_request.closed (merged) webhook body."""
    return {
        "action": "closed",
        "pull_request": {
            "merged": True,
            "number": 42,
            "base": {
                "ref": "main",
                "repo": {
                    "clone_url": "https://github.com/testowner/testrepo.git",
                },
            },
        },
    }


def _make_github_signature(secret: str, body: dict) -> str:
    """Create a valid X-Hub-Signature-256 value for the given secret + body."""
    payload = json.dumps(body)
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), payload.encode(), hashlib.sha256
    ).hexdigest()


# ── Test: valid GitHub payload → 202 ────────────────────────────────────

def test_webhook_valid_github_returns_202(
    test_client, monkeypatch, github_webhook_body,
):
    """POST /webhook with a valid GitHub payload returns 202 with job_id."""
    monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", "test-secret")

    mock_payload = WebhookPayload(
        provider=Provider.GITHUB,
        repo_url="https://github.com/testowner/testrepo.git",
        branch="main",
        changed_files=[
            {"path": "src/main.py", "action": "modified"},
            {"path": "src/utils.py", "action": "added"},
        ],
    )

    with patch(
        "treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
        return_value=mock_payload,
    ):
        # Mock list_sources in the actual application module
        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch(
                "treeloom.application.indexer_runners._run_index_repo_job",
                new_callable=AsyncMock,
            ), patch(
                "treeloom.application.routes_webhook._find_active_job_for_source",
                new_callable=AsyncMock, return_value=None,
            ), patch(
                "treeloom.application.indexer_runners._persist_job",
                new_callable=AsyncMock,
            ), patch(
                "treeloom.application.indexer_state._job_queue",
                new=MagicMock(enqueue=AsyncMock()),
            ):
                signature = _make_github_signature("test-secret", github_webhook_body)
                headers = {
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": signature,
                    "Content-Type": "application/json",
                }
                response = test_client.post(
                    "/webhook",
                    json=github_webhook_body,
                    headers=headers,
                )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


# ── Test: invalid HMAC → 401 ────────────────────────────────────────────

def test_webhook_invalid_hmac_returns_401(
    test_client, monkeypatch, github_webhook_body,
):
    """POST /webhook with a bad HMAC signature returns 401."""
    monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", "test-secret")

    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": "sha256=deadbeef",
        "Content-Type": "application/json",
    }
    response = test_client.post(
        "/webhook",
        json=github_webhook_body,
        headers=headers,
    )

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data


# ── Test: unknown provider → 400 ─────────────────────────────────────────

def test_webhook_unknown_provider_returns_400(test_client):
    """POST /webhook with no recognized provider header returns 400."""
    response = test_client.post(
        "/webhook",
        json={"some": "data"},
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "provider" in data["detail"].lower()


# ── Test: unindexed repo → 202 with full index message ──────────────────

def test_webhook_unindexed_repo_returns_202_full_index(
    test_client, monkeypatch, github_webhook_body,
):
    """POST /webhook for a repo not in sources → returns 202 with 'full index' message."""
    monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", "test-secret")

    mock_payload = WebhookPayload(
        provider=Provider.GITHUB,
        repo_url="https://github.com/newowner/newrepo.git",
        branch="main",
        changed_files=[
            {"path": "README.md", "action": "modified"},
        ],
    )

    with patch(
        "treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
        return_value=mock_payload,
    ):
        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch(
                "treeloom.application.indexer_runners._run_index_repo_job",
                new_callable=AsyncMock,
            ), patch(
                "treeloom.application.routes_webhook._find_active_job_for_source",
                new_callable=AsyncMock, return_value=None,
            ), patch(
                "treeloom.application.indexer_runners._persist_job",
                new_callable=AsyncMock,
            ), patch(
                "treeloom.application.indexer_state._job_queue",
                new=MagicMock(enqueue=AsyncMock()),
            ):
                signature = _make_github_signature("test-secret", github_webhook_body)
                headers = {
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": signature,
                    "Content-Type": "application/json",
                }
                response = test_client.post(
                    "/webhook",
                    json=github_webhook_body,
                    headers=headers,
                )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["source"] != ""


# ── Test: missing secret env var → 503 ──────────────────────────────────

def test_webhook_missing_secret_env_var_returns_503(
    test_client, monkeypatch, github_webhook_body,
):
    """POST /webhook when WEBHOOK_GITHUB_SECRET is not set returns 503."""
    monkeypatch.delenv("WEBHOOK_GITHUB_SECRET", raising=False)

    signature = _make_github_signature("test-secret", github_webhook_body)
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature,
        "Content-Type": "application/json",
    }
    response = test_client.post(
        "/webhook",
        json=github_webhook_body,
        headers=headers,
    )

    assert response.status_code == 503
    data = response.json()
    assert "detail" in data


# ── Test: incremental job creation for already-indexed repo ──────────────

def test_webhook_indexed_repo_creates_incremental_job(
    test_client, monkeypatch, github_webhook_body,
):
    """POST /webhook for an already-indexed repo creates an incremental job."""
    monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", "test-secret")
    monkeypatch.setenv("WEBHOOK_GITHUB_TOKEN", "")

    mock_payload = WebhookPayload(
        provider=Provider.GITHUB,
        repo_url="https://github.com/testowner/testrepo.git",
        branch="main",
        changed_files=[
            {"path": "src/main.py", "action": "modified"},
            {"path": "docs/readme.md", "action": "added"},
            {"path": "old/deprecated.py", "action": "removed"},
        ],
    )

    with patch(
        "treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
        return_value=mock_payload,
    ):
        existing_source = {
            "id": "existing_source_id",
            "url": "https://github.com/testowner/testrepo.git",
            "branch": "main",
        }
        with patch(
            "treeloom.application.indexer_service.graph_store.list_sources",
            new_callable=AsyncMock,
            return_value=[existing_source],
        ):
            with patch(
                "treeloom.application.indexer_runners._persist_job",
                new_callable=AsyncMock,
            ), patch(
                "treeloom.application.indexer_state._job_queue",
                new=MagicMock(enqueue=AsyncMock(), depth=AsyncMock(return_value=0)),
            ):
                signature = _make_github_signature("test-secret", github_webhook_body)
                headers = {
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": signature,
                    "Content-Type": "application/json",
                }
                response = test_client.post(
                    "/webhook",
                    json=github_webhook_body,
                    headers=headers,
                )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["files_to_process"] == 3


# ── Test: webhook for non-merge event returns 202 but no action ──────────

def test_webhook_non_merge_event_skips(
    test_client, monkeypatch,
):
    """POST /webhook for a non-merge PR event returns 202 without triggering indexing."""
    monkeypatch.setenv("WEBHOOK_GITHUB_SECRET", "test-secret")

    body = {
        "action": "opened",
        "pull_request": {
            "merged": False,
            "number": 10,
            "base": {
                "ref": "main",
                "repo": {
                    "clone_url": "https://github.com/testowner/testrepo.git",
                },
            },
        },
    }
    # No signature header — the mock returns None, simulating a non-merge
    # event where HMAC passed but the PR wasn't merged.
    headers = {
        "X-GitHub-Event": "pull_request",
        "Content-Type": "application/json",
    }

    with patch(
        "treeloom.adapters.webhook.github.GitHubAdapter.handle_webhook",
        return_value=None,
    ):
        response = test_client.post(
            "/webhook",
            json=body,
            headers=headers,
        )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "done"
    assert data["message"] != ""
