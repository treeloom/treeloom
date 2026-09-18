"""Gitea webhook adapter.

Implements webhook handling for Gitea pull_request merge events.
Uses domain/webhook.py for HMAC validation and file normalization.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from treeloom.domain.webhook import (
    Provider,
    WebhookPayload,
    normalize_changed_files,
    validate_hmac,
)

if TYPE_CHECKING:
    from typing import Optional


class GiteaAdapter:
    """Adapts Gitea webhooks to a normalized :class:`WebhookPayload`.

    Handles:
    - HMAC validation (X-Gitea-Signature header)
    - Parse pull_request.merged events
    - Fetch changed files via Gitea REST API
    - Normalize to domain payload
    """

    def __init__(
        self,
        secret: str,
        base_url: str,
        token: Optional[str] = None,
    ) -> None:
        """Initialize the Gitea adapter.

        Args:
            secret: Shared HMAC secret for webhook signature validation.
            base_url: Base URL of the Gitea instance (e.g., ``https://gitea.example.com``).
            token: Optional Gitea API token for authenticated API requests.
        """
        self._secret = secret
        self._base_url = base_url.rstrip("/")
        self._token = token

    # ── Parse ─────────────────────────────────────────────────────────────

    def parse_merge_event(self, body: dict) -> tuple[str, str, int] | None:
        """Extract repo_url, branch, and PR number from a Gitea merge webhook body.

        Args:
            body: Parsed JSON body from the webhook request.

        Returns:
            ``(repo_url, branch, pr_number)`` if the event is a merged PR;
            ``None`` otherwise or if required fields are missing.
        """
        try:
            action = body.get("action")
            pull_request = body.get("pull_request", {})

            if action != "closed" or not pull_request.get("merged"):
                return None

            base = pull_request.get("base", {})
            repo = base.get("repo", {})

            repo_url = repo.get("clone_url", "")
            branch = base.get("ref", "")
            pr_number = pull_request.get("number")

            if not repo_url or not branch or pr_number is None:
                return None

            return (repo_url, branch, pr_number)
        except (TypeError, AttributeError):
            return None

    # ── Fetch ─────────────────────────────────────────────────────────────

    def fetch_changed_files(
        self, repo_url: str, pr_number: int
    ) -> list[dict]:
        """Fetch changed files for a PR via the Gitea REST API.

        Calls ``GET /api/v1/repos/{owner}/{repo}/pulls/{index}/files``.

        Args:
            repo_url: Clone URL (e.g., ``https://gitea.example.com/owner/repo.git``).
            pr_number: Pull request number.

        Returns:
            List of file dicts from the Gitea API (``Filename``, ``Status``, etc).

        Raises:
            httpx.HTTPError: On API request failure.
        """
        slug = self._repo_slug(repo_url)

        url = f"{self._base_url}/api/v1/repos/{slug}/pulls/{pr_number}/files"

        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"

        with httpx.Client() as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _repo_slug(repo_url: str) -> str:
        """Extract ``owner/repo`` from a Gitea clone URL.

        Args:
            repo_url: Full clone URL.

        Returns:
            ``owner/repo`` slug.
        """
        url = repo_url
        # Strip protocol + host (e.g., https://gitea.example.com/)
        if "://" in url:
            url = url.split("://", 1)[1]
        # Remove host portion
        if "/" in url:
            url = url.split("/", 1)[1]

        # Strip trailing .git
        if url.endswith(".git"):
            url = url[:-4]

        # Strip trailing slash
        return url.rstrip("/")

    # ── Orchestration ─────────────────────────────────────────────────────

    def handle_webhook(
        self, headers: dict[str, str], body: dict, raw_body: bytes | None = None
    ) -> WebhookPayload | None:
        """Handle an incoming Gitea webhook: validate → parse → fetch → normalize.

        Args:
            headers: HTTP request headers.
            body: Parsed JSON body.

        Returns:
            Normalized :class:`WebhookPayload` on success, ``None`` on failure
            (invalid HMAC, non-merge event, or API error).
        """
        # 1. Validate HMAC signature
        lower_headers = {k.lower(): v for k, v in headers.items()}
        signature = lower_headers.get("x-gitea-signature", "")
        # Validate against the RAW request bytes Gitea signed; fall back to
        # re-serializing only when raw_body isn't supplied (back-compat).
        payload_bytes = raw_body if raw_body is not None else json.dumps(body).encode()

        if not validate_hmac(self._secret, payload_bytes, signature):
            return None

        # 2. Parse merge event
        parsed = self.parse_merge_event(body)
        if parsed is None:
            return None

        repo_url, branch, pr_number = parsed

        # 3. Fetch changed files from Gitea API
        try:
            raw_files = self.fetch_changed_files(repo_url, pr_number)
        except Exception:
            return None

        # 4. Normalize using domain logic
        normalized = normalize_changed_files(Provider.GITEA, raw_files)

        return WebhookPayload(
            provider=Provider.GITEA,
            repo_url=repo_url,
            branch=branch,
            changed_files=normalized,
        )
