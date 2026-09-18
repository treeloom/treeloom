"""GitHub webhook adapter.

Implements webhook handling for GitHub pull_request merge events.
Uses domain/webhook.py for HMAC validation and file normalization.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from treeloom.domain.webhook import Provider, WebhookPayload, normalize_changed_files, validate_hmac

if TYPE_CHECKING:
    from typing import Optional


class GitHubAdapter:
    """Adapts GitHub webhooks to a normalized :class:`WebhookPayload`.

    Handles:
    - HMAC validation (X-Hub-Signature-256 header)
    - Parse pull_request.merged events
    - Fetch changed files via GitHub REST API
    - Normalize to domain payload
    """

    def __init__(
        self,
        secret: str,
        token: Optional[str] = None,
        trust_inline_files: bool = False,
    ) -> None:
        """Initialize the GitHub adapter.

        Args:
            secret: Shared HMAC secret for webhook signature validation.
            token: Optional GitHub API token for authenticated API requests.
            trust_inline_files: TEST/LOAD mode (gated by WEBHOOK_TRUST_INLINE_FILES,
                default off). When true, read the changed-file list from the
                webhook payload (``pull_request.files``) instead of calling the
                GitHub API. Lets a load generator drive sustained synthetic
                traffic without real PRs or API rate limits. NEVER enable in
                production — the payload's file list is not authenticated beyond
                the HMAC signature, so a signer could claim arbitrary paths.
        """
        self._secret = secret
        self._token = token
        self._trust_inline_files = trust_inline_files

    # ── Parse ─────────────────────────────────────────────────────────────

    def parse_merge_event(self, body: dict) -> tuple[str, str, int] | None:
        """Extract repo_url, branch, and PR number from a GitHub merge webhook body.

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
        """Fetch changed files for a PR via the GitHub REST API.

        Calls ``GET /repos/{owner}/{repo}/pulls/{number}/files``.

        Args:
            repo_url: Clone URL (e.g., ``https://github.com/owner/repo.git``).
            pr_number: Pull request number.

        Returns:
            List of file dicts from the GitHub API (``filename``, ``status``, etc).

        Raises:
            httpx.HTTPError: On API request failure.
        """
        # Extract owner/repo from the clone URL
        # e.g., https://github.com/owner/repo.git → owner/repo
        slug = self._repo_slug(repo_url)

        url = f"https://api.github.com/repos/{slug}/pulls/{pr_number}/files"

        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        with httpx.Client() as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _repo_slug(repo_url: str) -> str:
        """Extract ``owner/repo`` from a GitHub clone URL.

        Args:
            repo_url: Full clone URL.

        Returns:
            ``owner/repo`` slug.
        """
        # Strip protocol
        url = repo_url
        for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
            if prefix in url:
                url = url.split(prefix, 1)[1]
                break

        # Strip trailing .git
        if url.endswith(".git"):
            url = url[:-4]

        # Strip trailing slash
        return url.rstrip("/")

    # ── Orchestration ─────────────────────────────────────────────────────

    def handle_webhook(
        self, headers: dict[str, str], body: dict, raw_body: bytes | None = None
    ) -> WebhookPayload | None:
        """Handle an incoming GitHub webhook: validate → parse → fetch → normalize.

        Args:
            headers: HTTP request headers.
            body: Parsed JSON body.

        Returns:
            Normalized :class:`WebhookPayload` on success, ``None`` on failure
            (invalid HMAC, non-merge event, or API error).
        """
        # 1. Validate HMAC signature
        lower_headers = {k.lower(): v for k, v in headers.items()}
        signature = lower_headers.get("x-hub-signature-256", "")
        # GitHub signs the RAW request body. Validate against those exact bytes
        # when available; only fall back to re-serializing the parsed dict when
        # the caller didn't supply raw_body (keeps older call sites/tests working).
        payload_bytes = raw_body if raw_body is not None else json.dumps(body).encode()

        if not validate_hmac(self._secret, payload_bytes, signature):
            return None

        # 2. Parse merge event
        parsed = self.parse_merge_event(body)
        if parsed is None:
            return None

        repo_url, branch, pr_number = parsed

        # 3. Resolve changed files. Normally fetched from the GitHub API; in
        # the gated inline mode read them straight off the payload so synthetic
        # load (no real PR / no rate limit) can exercise the pipeline.
        if self._trust_inline_files:
            raw_files = body.get("pull_request", {}).get("files", []) or []
        else:
            try:
                raw_files = self.fetch_changed_files(repo_url, pr_number)
            except Exception:
                return None

        # 4. Normalize using domain logic
        normalized = normalize_changed_files(Provider.GITHUB, raw_files)

        return WebhookPayload(
            provider=Provider.GITHUB,
            repo_url=repo_url,
            branch=branch,
            changed_files=normalized,
        )
