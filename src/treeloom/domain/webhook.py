"""Webhook domain — Provider enumeration, normalized payload, detection, and HMAC validation.

DDD: domain layer — NEVER imports from adapters/ or application/.
"""
from __future__ import annotations

import enum
import hashlib
import hmac
from dataclasses import dataclass, field


# ── Provider Enum ────────────────────────────────────────────────────────────


class Provider(str, enum.Enum):
    """Supported webhook providers."""

    GITHUB = "github"
    GITEA = "gitea"


# ── WebhookPayload ───────────────────────────────────────────────────────────


@dataclass
class WebhookPayload:
    """Normalized webhook payload across all providers.

    Attributes:
        provider: The detected provider.
        repo_url: Canonical clone URL (e.g., https://github.com/owner/repo.git).
        branch: Target branch of the merge.
        changed_files: List of dicts with ``path`` (str) and ``action`` (str).
            Actions: ``"added"``, ``"modified"``, ``"removed"``, ``"renamed"``.
    """

    provider: Provider
    repo_url: str
    branch: str
    changed_files: list[dict[str, str]] = field(default_factory=list)


# ── Provider Detection ───────────────────────────────────────────────────────


def detect_provider(headers: dict[str, str], body: dict) -> Provider:
    """Detect the webhook provider from HTTP headers and body.

    Detection strategy:
        - ``X-GitHub-Event``  → GitHub
        - ``X-Gitea-Event``   → Gitea

    Any other delivery raises ``ValueError``, which the route turns into a
    clear 400 rather than routing it to an adapter that cannot handle it.

    Args:
        headers: Case-insensitive HTTP headers dict.
        body: Parsed JSON body as dict.

    Returns:
        The detected :class:`Provider`.

    Raises:
        ValueError: If no known provider signature is found.
    """
    # Normalise header keys to lowercase for case-insensitive matching
    lower_headers = {k.lower(): v for k, v in headers.items()}

    if "x-github-event" in lower_headers:
        return Provider.GITHUB
    if "x-gitea-event" in lower_headers:
        return Provider.GITEA

    raise ValueError("Unknown webhook provider: no recognised header or body field")


# ── HMAC Validation ──────────────────────────────────────────────────────────


def validate_hmac(secret: str, payload: bytes, signature: str) -> bool:
    """Validate an HMAC-SHA256 signature against a shared secret.

    The *signature* parameter may optionally be prefixed with ``sha256=``
    (GitHub convention).  SHA1 signatures (``sha1=`` prefix) are rejected.

    Args:
        secret: Shared secret / webhook token (empty string disables validation).
        payload: Raw request body bytes.
        signature: Signature string from the provider header.

    Returns:
        ``True`` if the signature matches or secret is empty and signature is
        empty; ``False`` otherwise.
    """
    if not secret and not signature:
        return True
    if not secret or not signature:
        return False

    # Reject sha1 signatures
    sig_lower = signature.lower()
    if sig_lower.startswith("sha1="):
        return False

    # Strip optional sha256= prefix
    if sig_lower.startswith("sha256="):
        sig_value = signature[7:]
    else:
        sig_value = signature

    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_value)


# ── File Normalisation ───────────────────────────────────────────────────────


def normalize_changed_files(
    provider: Provider | None, raw_files: list[dict]
) -> list[dict[str, str]]:
    """Normalise provider-specific changed-file representations to a common
    ``[{path, action}]`` format.

    ============== =============================== ===========================
    Provider       Input field(s)                   Output action
    ============== =============================== ===========================
    GitHub         ``filename``, ``status``         ``modified``, ``added``,
                                                    ``removed``
    Gitea          ``Filename``, ``Status``          ``modified``, ``added``,
                                                    ``removed``
    ============== =============================== ===========================

    Args:
        provider: The :class:`Provider` of the webhook.
        raw_files: Provider-specific file-change dicts.

    Returns:
        Normalised list of ``{"path": str, "action": str}`` dicts.

    Raises:
        ValueError: If *provider* is ``None``.
    """
    if provider is None:
        return []

    result: list[dict[str, str]] = []

    if provider == Provider.GITHUB:
        for f in raw_files:
            status = f.get("status", "modified")
            action = _map_github_status(status)
            result.append({"path": f.get("filename", ""), "action": action})

    elif provider == Provider.GITEA:
        for f in raw_files:
            status = f.get("Status", f.get("status", "changed"))
            action = _map_gitea_status(status)
            path = f.get("Filename", f.get("filename", ""))
            result.append({"path": path, "action": action})

    return result


# ── Internal helpers ─────────────────────────────────────────────────────────


def _map_github_status(status: str) -> str:
    """Map GitHub file status to normalised action."""
    mapping: dict[str, str] = {
        "modified": "modified",
        "changed": "modified",
        "added": "added",
        "removed": "removed",
        "renamed": "renamed",
    }
    return mapping.get(status, "modified")


def _map_gitea_status(status: str) -> str:
    """Map Gitea file status to normalised action."""
    mapping: dict[str, str] = {
        "changed": "modified",
        "modified": "modified",
        "added": "added",
        "removed": "removed",
        "renamed": "renamed",
        "deleted": "removed",
    }
    return mapping.get(status.lower(), "modified")
