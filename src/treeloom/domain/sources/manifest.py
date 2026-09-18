"""Fleet onboarding manifest — pure parsing, no I/O.

A manifest declares a set of repositories to index in one shot (bulk onboarding). It is parsed here into ``ManifestEntry`` value objects; the
``treeloom.fleet`` CLI turns each entry into a ``POST /index-repo`` call.

Two accepted shapes (``yaml.safe_load`` parses both YAML and JSON):

    # Mapping form, with optional shared defaults merged into every entry
    defaults:
      branch: main
      skip_graph: false
    repos:
      - path: ~/source/featbit
      - url: https://github.com/acme/widgets.git
        branch: release
        skip_patterns: ["vendor/**"]

    # Bare-list form (no defaults)
    - path: ~/source/a
    - url: https://github.com/acme/b.git

Each entry must carry exactly one of ``path`` / ``url``. Git/URL safety is NOT
validated here — that stays server-side in ``/index-repo`` (``_is_safe_git_url``);
this module only shapes operator input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

# Keys an entry (or the defaults block) may carry. Anything else is rejected so
# typos surface loudly instead of being silently ignored.
_ALLOWED_KEYS = {"path", "url", "branch", "skip_patterns", "skip_graph", "force"}


@dataclass
class ManifestEntry:
    """One repository to onboard. Exactly one of ``path``/``url`` is set."""

    path: str = ""
    url: str = ""
    branch: str = ""
    skip_patterns: list[str] = field(default_factory=list)
    skip_graph: bool = False
    force: bool = False

    @property
    def label(self) -> str:
        """Human-readable identifier for logs/output."""
        return self.url or self.path

    def to_index_request(self, *, force_override: bool = False) -> dict[str, Any]:
        """Build the JSON body for ``POST /index-repo``.

        ``force_override`` (the CLI's global ``--force``) ORs with the entry's
        own ``force`` so a single flag can re-index the whole manifest.
        """
        body: dict[str, Any] = {"force": self.force or force_override}
        if self.path:
            body["path"] = self.path
        if self.url:
            body["url"] = self.url
        if self.branch:
            body["branch"] = self.branch
        if self.skip_patterns:
            body["skip_patterns"] = list(self.skip_patterns)
        if self.skip_graph:
            body["skip_graph"] = True
        return body


def _coerce_entry(raw: Any, *, index: int, defaults: dict[str, Any]) -> ManifestEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"manifest entry {index}: expected a mapping, got {type(raw).__name__}")

    merged = {**defaults, **raw}
    unknown = set(merged) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"manifest entry {index}: unknown key(s) {sorted(unknown)}; "
            f"allowed: {sorted(_ALLOWED_KEYS)}"
        )

    path = str(merged.get("path") or "").strip()
    url = str(merged.get("url") or "").strip()
    if bool(path) == bool(url):
        raise ValueError(
            f"manifest entry {index}: provide exactly one of `path` or `url` "
            f"(got path={path!r}, url={url!r})"
        )

    skip_patterns_raw = merged.get("skip_patterns") or []
    if not isinstance(skip_patterns_raw, list) or not all(
        isinstance(p, str) for p in skip_patterns_raw
    ):
        raise ValueError(f"manifest entry {index}: `skip_patterns` must be a list of strings")

    return ManifestEntry(
        path=path,
        url=url,
        branch=str(merged.get("branch") or "").strip(),
        skip_patterns=list(skip_patterns_raw),
        skip_graph=bool(merged.get("skip_graph", False)),
        force=bool(merged.get("force", False)),
    )


def parse_manifest(text: str) -> list[ManifestEntry]:
    """Parse a YAML/JSON manifest into ``ManifestEntry`` objects.

    Raises ``ValueError`` on malformed input (with the offending entry index).
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"manifest is not valid YAML/JSON: {exc}") from exc

    if data is None:
        return []

    if isinstance(data, dict):
        defaults = data.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise ValueError("manifest `defaults` must be a mapping")
        unknown_defaults = set(defaults) - _ALLOWED_KEYS
        if unknown_defaults:
            raise ValueError(
                f"manifest `defaults`: unknown key(s) {sorted(unknown_defaults)}"
            )
        repos = data.get("repos")
        if repos is None:
            raise ValueError("manifest mapping must contain a `repos` list")
        if not isinstance(repos, list):
            raise ValueError("manifest `repos` must be a list")
    elif isinstance(data, list):
        defaults = {}
        repos = data
    else:
        raise ValueError(
            f"manifest must be a mapping or a list, got {type(data).__name__}"
        )

    return [
        _coerce_entry(raw, index=i, defaults=defaults) for i, raw in enumerate(repos)
    ]
