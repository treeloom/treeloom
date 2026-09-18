"""XSS via indexed content: there is no sink, and this keeps it that way.

Indexed repositories are attacker-influenced, and repository-derived strings
(file paths, error text, `current_file`, source names) do reach the operator
dashboard. That is only safe because of two structural facts:

1. The indexer API never returns HTML — every response is JSON, so there is
   no server-side template to inject into.
2. The dashboard is React, which escapes interpolated text, and it uses no
   HTML-bypassing sink and no markdown/syntax-highlight renderer.

Neither is enforced by anything else, and both are one commit from changing:
a `dangerouslySetInnerHTML` added to render a snippet nicely, or an
`HTMLResponse` added for a status page, would turn indexed content into
stored XSS against an operator session.

This guard lives in the Python suite rather than the UI's vitest suite on
purpose: it covers both halves of the invariant, and it runs in the suite that
gates this repo's changes.
"""

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Bypass React's escaping, or execute a string as code.
JS_SINKS = (
    "dangerouslySetInnerHTML", "innerHTML", "outerHTML", "insertAdjacentHTML",
    "document.write", "new Function(", "srcdoc",
)

# Would make the API serve HTML built from server-side strings.
PY_HTML = ("HTMLResponse", "Jinja2Templates", "TemplateResponse")


def _sources(root: pathlib.Path, suffixes: tuple[str, ...]):
    if not root.exists():
        pytest.skip(f"{root} not present")
    for p in root.rglob("*"):
        if p.suffix in suffixes and "node_modules" not in p.parts:
            yield p


class TestTheDashboardHasNoHtmlBypass:
    def test_no_sink_in_ui_sources(self):
        hits = [
            f"{p.relative_to(REPO)}: {sink}"
            for p in _sources(REPO / "ui" / "src", (".ts", ".tsx", ".js", ".jsx"))
            for sink in JS_SINKS
            if sink in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert not hits, (
            "HTML-bypassing sink in the dashboard — indexed content reaches "
            f"these views and React's escaping is the only thing stopping XSS: {hits}"
        )


class TestTheApiServesNoHtml:
    def test_no_html_response_in_the_indexer(self):
        hits = [
            f"{p.relative_to(REPO)}: {marker}"
            for p in _sources(REPO / "src" / "treeloom", (".py",))
            for marker in PY_HTML
            if marker in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert not hits, f"API returning HTML built from server strings: {hits}"
