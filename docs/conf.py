"""Sphinx configuration for the Treeloom docs (Read the Docs).

The docs are plain Markdown under ``docs/``; MyST renders them. Relative
``foo.md`` links between docs resolve to Sphinx cross-references, so the
files keep working when read on GitHub as well.
"""

project = "Treeloom"
author = "Treeloom contributors"
copyright = "2026, Treeloom contributors"

extensions = ["myst_parser"]

source_suffix = {".md": "markdown"}
master_doc = "index"

# Local scratch/agent state that lives under docs/ but is not documentation.
exclude_patterns = ["_build", "superpowers", "requirements.txt", "Thumbs.db", ".DS_Store"]

# Let ``file.md#some-heading`` links resolve for headings up to level 3.
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "tasklist"]

html_theme = "furo"
html_title = "Treeloom"
html_static_path = []
