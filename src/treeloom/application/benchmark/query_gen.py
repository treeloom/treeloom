"""LLM-based query generation for benchmark evaluation."""
from __future__ import annotations

import os
import json
from pathlib import Path

from treeloom.adapters.llm_api.llm_adapter import _chat


async def generate_queries(source_dir: str, n: int = 5) -> list[dict]:
    """Generate N developer-style queries with ground-truth file paths.

    Uses the LLM to create realistic code-search questions that a developer
    might ask about the codebase, along with which files contain the answer.
    """
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {source_dir}")

    # List files (top 2 levels, skip hidden/dirs)
    files = []
    for fp in sorted(root.rglob("*")):
        if fp.name.startswith(".") or fp.is_dir():
            continue
        rel = str(fp.relative_to(root))
        if rel.count("/") <= 1:  # top 2 levels
            files.append(rel)
    if len(files) > 100:
        files = files[:100]  # cap for prompt size

    file_list = "\n".join(f"  - {f}" for f in files[:50])

    prompt = f"""You are a developer searching a codebase. Given these files:

{file_list}

Generate {n} realistic code-search queries that a developer might ask, along
with the EXACT files (from the list above) that contain the answer.

Return ONLY valid JSON as a list:
[
  {{"query": "How does <thing> handle <behavior>?",
    "ground_truth": ["file1.py", "file2.py"],
    "language": "python"}}
]

Rules:
- ground_truth MUST use filenames from the list above
- Queries should be natural developer questions
- Cover different aspects: API usage, error handling, config, routing, data flow
- Use the repository name "{root.name}" for context
"""

    messages = [
        {"role": "system", "content": "You are a helpful code search assistant."},
        {"role": "user", "content": prompt},
    ]
    response = await _chat(messages, max_tokens=2000)
    if not response:
        return _fallback_queries(files, n)

    try:
        # Parse JSON from response
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        queries = json.loads(text)
        # Validate ground_truth
        for q in queries:
            q["ground_truth"] = [g for g in q.get("ground_truth", [])
                                if g in files or any(g in f for f in files)]
        return queries[:n]
    except (json.JSONDecodeError, KeyError):
        return _fallback_queries(files, n)


def _fallback_queries(files: list[str], n: int) -> list[dict]:
    """Generate simple fallback queries when LLM fails."""
    if not files:
        return []
    langs = {"py": "python", "go": "go", "js": "javascript", "ts": "typescript",
             "java": "java", "cpp": "cpp", "rb": "ruby"}
    queries = []
    for i in range(min(n, len(files))):
        f = files[i]
        ext = f.rsplit(".", 1)[-1] if "." in f else ""
        lang = langs.get(ext, "code")
        queries.append({
            "query": f"How does the code in {f} work?",
            "ground_truth": [f],
            "language": lang,
        })
    return queries
