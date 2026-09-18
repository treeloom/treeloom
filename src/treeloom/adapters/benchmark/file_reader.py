"""Benchmark adapter: file system reader for context windows."""


def read_context_snippets(
    file_path: str, match_lines: list[int], context: int
) -> list[dict]:
    """Read a file and extract context windows around match lines.

    Returns list of dicts with file_path, start_line, end_line, snippet.
    Adjacent matches share a single context window to avoid duplication.
    """
    snippets: list[dict] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception:
        return snippets

    seen_ranges: set[tuple[int, int]] = set()
    for ml in match_lines:
        start = max(0, ml - context - 1)
        end = min(len(all_lines), ml + context)
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        snippet_text = "".join(all_lines[start:end])
        snippets.append({
            "file_path": file_path,
            "start_line": start + 1,
            "end_line": end,
            "snippet": snippet_text,
        })
    return snippets
