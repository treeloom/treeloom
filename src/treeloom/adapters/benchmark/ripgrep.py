"""Benchmark adapter: ripgrep subprocess wrapper."""
import subprocess


def run_ripgrep(
    repo: str, pattern: str, max_matches: int = 200
) -> list[tuple[str, int, str]]:
    """Run ripgrep and return (file_path, line_no, matched_line)."""
    try:
        result = subprocess.run(
            [
                "rg", "--no-heading", "--with-filename", "--line-number",
                "--max-count", str(max_matches), "--smart-case",
                "--no-ignore-vcs", pattern, repo,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode in (0, 1):
            matches: list[tuple[str, int, str]] = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        line_no = int(parts[1])
                    except ValueError:
                        continue
                    matches.append((parts[0], line_no, parts[2]))
            return matches
        return []
    except Exception:
        return []
