"""Benchmark domain: query loading and filtering."""
import json


def load_queries(
    path: str, difficulty: str = "", strategy: str = ""
) -> list[dict]:
    """Load queries from a JSONL file, optionally filtering by difficulty/strategy."""
    queries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if difficulty and difficulty not in q.get("difficulty", ""):
                continue
            if strategy and strategy not in q.get("strategy", ""):
                continue
            queries.append(q)
    return queries
