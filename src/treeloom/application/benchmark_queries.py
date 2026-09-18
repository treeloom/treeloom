"""Auto-generate benchmark queries from Neo4j entities.

For each Class/Function in the graph, generates a natural-language query
from the entity name/signature and records the defining file as ground truth.
"""
import argparse
import asyncio
import json
import os
import re

import httpx


def _camel_to_words(name: str) -> str:
    """Convert CamelCase to space-separated words."""
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s1).lower().replace('_', ' ')


def _make_query(name: str, file_path: str, entity_type: str = "") -> str:
    """Create a natural-language query from entity metadata."""
    words = _camel_to_words(name)
    if entity_type:
        return f"Where is the {entity_type} '{name}' ({words}) defined in {file_path}?"
    return f"Where is {name} ({words}) defined?"


def _short_file_name(path: str) -> str:
    """Extract the most relevant part of a file path."""
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-3:]) if len(parts) > 3 else parts[-1]


def _derive_grep_patterns(query: str) -> list[str]:
    """Derive grep patterns from a query."""
    words = query.lower().split()
    patterns = []
    for w in words:
        if len(w) > 2:
            patterns.append(w)
        elif len(w) == 2 and w[0].isalpha():
            patterns.append(f"{w[0].upper()}{w[1:]}")
    return list(set(patterns))


def _derive_glob_patterns(query: str) -> list[str]:
    """Derive glob patterns from a query."""
    # Look for file extensions or name patterns
    patterns = []
    for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
        base = query.lower().split()[-1]
        patterns.append(f"*{base.lower()}{ext}")
    return patterns


async def main():
    parser = argparse.ArgumentParser(description="Generate benchmark queries from Neo4j graph")
    parser.add_argument("--repo", required=True, help="Neo4j URI")
    parser.add_argument("--output", default="benchmarks/queries/generated.jsonl")
    parser.add_argument("--max-per-type", type=int, default=0)
    args = parser.parse_args()

    print(f"Connecting to Neo4j at {args.repo}...")

    # Connect to Neo4j and extract entities
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{args.repo}/query",
                json={
                    "statement": """
                    MATCH (n)
                    WHERE n.id IS NOT NULL AND n.type IN ['Class', 'Function', 'Method']
                    RETURN n.id AS id, n.name AS name, n.type AS type,
                           n.file_path AS file_path, n.signature AS signature
                    LIMIT 100
                    """,
                },
            )
            resp.raise_for_status()
            entities = resp.json().get("results", [])
        except Exception as e:
            print(f"Error: {e}")
            return

    queries = []
    for entity in entities:
        name = entity.get("name", "")
        file_path = entity.get("file_path", "")
        entity_type = entity.get("type", "")

        if not name or not file_path:
            continue

        if args.max_per_type > 0:
            # Count existing queries for this type
            count = sum(1 for q in queries if q.get("entity_type") == entity_type)
            if count >= args.max_per_type:
                continue

        query_text = _make_query(name, file_path, entity_type)
        gt = _short_file_name(file_path)

        queries.append({
            "query": query_text,
            "ground_truth": [gt],
            "entity_name": name,
            "entity_type": entity_type,
            "entity_file": file_path,
            "difficulty": "easy" if entity_type == "Function" else "medium",
            "strategy": "grep+read",
        })

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    print(f"Generated {len(queries)} queries → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
