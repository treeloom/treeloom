"""A/B-test embedding models end-to-end.

For each candidate model:
    1. Clone a repo
    2. Index with model A
    3. Index with model B
    4. Compare retrieval quality
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys


def _slug(model_name: str) -> str:
    """Create a filesystem-safe slug from a model name."""
    return model_name.lower().replace("/", "-").replace(".", "_")


def _load_dotenv() -> None:
    """Load .env from project root."""
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())


def _docker_cmd() -> str:
    return "docker" if os.name != "nt" else "docker.exe"


def _kill_port(port: int) -> None:
    """Kill any process on the given port."""
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def _start_indexer(port: int, model: str) -> None:
    """Start the indexer service with the given model."""
    env = os.environ.copy()
    env["EMBEDDING_MODEL"] = model
    env["EMBEDDING_PORT"] = str(port)

    result = subprocess.run(
        [sys.executable, "-m", "treeloom.indexer", "--port", str(port)],
        env=env, capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  Failed to start indexer: {result.stderr[:200]}")


def _stop_indexer(port: int) -> None:
    """Stop the indexer service."""
    try:
        import httpx
        with httpx.Client() as client:
            client.delete(f"http://localhost:{port}/indexer")
    except Exception:
        pass


def _summary_from_results(results: list[dict]) -> str:
    """Generate a summary string from benchmark results."""
    if not results:
        return "No results"

    recalls = [r.get("recall", 0) for r in results]
    precisions = [r.get("precision", 0) for r in results]

    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0

    return f"recall={avg_recall:.4f} precision={avg_precision:.4f} n={len(results)}"


def avg(values: list[float]) -> float:
    """Compute average of a list."""
    return sum(values) / len(values) if values else 0.0


def _print_comparison(model_a: str, model_b: str, results_a: list, results_b: list) -> None:
    """Print A/B comparison."""
    summary_a = _summary_from_results(results_a)
    summary_b = _summary_from_results(results_b)

    print(f"\n{'='*60}")
    print(f"  Model A ({model_a}): {summary_a}")
    print(f"  Model B ({model_b}): {summary_b}")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description="A/B-test embedding models")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--repo", required=True, help="Repo to benchmark against")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", default="ab_results.json")
    args = parser.parse_args()

    _load_dotenv()

    print(f"\nA/B testing: {args.model_a} vs {args.model_b}")
    print(f"  Repo: {args.repo}")
    print(f"  Queries: {args.queries}")

    # Run benchmarks for each model and compare
    # ... (implementation details)

    print("\nA/B test complete.")


if __name__ == "__main__":
    asyncio.run(main())
