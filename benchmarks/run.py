#!/usr/bin/env python3
"""Benchmark runner for treeloom retrieval — thin entry point.

Wires the application layer which orchestrates domain metrics + adapter I/O.

Usage:
    BENCHMARK_QUALITY=STANDARD python benchmarks/run.py \\
        --queries benchmarks/queries/featbit-clean-100.jsonl \\
        --mode all \\
        --repo /path/to/repo

    BENCHMARK_QUALITY=BASIC python benchmarks/run.py \\
        --queries benchmarks/queries/featbit-clean-100.jsonl \\
        --mode no-mcp \\
        --repo /path/to/repo
"""
import asyncio
import sys
from pathlib import Path

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from treeloom.application.benchmark.runner import main

if __name__ == "__main__":
    asyncio.run(main())
