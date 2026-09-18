# Contributing to Treeloom

Thanks for your interest! Bug reports, benchmark results on new repos/languages,
and PRs are all welcome.

## Development setup

```bash
git clone https://github.com/treeloom/treeloom.git && cd treeloom
# dev alone is test-tooling only (pytest); the full suite also exercises
# every optional-backend adapter, so pull in all-backends + simple too.
pip install -e ".[dev,all-backends,simple]"
pytest tests/unit -q        # no services needed
```

Running the full stack locally is described in the README (Quick Start).

## Ground rules

- **Architecture**: DDD layering — `domain/` never imports `adapters/`.
  New vector/graph/reranker backends implement the existing port surface;
  see CLAUDE.md for the dispatch pattern.
- **Tests first**: every change lands with Detroit-style unit tests
  (`tests/unit/`, mock only external services). CI must be green.
- **Retrieval changes need evidence**: anything that could move search
  quality gets a benchmark run (`python -m treeloom.benchmark run`) —
  see docs/benchmark-eval.md for the workflow.
- **Commits**: conventional-commit style (`feat:`, `fix:`, `chore:`, ...).

## Reporting bugs

Open a GitHub issue with repro steps, expected vs actual behavior, and
relevant log output (`docker compose logs`, indexer stdout).
