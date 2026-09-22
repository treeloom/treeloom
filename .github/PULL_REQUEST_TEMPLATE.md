## Summary

<!-- What changes and why. Link the issue if there is one. -->

## Kind of change

- [ ] Docs only
- [ ] Bug fix
- [ ] Feature or behavior change
- [ ] Retrieval-quality change (anything that could move search results: chunking, embedding, reranking, graph scoring, query expansion, search flags)
- [ ] Storage schema or migration

## Verification

<!-- The concrete check you ran, with its result. "Should work" is not a result. -->

- [ ] `env -u PYTHONPATH python -m pytest tests/unit -q` passes locally
- [ ] Retrieval-quality changes only: benchmark run attached or linked (`python -m treeloom.benchmark run ...`, see docs/benchmark-eval.md), including the reference set and the before/after recall@k / MRR
- [ ] Schema changes only: migration added and the re-index requirement stated in the PR description

## Checklist

- [ ] No secrets, internal hostnames, or private tracker references in the diff
- [ ] Docs updated where behavior changed (README, docs/, CLAUDE.md invariants)
- [ ] Conventional-commit style title (`feat:`, `fix:`, `docs:`, `chore:` ...)
