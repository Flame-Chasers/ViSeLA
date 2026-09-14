# Contributing to ViSeLA

Thank you for your interest in ViSeLA.

## Reporting Issues

1. Search existing issues first.
2. Use the bug report or question template.
3. Include the command, config path, dataset, checkpoint source, Python/PyTorch/CUDA versions, and the relevant error log.

## Submitting Changes

1. Fork the repository.
2. Create a focused branch.
3. Keep changes scoped to one fix, experiment, or documentation update.
4. Run the smallest relevant local check.
5. Open a pull request with the motivation, changed files, and verification.

Run the unit tests from the repository root:

```bash
python -B -m unittest discover -s tests -p 'test_*.py'
```

## Code Style

- Follow the existing research-code patterns.
- Preserve config keys and experiment semantics unless the PR explicitly changes them.
- Avoid broad refactors mixed with result-reproduction changes.
- Do not commit datasets, checkpoints, logs, or local machine paths.

## Questions

Use the question issue template with enough config and environment context for maintainers to reproduce the setup.
