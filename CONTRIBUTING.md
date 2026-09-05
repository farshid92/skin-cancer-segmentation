# Contributing

This repository treats reproducibility as part of the implementation. Small,
reviewable changes are preferred over large feature drops.

## Development Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
```

Run the local checks before opening a pull request:

```powershell
ruff check .
black --check .
mypy src
pytest -q --cov=src --cov-fail-under=70
```

## Pull Requests

Describe the motivation, the behavior changed, and the command used to verify
it. For experiments, include the configuration, seed, dataset split, and a short
interpretation of the result. Do not commit datasets, credentials, model weights,
or local experiment output.
