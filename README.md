# Skin Cancer Segmentation

Research-grade dermoscopic lesion segmentation with evolutionary model search,
heterogeneous ensembles, uncertainty estimation, and a deployable inference API.

## Project Status

**Current milestone: baseline foundation in progress.** The Python 3.11
environment, configuration contract, deterministic seeding, preprocessing
helpers, split-driven dataset loading, safe losses and metrics, a compact
smoke-test model, configurable U-Net++, and quality gates are implemented. The
next milestone is preprocessing ISIC 2018 and running the first reproducible
baseline. Results will be added only when they can be reproduced from a
committed configuration.

## Research Question

Can evolutionary optimization of segmentation architectures, loss recipes, and
ensemble weights improve lesion-boundary quality while keeping inference practical?

## Quickstart

This project uses Python 3.11 for broad compatibility with the scientific stack.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

Install the research stack when the baseline implementation is ready:

```powershell
python -m pip install -e ".[research,evolution,mlops]"
```

## Reproducibility Principles

- Configuration is loaded from `configs/`; paths and experiment settings are not
  hardcoded in source code.
- Raw datasets and model artifacts remain outside Git history.
- Every reported result will include its configuration, seed, split, and command.
- Failed or inconclusive experiments are recorded in the experiment notes.

## Planned Workflow

1. Validate configuration, determinism, data contracts, and a synthetic smoke run.
2. Establish a U-Net++ baseline on ISIC 2018 Task 1.
3. Add heterogeneous models, calibrated uncertainty, and test-time augmentation.
4. Run evolutionary search and ensemble ablations with MLflow tracking.
5. Export the selected model to ONNX and serve it through FastAPI.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the complete technical contract.
See [docs/DATASETS.md](docs/DATASETS.md) for the staged dataset-download plan.
