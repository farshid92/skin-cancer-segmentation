# Research Decisions

This log records choices that affect reproducibility or interpretation. Each
entry should explain the decision, its reason, and what evidence may change it.

## 2026-09-04: Python 3.11

Python 3.11 is the project interpreter because it offers broad compatibility
across PyTorch, MONAI, scientific Python, and serving dependencies. The project
allows Python 3.10 through 3.14 in packaging metadata, but development is pinned
to 3.11 for consistent local and CI behavior.

## 2026-09-04: Baseline Before Search

The first milestone is a deterministic U-Net++ baseline and validated data/loss
contract. Evolutionary search will be added only after this path is measurable,
so optimization results can be separated from pipeline defects.
