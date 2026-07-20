# Assignment 1 - ML From Scratch

Educational NumPy implementations of classical machine learning models and
basic neural-network components.

This project is adapted from Stanford CS231n homework-assignment notebooks. The
core assignment-style structure is intentionally preserved, while the code and
demos are organized so the work can live as a standalone GitHub repository.
The Python source files intentionally stay close to the original assignment
structure; changes are limited to local path/package adaptation and repository
organization unless a small fix is needed for local use.

## What This Repo Is

- A learning repository for implementing ML models from first principles.
- A notebook-friendly codebase for experimenting with CIFAR-10 classifiers.
- A personal adaptation and cleanup of CS231n-style homework scaffolding.

## What This Repo Is Not

- A production ML framework.
- An optimized deep-learning library.
- A replacement for the original CS231n course material.

## Implemented Topics

- k-Nearest Neighbor (kNN)
- Softmax linear classifiers
- Two-layer fully connected neural networks
- Multi-layer fully connected networks
- Affine, ReLU, dropout, normalization, convolution, and pooling layers
- SGD, momentum, RMSProp, and Adam optimizers
- Hand-crafted image features such as HOG and HSV histograms

## Repository Structure

- `src/` - reusable ML implementations
- `demos/` - notebooks demonstrating how to use the models
- `tests/` - lightweight smoke tests for core behavior
- `data/` - local datasets, ignored by git
- `artifacts/` - saved models and outputs, ignored by git

## Setup

Use the project-level virtual environment at the repository root so kernels and
dependencies stay consistent across assignments.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r assignment1/requirements.txt
```

The demos expect CIFAR-10-style data under a local data directory. Dataset files
are intentionally not tracked by git.

## Run Checks

```bash
python -m pytest
```

For a quick syntax-only check:

```bash
PYTHONPYCACHEPREFIX=/tmp/mlfs_pycache python -m compileall -q src tests
```

## Demos

The notebooks in `demos/` mirror the original assignment progression:

1. kNN classifier
2. Softmax classifier
3. Two-layer network
4. Hand-crafted image features
5. Fully connected networks

The per-demo PDFs and `demos/demos_combined.pdf` are tracked as GitHub-readable
snapshots. To regenerate the combined showcase PDF from the notebooks, run:

```bash
cd demos
bash collect_demos.sh
```

The generated code bundle zip is ignored because it is a packaging artifact.

## Attribution

This repository is derived from Stanford CS231n assignment materials. The
underlying course content and assignment scaffolding belong to their original
authors. My contributions are the adaptation, cleanup, organization, and any
additional tests or demos in this repository.
