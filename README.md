# ML From Scratch

Personal GitHub adaptation of Stanford CS231n-style assignments, organized as
standalone local projects with readable notebook/PDF snapshots.

## Structure

- `assignment0/` - tiny plain-Python fully connected neural network toy example
- `assignment1/` - kNN, softmax classifiers, two-layer nets, fully connected nets, hand-crafted features
- `assignment2/` - normalization, dropout, convolutional nets, PyTorch, and RNN captioning

Each assignment keeps the Python source close to the original CS231n assignment
layout. Changes should stay limited to local path/package adaptation, repository
organization, docs, tests, and generated showcase artifacts.

## Assignment 1

```bash
cd assignment1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

The demo notebooks and GitHub-readable PDF snapshots live in
`assignment1/demos/`. The combined showcase PDF is
`assignment1/demos/demos_combined.pdf`.

## Assignment 2

Assignment 2 currently includes the locally adapted Q1 batch normalization
notebook. Continue building it from the original CS231n assignment files using
the same pattern as Assignment 1: preserve source structure, adapt local
paths/imports only where needed, and add demo snapshots after the notebooks are
complete.

## Attribution

This repository is derived from Stanford CS231n assignment materials. The
underlying course content and assignment scaffolding belong to their original
authors. My contributions are the adaptation, cleanup, organization, and any
additional tests or demos in this repository.
