# ML From Scratch

Personal GitHub adaptation of Stanford CS231n-style assignments, organized as
standalone local projects with readable notebook/PDF snapshots.

See `COURSE_BACKBONE.md` for a concise summary of the course progression across
the three CS231n assignments.

## Structure

- `assignment0/` - tiny plain-Python fully connected neural network toy example
- `assignment1/` - kNN, linear classifiers, two-layer nets, fully connected nets, hand-crafted features
- `assignment2/` - normalization, dropout, convolutional nets, PyTorch, and vanilla RNN captioning
- `assignment3/` - Transformers, Vision Transformers, SimCLR, CLIP/DINO, and DDPMs
- `COURSE_BACKBONE.md` - high-level course backbone distilled from Assignments 1-3

Each assignment keeps the Python source close to the original CS231n assignment
layout. Changes should stay limited to local path/package adaptation, repository
organization, docs, tests, and generated showcase artifacts.

## Assignment 1

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r assignment1/requirements.txt
python -m pytest assignment1
```

The demo notebooks and GitHub-readable PDF snapshots live in
`assignment1/demos/`. The combined showcase PDF is
`assignment1/demos/demos_combined.pdf`.

## Assignment 2

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r assignment2/requirements.txt
```

The adapted demos cover batch/layer normalization, dropout, convolutional
networks, PyTorch on CIFAR-10, and vanilla RNN image captioning. See
`assignment2/README.md` for tests, notebook status, packaging, and rerun
commands.

## Assignment 3

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r assignment3/requirements.txt
```

Use this project-level `.venv` for the default notebook kernels across
assignments. If you keep an Apple Silicon/MPS environment, place it beside it as
`.venv-arm` at the same project root.

The adapted demos cover Transformer captioning, Vision Transformers,
self-supervised learning with SimCLR, CLIP/DINO representation learning, and
diffusion models. Some tracks require large downloads, GPU-oriented workflows,
or the CLIP package from GitHub; see `assignment3/README.md` for details.

## Course Arc

The three main assignments form a progression:

1. Classical image classification and explicit numerical optimization.
2. Neural-network layers, backpropagation, optimizers, and training checks.
3. Practical deep-learning tools such as normalization, dropout, CNNs, and
   PyTorch.
4. Sequence and attention models for image captioning and Vision Transformers.
5. Self-supervised, multimodal, and generative visual modeling.

## Attribution

This repository is derived from Stanford CS231n assignment materials. The
underlying course content and assignment scaffolding belong to their original
authors. My contributions are the adaptation, cleanup, organization, and any
additional tests or demos in this repository.
