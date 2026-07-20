# Assignment 3

CS231n-style Assignment 3 adaptation for this local `ml-from-scratch` repo.
The assignment keeps the original educational structure while adapting imports,
paths, datasets, and notebooks to run from this repository.

## Topics

- Image captioning with Transformers
- Vision Transformers on CIFAR-10
- Self-supervised learning with SimCLR
- CLIP and DINO visual representations
- Denoising diffusion probabilistic models

Keep the Python source close to the original assignment files. Prefer local
path/package adaptations, notebook cleanup, tests, and PDF snapshots over broad
source refactors.

## Setup

Use the project-level virtual environment at the repository root so kernels and
dependencies stay consistent across assignments.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r assignment3/requirements.txt
```

If you use the Apple Silicon/MPS environment, keep it beside `.venv` as
`.venv-arm` at the same project root and register the notebook kernel from that
interpreter.

The notebooks add the assignment root to `sys.path`, then import reusable code
from `src/`. Dataset scripts from the original assignment live under
`src/datasets/`; the local setup cells include commented commands for running
them when needed.

Some notebooks require large downloads or GPU-oriented packages/checkpoints.
Install the CLIP package separately if you plan to run the CLIP or DDPM tracks:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Notebook Status

The notebooks have been copied and locally adapted, but they have not been
executed in this repo yet:

| Notebook | Topic | Status |
| --- | --- | --- |
| `demos/01_transformer_captioning_repo.ipynb` | Transformers and ViT | Not run |
| `demos/02_self_supervised_learning_repo.ipynb` | SimCLR | Not run |
| `demos/03_clip_dino_repo.ipynb` | CLIP and DINO | Not run |
| `demos/04_ddpm_repo.ipynb` | DDPM | Not run |

## Notes

- The upstream `cs231n/` package has been copied to `src/`, and notebook imports
  have been rewritten from `cs231n.*` to `src.*`.
- Static notebook assets such as `CLIP.png`, `dino.gif`, `unet.png`, and
  `images/` are stored at the assignment root.
- The original pinned requirements were written for an older Python stack, so
  `requirements.txt` uses the same local style as Assignment 2 plus Assignment
  3-specific packages.
