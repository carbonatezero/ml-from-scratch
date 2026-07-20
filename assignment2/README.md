# Assignment 2

CS231n-style Assignment 2 adaptation for this local `ml-from-scratch` repo.
The assignment keeps the original educational structure while adapting imports,
paths, datasets, and notebooks to run from this repository.

## Topics

- Batch normalization and layer normalization
- Dropout
- Convolutional neural networks
- PyTorch on CIFAR-10
- Image captioning with vanilla RNNs

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
pip install -r assignment2/requirements.txt
```

The notebooks add the assignment root to `sys.path`, then import reusable code
from `src/`. CIFAR-10 data is expected under `src/datasets/`; the PyTorch
notebook can download CIFAR-10 there if needed.

## Run Checks

Run the lightweight local test suite:

```bash
python -m pytest
```

For a quick syntax-only check:

```bash
PYTHONPYCACHEPREFIX=/tmp/mlfs_a2_pycache python -m compileall -q src tests
```

The tests are intentionally small and dataset-free. They cover the core layer
contracts, normalization/dropout behavior, the fully connected net API, PyTorch
RNN helpers, and notebook JSON/code-cell parseability. They do not replace the
exercise-specific notebook checks or full notebook reruns.

## Notebook Status

The first four notebooks have been rerun and saved successfully:

| Notebook | Topic | Status |
| --- | --- | --- |
| `demos/01_batch_normalization_repo.ipynb` | Batch normalization and layer normalization | Ready |
| `demos/02_dropout_repo.ipynb` | Dropout | Ready |
| `demos/03_convolutional_neural_networks_repo.ipynb` | From-scratch CNNs | Ready |
| `demos/04_pytorch_on_cifar10_repo.ipynb` | PyTorch on CIFAR-10 | Ready |
| `demos/05_image_captioning_vanilla_rnns_repo.ipynb` | Vanilla RNN image captioning | Not ready yet |

The current readiness check for notebooks 01-04 is:

- no missing execution counts
- no empty code cells
- no saved error outputs

Notebook 05 is the next unfinished track. It has unexecuted code cells and an
empty trailing code cell.

## Demos

The notebooks in `demos/` mirror the original Assignment 2 progression:

1. Batch normalization and layer normalization
2. Dropout
3. Convolutional neural networks
4. PyTorch on CIFAR-10
5. Vanilla RNN image captioning

To package the adapted notebooks and source into a zip and combined PDF, run:

```bash
cd demos
bash collect_demos.sh
```

The PDF step requires a working local Jupyter/PDF toolchain. The generated code
bundle zip is ignored because it is a packaging artifact.

## Rerun Commands

Notebooks 01-03 use the default `python3` kernel:

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace \
demos/01_batch_normalization_repo.ipynb \
--ExecutePreprocessor.kernel_name=python3 \
--ExecutePreprocessor.timeout=1200

python3 -m jupyter nbconvert --to notebook --execute --inplace \
demos/02_dropout_repo.ipynb \
--ExecutePreprocessor.kernel_name=python3 \
--ExecutePreprocessor.timeout=1200

python3 -m jupyter nbconvert --to notebook --execute --inplace \
demos/03_convolutional_neural_networks_repo.ipynb \
--ExecutePreprocessor.kernel_name=python3 \
--ExecutePreprocessor.timeout=1800
```

Notebook 04 should be run with the PyTorch-capable `cs231n-arm` kernel:

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace \
demos/04_pytorch_on_cifar10_repo.ipynb \
--ExecutePreprocessor.kernel_name=cs231n-arm \
--ExecutePreprocessor.timeout=3600
```

## Notes

- `demos/03_convolutional_neural_networks_repo.ipynb` builds the optional
  Cython fast-layer extension in `src/` and reloads `src.fast_layers` before the
  fast convolution checks.
- `demos/04_pytorch_on_cifar10_repo.ipynb` includes the CIFAR-10 challenge model;
  the saved rerun completed successfully with the `cs231n-arm` kernel.
- `part5_best_model.pt` stores the saved PyTorch challenge model checkpoint.
