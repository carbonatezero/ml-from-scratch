# Assignment 2

Planned CS231n-style Assignment 2 adaptation.

Expected topics:

- Batch normalization and layer normalization
- Dropout
- Convolutional neural networks
- PyTorch on CIFAR-10
- Image captioning with vanilla RNNs

Keep the Python source close to the original assignment files. Prefer local
path/package adaptations, notebook cleanup, tests, and PDF snapshots over broad
source refactors.

## Q1: Batch Normalization

The adapted Q1 notebook is:

- `demos/01_batch_normalization_repo.ipynb`

Run it locally from `assignment2/` or `assignment2/demos/`. The first notebook
cell adds `src/` to `sys.path` and downloads CIFAR-10 into
`src/datasets/` if needed.

```bash
cd assignment2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook demos/01_batch_normalization_repo.ipynb
```
