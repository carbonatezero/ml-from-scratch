# Assignment 2 Demos

These notebooks adapt the CS231n Assignment 2 materials from
`../_resources/cs231n-2026spring/assignment2` for this local repo. Treat each
notebook as a guided spec for code in `../src/`: read one section, implement
the referenced function, run the matching check cell, then move on.

## Notebooks

### 01_batch_normalization_repo

Covers:

- Batch normalization
- Layer normalization
- Fully connected networks with normalization

Notes: Locally adapted and runnable from `assignment2/` or
`assignment2/demos/`.

### 02_dropout_repo

Covers:

- Inverted dropout forward and backward passes
- Fully connected networks with dropout
- Dropout as regularization

Notes: Locally adapted and verified through setup, imports, and CIFAR-10
loading.

### 03_convolutional_neural_networks_repo

Covers:

- Naive convolution forward and backward passes
- Max-pooling forward and backward passes
- Fast layers and sandwich layers
- Three-layer convolutional network
- Spatial batch normalization and spatial group normalization

Notes: This is the main from-scratch CNN notebook for Assignment 2. Work
through it section by section; do not run it top to bottom until the referenced
`../src/` functions have been implemented.

### 04_pytorch_on_cifar10_repo

Covers:

- PyTorch tensors and training loops
- Module API
- Sequential API
- CIFAR-10 challenge model

Notes: This notebook is the framework-based counterpart to the from-scratch
work. It is structurally the longest notebook in this demo set.

### 05_image_captioning_vanilla_rnns_repo

Covers:

- Vanilla RNN layers
- Word embeddings
- Temporal affine and softmax layers
- RNN image captioning model
- Small-data overfit check
- Test-time caption sampling

Notes: This is a separate sequence-modeling track scoped to vanilla RNN
captioning. It depends on the RNN and COCO helper source files in `../src/`, and
expects the compact CS231n COCO captioning data under
`../src/datasets/coco_captioning`. Image URL display errors are harmless because
training and sampling use precomputed features on disk.

## Suggested Workflow

1. Start from `assignment2/` or `assignment2/demos/`.
2. Run the first local setup cell.
3. Read one exercise section.
4. Implement the matching function in `../src/`.
5. Run only that section's check cell.
6. Commit after each stable chunk.

Recommended order:

```text
01 Batch normalization
02 Dropout
03 Convolutional neural networks
04 PyTorch on CIFAR-10
05 Image captioning with vanilla RNNs
```

Recommended order inside `03_convolutional_neural_networks_repo`:

```text
conv_forward_naive
conv_backward_naive
max_pool_forward_naive
max_pool_backward_naive
fast layers
sandwich layers
ThreeLayerConvNet
spatial_batchnorm_forward / spatial_batchnorm_backward
spatial_groupnorm_forward / spatial_groupnorm_backward
```
