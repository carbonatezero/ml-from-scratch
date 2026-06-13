# Assignment 0 - Tiny Fully Connected Neural Network

This assignment is a small, fully runnable introduction to a fully connected
neural network.

It uses plain Python lists instead of NumPy so the moving parts are easy to
write from scratch.

`toy_nn.py` shows:

- an input vector
- a hidden layer with two neurons
- ReLU
- one final linear class score

Run:

```bash
python3 toy_nn.py
```

See `MATH.md` for the underlying math.

`toy_nn_train.py` shows the same network with a tiny training loop:

- forward pass
- squared error loss
- backpropagation
- gradient descent updates

Run:

```bash
python3 toy_nn_train.py
```
