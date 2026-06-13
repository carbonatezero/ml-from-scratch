# Math Behind `toy_nn.py`

The code is a tiny fully connected neural network:

```text
input -> hidden layer -> ReLU -> score
```

## Input

The input is a vector with three numbers:

```text
x = [2, 1, 3]
```

You can think of these as three simple features.

## Dot Product

Each neuron computes a dot product:

```text
dot(x, w) = x[0]*w[0] + x[1]*w[1] + x[2]*w[2]
```

Then it adds a bias:

```text
neuron output = dot(x, w) + b
```

That is what this code does:

```python
h.append(dot(x, w1[neuron]) + b1[neuron])
```

## Hidden Layer

The hidden layer has two neurons.

The first hidden neuron uses:

```text
w1[0] = [0.5, -1, 0.2]
b1[0] = 0.1
```

Its value is:

```text
h[0] = 2*0.5 + 1*(-1) + 3*0.2 + 0.1
     = 1.0 - 1.0 + 0.6 + 0.1
     = 0.7
```

The second hidden neuron uses:

```text
w1[1] = [1, 0.1, -0.5]
b1[1] = -0.2
```

Its value is:

```text
h[1] = 2*1 + 1*0.1 + 3*(-0.5) - 0.2
     = 2.0 + 0.1 - 1.5 - 0.2
     = 0.4
```

So the hidden layer output is:

```text
h = [0.7, 0.4]
```

## ReLU

ReLU means "keep positive values, replace negative values with zero."

```text
ReLU(v) = max(0, v)
```

Both hidden values are positive, so ReLU does not change them:

```text
r = [0.7, 0.4]
```

## Final Score

The final layer is another dot product plus a bias:

```text
score = dot(r, w2) + b2
```

In this code:

```text
w2 = [1.5, -1]
b2 = 0.3
```

So:

```text
score = 0.7*1.5 + 0.4*(-1) + 0.3
      = 1.05 - 0.4 + 0.3
      = 0.95
```

This score is not a probability. It is just a raw class score from the tiny
neural network.

