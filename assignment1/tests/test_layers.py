import numpy as np

from src.layers import affine_backward, affine_forward, relu_backward, relu_forward


def test_affine_forward_backward_shapes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 2, 3))
    w = rng.standard_normal((6, 5))
    b = rng.standard_normal(5)
    dout = rng.standard_normal((4, 5))

    out, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    assert out.shape == (4, 5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape


def test_relu_forward_backward():
    x = np.array([[-1.0, 0.0, 2.0]])
    dout = np.ones_like(x)

    out, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    assert np.array_equal(out, np.array([[0.0, 0.0, 2.0]]))
    assert np.array_equal(dx, np.array([[0.0, 0.0, 1.0]]))
