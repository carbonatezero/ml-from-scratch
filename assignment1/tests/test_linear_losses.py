import numpy as np

from src.models.softmax import softmax_loss_naive, softmax_loss_vectorized


def test_softmax_naive_and_vectorized_match():
    rng = np.random.default_rng(0)
    W = 0.001 * rng.standard_normal((5, 4))
    X = rng.standard_normal((6, 5))
    y = np.array([0, 1, 2, 3, 1, 0])

    loss_naive, grad_naive = softmax_loss_naive(W, X, y, reg=0.1)
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X, y, reg=0.1)

    assert np.allclose(loss_naive, loss_vectorized)
    assert np.allclose(grad_naive, grad_vectorized)
