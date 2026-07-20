import numpy as np

from src.layers import (
    affine_backward,
    affine_forward,
    batchnorm_backward,
    batchnorm_backward_alt,
    batchnorm_forward,
    conv_forward_naive,
    dropout_backward,
    dropout_forward,
    layernorm_forward,
    max_pool_forward_naive,
    relu_backward,
    relu_forward,
    softmax_loss,
    spatial_batchnorm_forward,
)


def test_affine_relu_and_softmax_shapes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 2, 3))
    w = rng.standard_normal((6, 5))
    b = rng.standard_normal(5)
    y = np.array([0, 1, 2, 3])

    out, affine_cache = affine_forward(x, w, b)
    relu_out, relu_cache = relu_forward(out)
    loss, dout = softmax_loss(relu_out, y)
    drelu = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(drelu, affine_cache)

    assert np.isfinite(loss)
    assert relu_out.shape == (4, 5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape


def test_batchnorm_backward_alt_matches_standard_backward():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 4))
    gamma = rng.standard_normal(4)
    beta = rng.standard_normal(4)
    dout = rng.standard_normal((6, 4))

    _, cache = batchnorm_forward(x, gamma, beta, {"mode": "train"})
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx_alt, dgamma_alt, dbeta_alt = batchnorm_backward_alt(dout, cache)

    assert np.allclose(dx, dx_alt)
    assert np.allclose(dgamma, dgamma_alt)
    assert np.allclose(dbeta, dbeta_alt)


def test_layernorm_normalizes_each_example():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 6))
    gamma = np.ones(6)
    beta = np.zeros(6)

    out, _ = layernorm_forward(x, gamma, beta, {"eps": 1e-8})

    assert np.allclose(out.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(out.var(axis=1), 1, atol=1e-6)


def test_dropout_train_and_test_modes():
    x = np.ones((200, 200))
    train_out, train_cache = dropout_forward(x, {"mode": "train", "p": 0.5, "seed": 3})
    test_out, test_cache = dropout_forward(x, {"mode": "test", "p": 0.5})

    assert np.isclose(train_out.mean(), 1.0, atol=0.05)
    assert np.array_equal(test_out, x)
    assert np.array_equal(dropout_backward(np.ones_like(x), train_cache), train_cache[1])
    assert np.array_equal(dropout_backward(np.ones_like(x), test_cache), np.ones_like(x))


def test_conv_and_pool_forward_known_values():
    x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    w = np.ones((1, 1, 2, 2), dtype=np.float64)
    b = np.array([0.5])

    conv_out, _ = conv_forward_naive(x, w, b, {"stride": 1, "pad": 0})
    pool_out, _ = max_pool_forward_naive(x, {"pool_height": 2, "pool_width": 2, "stride": 2})

    expected_conv = np.array([[[[10.5, 14.5, 18.5], [26.5, 30.5, 34.5], [42.5, 46.5, 50.5]]]])
    expected_pool = np.array([[[[5.0, 7.0], [13.0, 15.0]]]])

    assert np.array_equal(conv_out, expected_conv)
    assert np.array_equal(pool_out, expected_pool)


def test_spatial_batchnorm_preserves_nchw_shape():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 4, 5))
    gamma = np.ones(3)
    beta = np.zeros(3)

    out, _ = spatial_batchnorm_forward(x, gamma, beta, {"mode": "train"})

    assert out.shape == x.shape
