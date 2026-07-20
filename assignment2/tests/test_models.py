import numpy as np

from src.classifiers.fc_net import FullyConnectedNet


def test_fully_connected_net_forward_backward_contract():
    rng = np.random.default_rng(5)
    model = FullyConnectedNet(
        [6, 7],
        input_dim=4,
        num_classes=3,
        normalization="batchnorm",
        dropout_keep_ratio=0.8,
        reg=0.1,
        weight_scale=1e-2,
        dtype=np.float64,
        seed=123,
    )
    x = rng.standard_normal((5, 4))
    y = np.array([0, 1, 2, 2, 1])

    scores = model.loss(x)
    loss, grads = model.loss(x, y)

    assert scores.shape == (5, 3)
    assert np.isfinite(loss)
    assert set(grads) == set(model.params)
    for name, value in model.params.items():
        assert grads[name].shape == value.shape
