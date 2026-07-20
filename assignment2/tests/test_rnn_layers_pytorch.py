import pytest

torch = pytest.importorskip("torch")

from src.rnn_layers_pytorch import (
    rnn_forward,
    rnn_step_forward,
    temporal_affine_forward,
    temporal_softmax_loss,
    word_embedding_forward,
)


def test_rnn_step_and_sequence_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    h0 = torch.randn(2, 5)
    wx = torch.randn(4, 5)
    wh = torch.randn(5, 5)
    b = torch.randn(5)

    first_step = rnn_step_forward(x[:, 0], h0, wx, wh, b)
    h = rnn_forward(x, h0, wx, wh, b)

    assert first_step.shape == (2, 5)
    assert h.shape == (2, 3, 5)
    assert torch.allclose(h[:, 0], first_step)


def test_word_embedding_temporal_affine_and_loss():
    torch.manual_seed(1)
    word_indices = torch.tensor([[0, 2, 1], [3, 1, 0]])
    embedding = torch.randn(4, 5)
    w = torch.randn(5, 7)
    b = torch.randn(7)
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[True, True, False], [True, False, True]])

    embedded = word_embedding_forward(word_indices, embedding)
    scores = temporal_affine_forward(embedded, w, b)
    loss = temporal_softmax_loss(scores, y, mask)

    assert embedded.shape == (2, 3, 5)
    assert scores.shape == (2, 3, 7)
    assert torch.isfinite(loss)
