import pytest
import torch
import torch.nn as nn

from src.dataset import text_to_indices
from src.models import (
    BiLSTMClassifier,
    BiLSTMConvClassifier,
    CNNClassifier,
    LSTMClassifier,
    build_model,
)
from src.train import overfit_test


def test_model_forward_shapes():
    batch_size = 4
    seq_len = 22
    vocab_size = 100
    emb_dim = 32
    hidden_dim = 32

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    models = [
        LSTMClassifier(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_dim=hidden_dim),
        BiLSTMClassifier(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_dim=hidden_dim),
        CNNClassifier(vocab_size=vocab_size, embedding_dim=emb_dim, num_filters=16),
        BiLSTMConvClassifier(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_dim=hidden_dim, num_filters=16),
    ]

    for model in models:
        out = model(x)
        assert out.shape == (batch_size, 3)
        assert not torch.isnan(out).any()


def test_build_model_factory():
    for name in ["lstm", "bilstm", "cnn", "bilstm_conv"]:
        m = build_model(name, vocab_size=50, embedding_dim=16)
        assert isinstance(m, nn.Module)

    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("unsupported_transformer", vocab_size=50)


def test_text_to_indices_padding_and_truncation():
    vocab = {"hello": 2, "world": 3}
    tokens = ["hello", "world", "extra"]

    # Length 5 -> should pad with 0
    padded = text_to_indices(tokens, vocab, max_len=5, pad_idx=0, unk_idx=1)
    assert padded == [2, 3, 1, 0, 0]

    # Length 2 -> should truncate
    truncated = text_to_indices(tokens, vocab, max_len=2, pad_idx=0, unk_idx=1)
    assert truncated == [2, 3]


def test_overfit_test_decreases_loss():
    vocab_size = 50
    model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=16, hidden_dim=16)
    device = torch.device("cpu")
    batch = {
        "input_ids": torch.randint(0, vocab_size, (8, 10)),
        "label": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
    }

    final_loss = overfit_test(model, batch, device, iterations=25, lr=5e-2)
    assert final_loss < 0.5
