"""Neural network architectures for Hinglish sentiment analysis.

Implemented by Person A for Phase 2 neural baselines:
1. LSTMClassifier: Embedding -> LSTM -> Linear
2. BiLSTMClassifier: Embedding -> Bidirectional LSTM -> Linear
3. CNNClassifier: Embedding -> Multi-kernel 1D CNN -> MaxPool -> Linear
4. BiLSTMConvClassifier: Embedding -> BiLSTM -> Conv1D (k=3) -> (AvgPool + MaxPool) -> Linear
   (Based on NITS-Hinglish-SentiMix architecture)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSentimentClassifier(nn.Module):
    """Base class providing common embedding setup and initialization."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False


class LSTMClassifier(BaseSentimentClassifier):
    """Unidirectional LSTM sentiment classification model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, emb_dim)
        output, (hn, _) = self.lstm(embedded)
        # hn[-1]: last layer hidden state -> (batch_size, hidden_dim)
        feat = self.dropout(hn[-1])
        logits = self.classifier(feat)
        return logits


class BiLSTMClassifier(BaseSentimentClassifier):
    """Bidirectional LSTM sentiment classification model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, emb_dim)
        output, (hn, _) = self.lstm(embedded)
        # Concat last forward and backward hidden states
        # hn is (num_layers * 2, batch_size, hidden_dim)
        last_forward = hn[-2]
        last_backward = hn[-1]
        feat = torch.cat([last_forward, last_backward], dim=1)  # (batch_size, hidden_dim * 2)
        logits = self.classifier(self.dropout(feat))
        return logits


class CNNClassifier(BaseSentimentClassifier):
    """1D Convolutional Neural Network for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: tuple[int, ...] = (2, 3, 4),
        num_classes: int = 3,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, emb_dim)
        # Conv1d expects (batch_size, in_channels, seq_len)
        transposed = embedded.transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(transposed))  # (batch_size, num_filters, L_out)
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)  # (batch_size, num_filters)
            pooled_outputs.append(p)
        feat = torch.cat(pooled_outputs, dim=1)
        logits = self.classifier(self.dropout(feat))
        return logits


class BiLSTMConvClassifier(BaseSentimentClassifier):
    """BiLSTM + 1D Convolution sentiment classifier (NITS-Hinglish-SentiMix).

    Sequence flow:
    Embedding -> BiLSTM -> Conv1D (kernel_size=3) -> [GlobalAvgPool, GlobalMaxPool] -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_filters: int = 128,
        kernel_size: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)
        # Concatenation of Global Average Pooling and Global Max Pooling
        self.classifier = nn.Linear(num_filters * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, emb_dim)
        lstm_out, _ = self.bilstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)

        # Transpose for Conv1d: (batch_size, hidden_dim * 2, seq_len)
        lstm_transposed = lstm_out.transpose(1, 2)
        conv_out = F.relu(self.conv(lstm_transposed))  # (batch_size, num_filters, seq_len)

        # Global average pooling and global max pooling over time
        avg_pool = torch.mean(conv_out, dim=2)  # (batch_size, num_filters)
        max_pool, _ = torch.max(conv_out, dim=2)  # (batch_size, num_filters)

        feat = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, num_filters * 2)
        logits = self.classifier(self.dropout(feat))
        return logits


def build_model(
    model_name: str,
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_filters: int = 100,
    num_classes: int = 3,
    dropout: float = 0.3,
    padding_idx: int = 0,
    pretrained_embeddings: torch.Tensor | None = None,
    freeze_embeddings: bool = False,
) -> nn.Module:
    """Factory function for instantiating neural baseline architectures."""
    name = model_name.lower()
    if name == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
    elif name == "bilstm":
        return BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
    elif name == "cnn":
        return CNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
    elif name in {"bilstm_conv", "bilstm+conv", "lstm_conv"}:
        return BiLSTMConvClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_filters=num_filters,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}. "
            "Supported: 'lstm', 'bilstm', 'cnn', 'bilstm_conv'"
        )
