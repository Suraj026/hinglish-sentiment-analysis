"""Dataset and DataLoader utilities for Hinglish sentiment neural models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.contracts import validate_split_frame
from src.preprocess import Preprocessor


def text_to_indices(
    tokens: Sequence[str],
    vocab: dict[str, int],
    max_len: int = 22,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> list[int]:
    """Convert a sequence of tokens into padded/truncated integer indices."""
    indices = [vocab.get(token, unk_idx) for token in tokens[:max_len]]
    if len(indices) < max_len:
        indices.extend([pad_idx] * (max_len - len(indices)))
    return indices


class HinglishDataset(Dataset):
    """PyTorch Dataset for Hinglish sentiment classification."""

    def __init__(
        self,
        uids: Sequence[object],
        texts: Sequence[str],
        labels: Sequence[int] | None,
        preprocessor: Preprocessor,
        vocab: dict[str, int],
        max_len: int = 22,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ) -> None:
        self.uids = list(uids)
        self.labels = list(labels) if labels is not None else None
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        # Tokenize all texts with preprocessor
        tokenized_texts = preprocessor.transform(texts)
        self.encoded_sequences = [
            text_to_indices(
                tokens=toks,
                vocab=vocab,
                max_len=max_len,
                pad_idx=pad_idx,
                unk_idx=unk_idx,
            )
            for toks in tokenized_texts
        ]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | object]:
        item: dict[str, torch.Tensor | object] = {
            "uid": self.uids[idx],
            "input_ids": torch.tensor(
                self.encoded_sequences[idx], dtype=torch.long
            ),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_vocab(vocab_path: Path) -> dict[str, int]:
    """Load vocabulary JSON mapping and ensure pad and unk tokens exist."""
    with vocab_path.open("r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    # Reserve 0 for <pad> and 1 for <unk>
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    next_idx = 2
    for token in raw_vocab.keys():
        if token not in vocab:
            vocab[token] = next_idx
            next_idx += 1
    return vocab


def create_dataloader(
    csv_path: Path,
    preprocessor: Preprocessor,
    vocab: dict[str, int],
    max_len: int = 22,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from a CSV split frame."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    if "label" in df.columns:
        validate_split_frame(df, str(csv_path.name))
        labels = df["label"].tolist()
    else:
        labels = None

    dataset = HinglishDataset(
        uids=df["uid"].tolist(),
        texts=df["text_clean"].tolist(),
        labels=labels,
        preprocessor=preprocessor,
        vocab=vocab,
        max_len=max_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
