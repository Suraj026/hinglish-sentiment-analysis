from pathlib import Path

import numpy as np

from src.features import (
    build_embedding_matrix,
    export_vocab,
    nearest_neighbors,
    train_fasttext,
)


def test_fasttext_artifact_vocab_neighbors_and_matrix(tmp_path: Path):
    model_path = tmp_path / "fasttext.model"
    vocab_path = tmp_path / "vocab.json"
    corpus = [
        ["good", "happy"],
        ["good", "nice"],
        ["bad", "sad"],
        ["bad", "awful"],
    ]

    result = train_fasttext(corpus, model_path, seed=42)
    export_vocab(result, vocab_path)
    neighbors = nearest_neighbors(result, "good", topn=2)
    matrix = build_embedding_matrix(
        result, {"<pad>": 0, "good": 1, "missing": 2}, embedding_dim=16
    )

    assert result == model_path
    assert vocab_path.exists()
    assert len(neighbors) == 2
    assert matrix.shape == (3, 16)
    assert np.all(matrix[0] == 0)
