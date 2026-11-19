from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer


def build_word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), analyzer="word")


def build_char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(2, 5), analyzer="char")


def train_fasttext(
    train_token_sequences: Iterable[Sequence[str]],
    output_path: Path,
    seed: int = 42,
) -> Path:
    sentences = [list(tokens) for tokens in train_token_sequences]
    model = FastText(
        vector_size=16,
        window=3,
        min_count=1,
        workers=1,
        sg=1,
        seed=seed,
        epochs=20,
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=model.epochs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return output_path


def _load_fasttext(model_path: Path) -> FastText:
    return FastText.load(str(model_path))


def export_vocab(model_path: Path, output_path: Path) -> None:
    model = _load_fasttext(model_path)
    output_path.write_text(
        json.dumps(
            {token: index for index, token in enumerate(model.wv.index_to_key)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def nearest_neighbors(
    model_path: Path, token: str, topn: int = 10
) -> list[tuple[str, float]]:
    return [
        (str(name), float(score))
        for name, score in _load_fasttext(model_path).wv.most_similar(
            token, topn=topn
        )
    ]


def build_embedding_matrix(
    model_path: Path, token_to_id: Mapping[str, int], embedding_dim: int
) -> np.ndarray:
    model = _load_fasttext(model_path)
    matrix = np.zeros((len(token_to_id), embedding_dim), dtype=np.float32)
    for token, index in token_to_id.items():
        if token == "<pad>":
            continue
        if token in model.wv:
            vector = model.wv[token]
            copy_dim = min(len(vector), embedding_dim)
            matrix[index, :copy_dim] = vector[:copy_dim]
    return matrix
