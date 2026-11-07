from pathlib import Path

import numpy as np

from src.classical import (
    extract_top_ngrams,
    predict_classical_model,
    save_prediction_file,
    train_classical_model,
)
from src.features import build_char_tfidf, build_word_tfidf


TEXTS = ["good happy", "bad sad", "okay neutral", "good nice", "bad awful", "okay fine"]
LABELS = [2, 0, 1, 2, 0, 1]


def test_vectorizers_use_required_ngram_ranges():
    assert build_word_tfidf().ngram_range == (1, 2)
    assert build_char_tfidf().ngram_range == (2, 5)


def test_classical_model_predicts_labels_and_probabilities():
    model, vectorizer = train_classical_model(
        TEXTS, LABELS, "logistic_regression", "word"
    )

    labels, probabilities = predict_classical_model(
        model, vectorizer, ["good happy", "bad sad"]
    )

    assert labels.tolist() == [2, 0]
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_linear_svm_uses_probability_adapter_and_top_features(tmp_path: Path):
    model, vectorizer = train_classical_model(
        TEXTS, LABELS, "linear_svm", "word"
    )
    labels, probabilities = predict_classical_model(
        model, vectorizer, ["okay fine"]
    )
    output = tmp_path / "predictions.csv"
    save_prediction_file(
        output, ["id-1"], [1], labels, probabilities
    )

    assert output.exists()
    assert extract_top_ngrams(model, vectorizer, 0, n=2)
