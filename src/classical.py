from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .contracts import validate_prediction_frame
from .features import build_char_tfidf, build_word_tfidf


def train_classical_model(
    train_texts: Sequence[str],
    y_train: Sequence[int],
    model_name: str,
    feature_name: str,
    seed: int = 42,
) -> tuple[object, object]:
    vectorizer = (
        build_word_tfidf() if feature_name == "word" else build_char_tfidf()
    )
    matrix = vectorizer.fit_transform(train_texts)
    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="saga",
        )
    elif model_name == "linear_svm":
        model = LinearSVC(random_state=seed)
    else:
        raise ValueError(f"unsupported classical model: {model_name}")
    model.fit(matrix, y_train)
    return model, vectorizer


def _probabilities(model: object, matrix: object) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)
    scores = model.decision_function(matrix)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def predict_classical_model(
    model: object, vectorizer: object, texts: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = vectorizer.transform(texts)
    labels = model.predict(matrix)
    probabilities = _probabilities(model, matrix)
    return np.asarray(labels), probabilities


def save_prediction_file(
    path: Path,
    uid: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "uid": list(uid),
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "p_neg": probabilities[:, 0],
            "p_neu": probabilities[:, 1],
            "p_pos": probabilities[:, 2],
        }
    )
    validate_prediction_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def extract_top_ngrams(
    model: object, vectorizer: object, class_id: int, n: int = 20
) -> list[tuple[str, float]]:
    if not hasattr(model, "coef_"):
        raise ValueError("top n-grams require a linear model")
    coefficients = np.asarray(model.coef_)
    row = coefficients[class_id]
    indices = np.argsort(row)[::-1][:n]
    names = vectorizer.get_feature_names_out()
    return [(str(names[index]), float(row[index])) for index in indices]
