from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .contracts import PREDICTION_COLUMNS, RESULT_COLUMNS, validate_prediction_frame


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_prediction_frame(frame)
    return frame


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    validate_prediction_frame(predictions)
    return {
        "accuracy": float(accuracy_score(predictions.y_true, predictions.y_pred)),
        "macro_f1": float(
            f1_score(predictions.y_true, predictions.y_pred, average="macro")
        ),
        "weighted_f1": float(
            f1_score(predictions.y_true, predictions.y_pred, average="weighted")
        ),
    }


def build_classification_report(predictions: pd.DataFrame) -> pd.DataFrame:
    validate_prediction_frame(predictions)
    report = classification_report(
        predictions.y_true,
        predictions.y_pred,
        labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def build_confusion_matrix(predictions: pd.DataFrame) -> np.ndarray:
    validate_prediction_frame(predictions)
    return confusion_matrix(predictions.y_true, predictions.y_pred, labels=[0, 1, 2])


def append_result_row(path: Path, row: Mapping[str, object]) -> None:
    missing = [column for column in RESULT_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"result row missing columns: {', '.join(missing)}")
    frame = pd.DataFrame([{column: row[column] for column in RESULT_COLUMNS}])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != list(RESULT_COLUMNS):
            raise ValueError("results file has invalid columns")
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def compute_bucket_accuracy(
    predictions: pd.DataFrame, buckets: Sequence[str]
) -> pd.DataFrame:
    validate_prediction_frame(predictions)
    if len(predictions) != len(buckets):
        raise ValueError("bucket count must match prediction count")
    frame = predictions.copy()
    frame["bucket"] = list(buckets)
    rows = []
    for bucket, group in frame.groupby("bucket", sort=True):
        rows.append(
            {
                "bucket": bucket,
                "count": len(group),
                "accuracy": float(accuracy_score(group.y_true, group.y_pred)),
            }
        )
    return pd.DataFrame(rows, columns=["bucket", "count", "accuracy"])


def compare_runs(prediction_paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in prediction_paths:
        metrics = compute_metrics(load_predictions(path))
        rows.append({"path": str(path), **metrics})
    return pd.DataFrame(rows, columns=["path", "accuracy", "macro_f1", "weighted_f1"])
