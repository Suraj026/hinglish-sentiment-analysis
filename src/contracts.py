from __future__ import annotations

from typing import Final

import pandas as pd


LABEL_TO_ID: Final[dict[str, int]] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

SPLIT_COLUMNS: Final[tuple[str, ...]] = ("uid", "text_clean", "label")
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "uid",
    "y_true",
    "y_pred",
    "p_neg",
    "p_neu",
    "p_pos",
)
RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "embedding",
    "normalized",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "seed",
)


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], name: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(missing)}")


def validate_split_frame(frame: pd.DataFrame, name: str) -> None:
    """Validate an H1/H2 split frame without modifying it."""
    _require_columns(frame, SPLIT_COLUMNS, name)
    if frame["uid"].isna().any() or frame["uid"].duplicated().any():
        raise ValueError(f"{name} contains missing or duplicate uid values")
    if frame["text_clean"].isna().any():
        raise ValueError(f"{name} contains missing text_clean values")
    labels = set(frame["label"].dropna().tolist())
    invalid = labels.difference(LABEL_TO_ID.values())
    if frame["label"].isna().any() or invalid:
        raise ValueError(f"{name} contains invalid labels: {sorted(invalid)}")


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Validate a prediction file against the shared handoff contract."""
    _require_columns(frame, PREDICTION_COLUMNS, "prediction frame")
    if frame["uid"].isna().any() or frame["uid"].duplicated().any():
        raise ValueError("prediction frame contains duplicate uid values")
    for column in ("y_true", "y_pred"):
        invalid = set(frame[column].dropna().tolist()).difference(
            LABEL_TO_ID.values()
        )
        if frame[column].isna().any() or invalid:
            raise ValueError(
                f"prediction frame contains invalid labels in {column}: "
                f"{sorted(invalid)}"
            )
    probabilities = frame[["p_neg", "p_neu", "p_pos"]]
    if probabilities.isna().any().any():
        raise ValueError("prediction frame contains missing probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError("prediction frame contains probabilities outside [0, 1]")
