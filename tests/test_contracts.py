import pandas as pd
import pytest

from src.contracts import (
    LABEL_TO_ID,
    PREDICTION_COLUMNS,
    RESULT_COLUMNS,
    validate_prediction_frame,
    validate_split_frame,
)


def test_fixed_label_order_and_shared_columns():
    assert LABEL_TO_ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert PREDICTION_COLUMNS == (
        "uid",
        "y_true",
        "y_pred",
        "p_neg",
        "p_neu",
        "p_pos",
    )
    assert RESULT_COLUMNS == (
        "model",
        "embedding",
        "normalized",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "seed",
    )


def test_valid_split_frame_is_accepted():
    frame = pd.DataFrame(
        {"uid": ["a"], "text_clean": ["hello"], "label": [0]}
    )

    validate_split_frame(frame, "train")


def test_split_frame_rejects_invalid_labels():
    frame = pd.DataFrame(
        {"uid": ["a"], "text_clean": ["hello"], "label": [3]}
    )

    with pytest.raises(ValueError, match="invalid labels"):
        validate_split_frame(frame, "train")


def test_prediction_frame_rejects_duplicate_ids():
    frame = pd.DataFrame(
        {
            "uid": ["a", "a"],
            "y_true": [0, 1],
            "y_pred": [0, 1],
            "p_neg": [0.8, 0.1],
            "p_neu": [0.1, 0.8],
            "p_pos": [0.1, 0.1],
        }
    )

    with pytest.raises(ValueError, match="duplicate uid"):
        validate_prediction_frame(frame)


def test_prediction_frame_rejects_missing_probability_columns():
    frame = pd.DataFrame(
        {"uid": ["a"], "y_true": [0], "y_pred": [0]}
    )

    with pytest.raises(ValueError, match="missing columns"):
        validate_prediction_frame(frame)
