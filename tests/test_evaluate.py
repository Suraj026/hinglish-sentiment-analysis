from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    append_result_row,
    build_classification_report,
    build_confusion_matrix,
    compute_bucket_accuracy,
    compute_metrics,
    load_predictions,
)


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "y_true": [0, 1, 2],
            "y_pred": [0, 2, 2],
            "p_neg": [0.8, 0.1, 0.1],
            "p_neu": [0.1, 0.2, 0.1],
            "p_pos": [0.1, 0.7, 0.8],
        }
    )


def test_evaluation_metrics_and_outputs(predictions, tmp_path: Path):
    source = tmp_path / "predictions.csv"
    predictions.to_csv(source, index=False)
    loaded = load_predictions(source)
    metrics = compute_metrics(loaded)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["macro_f1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert build_classification_report(loaded).shape[0] >= 3
    assert build_confusion_matrix(loaded).shape == (3, 3)


def test_results_append_and_bucket_accuracy(predictions, tmp_path: Path):
    results = tmp_path / "results.csv"
    append_result_row(
        results,
        {
            "model": "lr",
            "embedding": "none",
            "normalized": True,
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "weighted_f1": 0.5,
            "seed": 42,
        },
    )
    buckets = pd.Series(["low", "high", "high"])
    result = compute_bucket_accuracy(predictions, buckets)

    assert list(pd.read_csv(results).columns) == [
        "model",
        "embedding",
        "normalized",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "seed",
    ]
    assert result.set_index("bucket").loc["high", "accuracy"] == 0.5


def test_load_predictions_rejects_malformed_file(tmp_path: Path):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"uid": ["a"]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_predictions(source)
