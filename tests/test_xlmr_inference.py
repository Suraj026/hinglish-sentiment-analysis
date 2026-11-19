import pandas as pd

from scripts.predict_xlmr import format_probability_row


def test_format_probability_row_maps_transformer_labels():
    result = format_probability_row(
        "good",
        [{"label": "negative", "score": 0.1},
         {"label": "neutral", "score": 0.2},
         {"label": "positive", "score": 0.7}],
    )

    assert result == {
        "text": "good",
        "sentiment": "positive",
        "confidence": 0.7,
        "negative": 0.1,
        "neutral": 0.2,
        "positive": 0.7,
    }
