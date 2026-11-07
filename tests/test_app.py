from pathlib import Path

import numpy as np
import pytest

from app import EXAMPLES, build_demo, load_artifacts, predict_sentiment


class FakeModel:
    def predict(self, values):
        return np.array([[0.1, 0.2, 0.7]])


def test_demo_has_four_examples():
    assert len(EXAMPLES) == 4
    assert all(len(example) == 1 for example in EXAMPLES)


def test_predict_sentiment_reuses_preprocessor():
    class FakePreprocessor:
        def transform(self, texts):
            return [["token"]]

    result = predict_sentiment("hello", (FakeModel(), FakePreprocessor()))

    assert result == {
        "negative": pytest.approx(0.1),
        "neutral": pytest.approx(0.2),
        "positive": pytest.approx(0.7),
    }


def test_load_artifacts_reports_missing_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="model"):
        load_artifacts(tmp_path / "missing", tmp_path / "preprocessor.json")


def test_build_demo_returns_gradio_block():
    demo = build_demo()
    assert demo is not None
