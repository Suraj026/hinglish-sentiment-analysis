from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gradio as gr
import joblib

from src.preprocess import Preprocessor


EXAMPLES: list[list[str]] = [
    ["This product is bahut accha"],
    ["Very disappointing experience"],
    ["The order arrived on time"],
    ["Mujhe samajh nahi aaya"],
]


def load_artifacts(model_path: Path, preprocessor_path: Path) -> tuple[object, Preprocessor]:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessor artifact: {preprocessor_path}"
        )
    model = joblib.load(model_path)
    preprocessor = Preprocessor.load(preprocessor_path)
    return model, preprocessor


def predict_sentiment(
    text: str,
    artifacts: tuple[object, Preprocessor],
) -> dict[str, float]:
    model, preprocessor = artifacts
    if hasattr(preprocessor, "transform_text"):
        cleaned = [preprocessor.transform_text(text)]
    else:
        cleaned = [" ".join(preprocessor.transform([text])[0])]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(cleaned)[0]
    else:
        probabilities = model.predict(cleaned)[0]
    return {
        "negative": float(probabilities[0]),
        "neutral": float(probabilities[1]),
        "positive": float(probabilities[2]),
    }


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("Hinglish sentiment analysis")
    return demo
