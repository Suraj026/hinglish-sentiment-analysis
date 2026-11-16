from __future__ import annotations

import pickle
from pathlib import Path

import gradio as gr
import numpy as np

from src.preprocess import Preprocessor


EXAMPLES = [
    ["Yeh movie bahut acchi hai 😊"],
    ["Mujhe bilkul pasand nahi aaya"],
    ["Aaj ka din theek tha"],
    ["Soooo happy with this result!"],
]
LABELS = ("negative", "neutral", "positive")


def load_artifacts(
    model_path: Path, preprocessor_path: Path
) -> tuple[object, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"preprocessor artifact not found: {preprocessor_path}"
        )
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    return model, Preprocessor.load(preprocessor_path)


def predict_sentiment(text: str, artifacts: tuple[object, object]) -> dict[str, float]:
    model, preprocessor = artifacts
    transformed = preprocessor.transform([text])
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(transformed))[0]
    else:
        probabilities = np.asarray(model.predict(transformed))[0]
        if probabilities.ndim == 0:
            values = np.zeros(3, dtype=float)
            values[int(probabilities)] = 1.0
            probabilities = values
    if len(probabilities) != 3:
        raise ValueError("model must return three class probabilities")
    return {label: float(probability) for label, probability in zip(LABELS, probabilities)}


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Hinglish Sentiment Analysis")
        text = gr.Textbox(label="Tweet")
        output = gr.Label(label="Sentiment probabilities")
        button = gr.Button("Predict")
        button.click(
            lambda value: predict_sentiment(
                value,
                load_artifacts(
                    Path("results/weights/best_model.pkl"),
                    Path("data/processed/preprocessor.json"),
                ),
            ),
            inputs=text,
            outputs=output,
        )
        gr.Examples(examples=EXAMPLES, inputs=text)
    return demo


if __name__ == "__main__":
    build_demo().launch()
