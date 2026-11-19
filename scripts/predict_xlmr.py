from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def format_probability_row(
    text: str,
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    probabilities = {
        str(item["label"]).lower(): float(item["score"])
        for item in predictions
    }
    scores = {
        label: float(probabilities.get(label, 0.0))
        for label in ("negative", "neutral", "positive")
    }
    sentiment = max(scores, key=scores.get)
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": scores[sentiment],
        **scores,
    }


def predict(texts: list[str], model_dir: Path) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

    id_to_label = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }
    results = []
    for text, scores in zip(texts, probabilities):
        predictions = [
            {"label": id_to_label[index], "score": score}
            for index, score in enumerate(scores)
        ]
        results.append(format_probability_row(text, predictions))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference with the merged XLM-R model."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text:
        raise SystemExit("Provide at least one --text value")
    results = predict(args.text, args.model_dir)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(
                f"{result['text']}\n"
                f"  sentiment={result['sentiment']} "
                f"confidence={result['confidence']:.4f}"
            )


if __name__ == "__main__":
    main()
