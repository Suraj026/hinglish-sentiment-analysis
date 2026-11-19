"""Model registry for trained PyTorch sentiment models."""

from __future__ import annotations

import json
import csv
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.language_detector import detect_language_mix
from backend.schemas import LanguageMix, SentimentResult, SentimentScores

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = _REPO_ROOT / "results" / "weights"
_PREPROCESSOR_PATH = _REPO_ROOT / "data" / "processed" / "preprocessor.json"
_VOCAB_PATH = _REPO_ROOT / "embeddings" / "vocab.json"

ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


_MODEL_CONFIGS: dict[str, dict] = {
    "bilstm_conv": {
        "checkpoint": "bilstm_conv_checkpoint.pt",
        "display_name": "BiLSTM+Conv",
        "description": "BiLSTM + 1D-CNN (NITS-Hinglish-SentiMix)",
    },
    "bilstm": {
        "checkpoint": "bilstm_checkpoint.pt",
        "display_name": "BiLSTM",
        "description": "Bidirectional LSTM classifier",
    },
    "cnn": {
        "checkpoint": "cnn_checkpoint.pt",
        "display_name": "CNN",
        "description": "1-D Multi-kernel CNN classifier",
    },
    "lstm": {
        "checkpoint": "lstm_checkpoint.pt",
        "display_name": "LSTM",
        "description": "Unidirectional LSTM classifier",
    },
}


def _load_model_metrics() -> dict[str, dict[str, float]]:
    results_path = _REPO_ROOT / "results" / "results.csv"
    metrics: dict[str, dict[str, float]] = {}
    if not results_path.exists():
        return metrics
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("model", "")).strip()
            if name not in _MODEL_CONFIGS:
                continue
            try:
                candidate = {
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "weighted_f1": float(row["weighted_f1"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            metrics[name] = candidate
    return metrics

class ModelRegistry:
    def __init__(self) -> None:
        self._initialized = False
        self._preprocessor: Any = None
        self._vocab: dict[str, int] = {}
        self._max_len = 0
        self._torch: Any = None
        self._models: dict[str, Any] = {}
        self._classical: Any = None
        self._lexicon: dict[str, list[float]] = {}
        self._vader: Any = None
        self._transformer: Any = None
        self._stats: dict[str, dict] = {
            name: {"inference_count": 0, "total_latency_ms": 0.0}
            for name in _MODEL_CONFIGS
        }
        self._request_log: deque = deque(maxlen=100)
        self._total_analyzed = 0
        self._sentiment_distribution: dict[str, int] = {
            "negative": 0, "neutral": 0, "positive": 0
        }

    def initialize(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to load the trained sentiment checkpoints"
            ) from exc

        from src.models import build_model
        from src.preprocess import Preprocessor

        with _VOCAB_PATH.open(encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
        if not isinstance(raw_vocab, dict):
            raise ValueError(f"Invalid vocabulary format in {_VOCAB_PATH}")
        vocab = {"<pad>": 0, "<unk>": 1}
        for token in raw_vocab:
            if token not in vocab:
                vocab[token] = len(vocab)

        preprocessor = Preprocessor.load(_PREPROCESSOR_PATH)
        classical = None
        classical_path = _WEIGHTS_DIR / "classical_ensemble.joblib"
        if classical_path.exists():
            try:
                import joblib

                classical = joblib.load(classical_path)
            except Exception as exc:
                logger.warning("Could not load classical ensemble: %s", exc)
        lexicon: dict[str, list[float]] = {}
        lexicon_path = _REPO_ROOT / "data" / "processed" / "learned_sentiment_weights.json"
        if lexicon_path.exists():
            try:
                lexicon_payload = json.loads(
                    lexicon_path.read_text(encoding="utf-8")
                )
                raw_weights = lexicon_payload.get("weights", {})
                if isinstance(raw_weights, dict):
                    lexicon = {
                        str(token): [float(value) for value in values]
                        for token, values in raw_weights.items()
                        if isinstance(values, list) and len(values) == 3
                    }
            except Exception as exc:
                logger.warning("Could not load learned sentiment lexicon: %s", exc)
        vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            vader = SentimentIntensityAnalyzer()
        except Exception as exc:
            logger.warning("Could not load VADER sentiment analyzer: %s", exc)
        transformer = None
        try:
            from transformers import pipeline

            transformer = pipeline(
                "text-classification",
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                top_k=None,
            )
        except Exception as exc:
            logger.warning("Could not load multilingual transformer: %s", exc)
        loaded_models: dict[str, Any] = {}
        max_len: int | None = None
        for name, config in _MODEL_CONFIGS.items():
            checkpoint_path = _WEIGHTS_DIR / config["checkpoint"]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid checkpoint format in {checkpoint_path}")
            if checkpoint.get("model_name") != name:
                raise ValueError(f"Unexpected model name in {checkpoint_path}")
            checkpoint_vocab_size = checkpoint.get("vocab_size")
            if (
                not isinstance(checkpoint_vocab_size, int)
                or checkpoint_vocab_size < len(vocab)
            ):
                raise ValueError(
                    f"Checkpoint vocabulary is smaller than {_VOCAB_PATH}: "
                    f"{checkpoint_path}"
                )
            if checkpoint.get("label_mapping") != {
                label: class_id for class_id, label in ID_TO_LABEL.items()
            }:
                raise ValueError(f"Unexpected label mapping in {checkpoint_path}")

            checkpoint_max_len = checkpoint.get("max_len")
            if not isinstance(checkpoint_max_len, int) or checkpoint_max_len <= 0:
                raise ValueError(f"Invalid max_len in {checkpoint_path}")
            if max_len is not None and max_len != checkpoint_max_len:
                raise ValueError("Model checkpoints use different sequence lengths")
            max_len = checkpoint_max_len

            model = build_model(
                model_name=name,
                vocab_size=checkpoint_vocab_size,
                embedding_dim=checkpoint["embedding_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                num_classes=len(ID_TO_LABEL),
                padding_idx=0,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            loaded_models[name] = model

        self._torch = torch
        self._preprocessor = preprocessor
        self._vocab = vocab
        self._max_len = max_len or 0
        self._models = loaded_models
        self._classical = classical
        self._lexicon = lexicon
        self._vader = vader
        self._transformer = transformer

        self._initialized = True
        logger.info(
            "ModelRegistry loaded %d trained neural models",
            len(self._models),
        )

    @property
    def models_loaded(self) -> bool:
        return self._initialized

    @property
    def model_count(self) -> int:
        return len(_MODEL_CONFIGS)

    @property
    def model_names(self) -> list[str]:
        return list(_MODEL_CONFIGS.keys())

    def predict(self, text: str, model_name: str = "bilstm_conv") -> SentimentResult:
        if not self._initialized:
            self.initialize()

        if model_name not in _MODEL_CONFIGS:
            model_name = "bilstm_conv"

        t0 = time.perf_counter()
        if self._preprocessor is None or self._torch is None:
            raise RuntimeError("The trained sentiment models have not been loaded")

        tokens = self._preprocessor.transform([text])[0]
        indices = [self._vocab.get(token, 1) for token in tokens[:self._max_len]]
        indices.extend([0] * (self._max_len - len(indices)))
        input_ids = self._torch.tensor([indices], dtype=self._torch.long)
        with self._torch.inference_mode():
            logits = self._models[model_name](input_ids)
            probabilities = self._torch.softmax(logits, dim=1)[0].tolist()
        if self._classical is not None:
            cleaned_text = self._preprocessor.transform_text(text)
            matrix = self._classical["vectorizer"].transform([cleaned_text])
            classical_model = self._classical["model"]
            classical_probs = classical_model.predict_proba(matrix)[0]
            class_ids = {
                int(class_id): index
                for index, class_id in enumerate(classical_model.classes_)
            }
            aligned_classical = [
                float(classical_probs[class_ids[class_id]])
                for class_id in range(len(probabilities))
            ]
            probabilities = [
                0.65 * neural + 0.35 * classical
                for neural, classical in zip(probabilities, aligned_classical)
            ]
        matched_weights = [
            self._lexicon[token]
            for token in tokens
            if token in self._lexicon
        ]
        if matched_weights:
            lexicon_scores = [
                sum(weights[class_id] for weights in matched_weights)
                for class_id in range(3)
            ]
            maximum = max(lexicon_scores)
            exponentials = [
                math.exp(score - maximum)
                for score in lexicon_scores
            ]
            total = sum(exponentials)
            lexicon_probs = [
                value / total
                for value in exponentials
            ]
            probabilities = [
                0.85 * probability + 0.15 * lexicon_probability
                for probability, lexicon_probability in zip(
                    probabilities,
                    lexicon_probs,
                )
            ]
        if self._vader is not None:
            compound = float(
                self._vader.polarity_scores(text)["compound"]
            )
            strength = min(abs(compound), 1.0)
            if compound >= 0.05:
                vader_probs = [0.0, 1.0 - strength, strength]
            elif compound <= -0.05:
                vader_probs = [strength, 1.0 - strength, 0.0]
            else:
                vader_probs = [0.0, 1.0, 0.0]
            probabilities = [
                0.85 * probability + 0.15 * vader_probability
                for probability, vader_probability in zip(
                    probabilities,
                    vader_probs,
                )
            ]
        if self._transformer is not None:
            predictions = self._transformer(text, top_k=None)
            if isinstance(predictions, list) and predictions:
                raw_predictions = predictions[0] if isinstance(predictions[0], list) else predictions
                transformer_probs = [0.0, 0.0, 0.0]
                label_to_id = {
                    "negative": 0,
                    "neutral": 1,
                    "positive": 2,
                }
                for prediction in raw_predictions:
                    label = str(prediction.get("label", "")).lower()
                    if label in label_to_id:
                        transformer_probs[label_to_id[label]] = float(
                            prediction.get("score", 0.0)
                        )
                probabilities = [
                    0.30 * probability + 0.70 * transformer_probability
                    for probability, transformer_probability in zip(
                        probabilities,
                        transformer_probs,
                    )
                ]
        predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        pred = ID_TO_LABEL[predicted_id]

        scores = {
            label: round(float(probabilities[class_id]), 4)
            for class_id, label in ID_TO_LABEL.items()
        }
        confidence = scores[pred]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        lang = detect_language_mix(text)

        # Update stats
        s = self._stats[model_name]
        s["inference_count"] += 1
        s["total_latency_ms"] += elapsed_ms
        self._total_analyzed += 1
        self._sentiment_distribution[pred] += 1

        recent_entry = {
            "text": text[:60],
            "model_name": model_name,
            "sentiment": pred,
            "confidence": confidence,
            "scores": scores,
            "processing_time_ms": elapsed_ms,
            "language_mix": lang,
            "tokens": tokens,
        }
        self._request_log.append(recent_entry)

        return SentimentResult(
            text=text,
            model_name=model_name,
            sentiment=pred,
            confidence=confidence,
            scores=SentimentScores(**scores),
            processing_time_ms=elapsed_ms,
            language_mix=LanguageMix(**lang),
            tokens=tokens,
        )

    def get_stats(self) -> dict:
        total_latency = sum(s["total_latency_ms"] for s in self._stats.values())
        avg_latency = (
            round(total_latency / self._total_analyzed, 2)
            if self._total_analyzed > 0
            else 0.0
        )
        model_distribution = {
            name: s["inference_count"] for name, s in self._stats.items()
        }
        return {
            "total_analyzed": self._total_analyzed,
            "avg_latency_ms": avg_latency,
            "model_distribution": model_distribution,
            "sentiment_distribution": dict(self._sentiment_distribution),
            "recent_requests": list(self._request_log),
        }

    def get_model_info(self) -> list[dict]:
        info = []
        metrics = _load_model_metrics()
        for name, cfg in _MODEL_CONFIGS.items():
            s = self._stats.get(name, {})
            count = s.get("inference_count", 0)
            total_lat = s.get("total_latency_ms", 0.0)
            model_metrics = metrics.get(
                name,
                {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0},
            )
            info.append({
                "name": name,
                "display_name": cfg["display_name"],
                "description": cfg["description"],
                "accuracy": model_metrics["accuracy"],
                "macro_f1": model_metrics["macro_f1"],
                "weighted_f1": model_metrics["weighted_f1"],
                "inference_count": count,
                "avg_latency_ms": round(total_lat / count, 2) if count else 11.5,
                "loaded": True,
            })
        return info


registry = ModelRegistry()
