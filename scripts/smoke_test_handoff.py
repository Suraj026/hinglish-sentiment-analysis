"""Phase 1 Handoff Smoke-Test Script.

Validates the Person B -> Person A handoff:
1. Loads processed train and validation files.
2. Validates schemas, label mappings, and UID uniqueness via src.contracts.
3. Loads preprocessor, FastText embeddings, and vocabulary.
4. Confirms embedding dimensions and recommended sequence length.
5. Generates and validates a smoke-test prediction file following the agreed schema.
6. Evaluates the smoke-test prediction file via src.evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.contracts import (
    LABEL_TO_ID,
    PREDICTION_COLUMNS,
    SPLIT_COLUMNS,
    validate_prediction_frame,
    validate_split_frame,
)
from src.evaluate import compute_metrics, load_predictions
from src.features import _load_fasttext, build_embedding_matrix
from src.preprocess import Preprocessor


def run_smoke_test(repo_root: Path | None = None) -> dict[str, object]:
    root = repo_root or Path(__file__).resolve().parent.parent

    # 1. Paths to handoff artifacts
    data_dir = root / "data" / "processed"
    embeddings_dir = root / "embeddings"
    results_dir = root / "results" / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    preprocessor_path = data_dir / "preprocessor.json"
    vocab_path = embeddings_dir / "vocab.json"
    fasttext_path = embeddings_dir / "fasttext.model"
    h2_manifest_path = data_dir / "H2_MANIFEST.json"

    # 2. Check artifact existence
    for path in [
        train_path,
        val_path,
        preprocessor_path,
        vocab_path,
        fasttext_path,
        h2_manifest_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required handoff artifact: {path}")

    # 3. Load and validate data splits
    train_df = pd.read_csv(train_path, keep_default_na=False)
    val_df = pd.read_csv(val_path, keep_default_na=False)

    validate_split_frame(train_df, "train")
    validate_split_frame(val_df, "validation")

    train_counts = train_df["label"].value_counts().to_dict()
    val_counts = val_df["label"].value_counts().to_dict()

    # 4. Load preprocessor and calculate sequence lengths
    preprocessor = Preprocessor.load(preprocessor_path)
    train_tokens = preprocessor.transform(train_df["text_clean"])
    token_lengths = pd.Series([len(t) for t in train_tokens])
    p95_length = int(token_lengths.quantile(0.95))

    # 5. Load vocabulary and embeddings
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    fasttext_model = _load_fasttext(fasttext_path)
    embedding_dim = fasttext_model.wv.vector_size

    # Verify embedding matrix creation
    emb_matrix = build_embedding_matrix(
        fasttext_path, {"<pad>": 0, "<unk>": 1, **vocab}, embedding_dim=embedding_dim
    )

    # 6. Generate smoke-test prediction file
    # For smoke test, create mock valid probabilities for validation set
    np.random.seed(42)
    raw_logits = np.random.randn(len(val_df), 3)
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    smoke_pred_df = pd.DataFrame(
        {
            "uid": val_df["uid"],
            "y_true": val_df["label"],
            "y_pred": preds,
            "p_neg": probs[:, 0],
            "p_neu": probs[:, 1],
            "p_pos": probs[:, 2],
        }
    )

    validate_prediction_frame(smoke_pred_df)
    smoke_pred_path = results_dir / "smoke_test.csv"
    smoke_pred_df.to_csv(smoke_pred_path, index=False)

    # 7. Evaluate smoke test predictions through Person B's evaluator
    loaded_pred = load_predictions(smoke_pred_path)
    metrics = compute_metrics(loaded_pred)

    report = {
        "status": "PASSED",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_label_distribution": train_counts,
        "val_label_distribution": val_counts,
        "label_mapping": LABEL_TO_ID,
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim,
        "embedding_matrix_shape": list(emb_matrix.shape),
        "recommended_p95_seq_length": p95_length,
        "smoke_test_prediction_file": str(smoke_pred_path),
        "smoke_test_metrics": metrics,
    }

    return report


if __name__ == "__main__":
    report = run_smoke_test()
    print("=" * 60)
    print("PHASE 1 HANDOFF SMOKE TEST: PASSED")
    print("=" * 60)
    print(json.dumps(report, indent=2))
