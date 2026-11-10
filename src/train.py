"""Training pipeline and prediction export for neural baseline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.contracts import LABEL_TO_ID, validate_prediction_frame
from src.dataset import HinglishDataset, create_dataloader, load_vocab
from src.evaluate import append_result_row, compute_metrics, load_predictions
from src.models import build_model
from src.preprocess import Preprocessor


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def overfit_test(
    model: nn.Module,
    sample_batch: dict[str, torch.Tensor],
    device: torch.device,
    iterations: int = 40,
    lr: float = 1e-2,
) -> float:
    """Smoke test to confirm model can overfit on a tiny batch before full training."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    x = sample_batch["input_ids"].to(device)
    y = sample_batch["label"].to(device)

    final_loss = 0.0
    for _ in range(iterations):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return final_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * len(y)
    return total_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * len(y)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().tolist())

    val_loss = total_loss / len(loader.dataset)
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted"))
    return val_loss, acc, macro_f1, weighted_f1


def generate_predictions_and_export(
    model: nn.Module,
    val_loader: DataLoader,
    output_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Generate prediction frame matching PREDICTION_COLUMNS contract and save to CSV."""
    model.eval()
    uids: list[object] = []
    y_trues: list[int] = []
    y_preds: list[int] = []
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(device)
            y = batch["label"]
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()

            uids.extend(batch["uid"])
            y_trues.extend(y.tolist())
            y_preds.extend(preds)
            probabilities.append(probs)

    all_probs = np.vstack(probabilities)
    pred_df = pd.DataFrame(
        {
            "uid": uids,
            "y_true": y_trues,
            "y_pred": y_preds,
            "p_neg": all_probs[:, 0],
            "p_neu": all_probs[:, 1],
            "p_pos": all_probs[:, 2],
        }
    )

    validate_prediction_frame(pred_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    return pred_df


def train_baseline_model(
    model_name: str,
    repo_root: Path | None = None,
    embedding_type: str = "random",
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    max_len: int = 22,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 15,
    patience: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    """Train a neural baseline model with early stopping, export predictions and checkpoint."""
    root = repo_root or Path(__file__).resolve().parent.parent
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load artifacts
    preprocessor = Preprocessor.load(root / "data" / "processed" / "preprocessor.json")
    vocab = load_vocab(root / "embeddings" / "vocab.json")
    vocab_size = len(vocab)

    train_csv = root / "data" / "processed" / "train.csv"
    val_csv = root / "data" / "processed" / "val.csv"

    train_loader = create_dataloader(
        csv_path=train_csv,
        preprocessor=preprocessor,
        vocab=vocab,
        max_len=max_len,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        csv_path=val_csv,
        preprocessor=preprocessor,
        vocab=vocab,
        max_len=max_len,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model
    model = build_model(
        model_name=model_name,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
        padding_idx=0,
    ).to(device)

    # Small-batch overfit sanity test
    sample_batch = next(iter(train_loader))
    overfit_loss = overfit_test(
        build_model(model_name, vocab_size, embedding_dim, hidden_dim).to(device),
        sample_batch,
        device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = -1
    best_weights: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    no_improve_epochs = 0

    weights_dir = root / "results" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / f"{model_name}_checkpoint.pt"

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = evaluate_model(
            model, val_loader, criterion, device
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "accuracy": val_acc,
                "macro_f1": val_macro_f1,
                "weighted_f1": val_weighted_f1,
                "val_loss": val_loss,
            }
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    elapsed_time = time.time() - start_time

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save checkpoint with complete metadata
    checkpoint_data = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "max_len": max_len,
        "embedding_type": embedding_type,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "seed": seed,
        "label_mapping": LABEL_TO_ID,
    }
    torch.save(checkpoint_data, checkpoint_path)

    # Export validation predictions
    pred_path = root / "results" / "predictions" / f"{model_name}.csv"
    pred_df = generate_predictions_and_export(model, val_loader, pred_path, device)

    # Re-evaluate exported predictions to confirm Person B evaluator reproducibility
    computed = compute_metrics(load_predictions(pred_path))

    # Append row to results/results.csv
    results_path = root / "results" / "results.csv"
    append_result_row(
        results_path,
        {
            "model": model_name,
            "embedding": embedding_type,
            "normalized": True,
            "accuracy": computed["accuracy"],
            "macro_f1": computed["macro_f1"],
            "weighted_f1": computed["weighted_f1"],
            "seed": seed,
        },
    )

    summary = {
        "model": model_name,
        "embedding": embedding_type,
        "best_epoch": best_epoch,
        "best_macro_f1": computed["macro_f1"],
        "accuracy": computed["accuracy"],
        "weighted_f1": computed["weighted_f1"],
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(pred_path),
        "training_time_sec": round(elapsed_time, 2),
        "overfit_test_loss": round(overfit_loss, 4),
    }
    return summary


def run_all_baselines(repo_root: Path | None = None, seed: int = 42) -> list[dict[str, object]]:
    """Train all Phase 2 baseline models in order: LSTM, BiLSTM, CNN, BiLSTM+Conv."""
    models_to_run = ["lstm", "bilstm", "cnn", "bilstm_conv"]
    summaries = []
    for model_name in models_to_run:
        summary = train_baseline_model(model_name=model_name, repo_root=repo_root, seed=seed)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural baseline models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lstm", "bilstm", "cnn", "bilstm_conv", "all"],
        help="Model architecture to train",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.model == "all":
        results = run_all_baselines(seed=args.seed)
        print("=" * 60)
        print("ALL NEURAL BASELINES COMPLETED:")
        print("=" * 60)
        print(json.dumps(results, indent=2))
    else:
        result = train_baseline_model(model_name=args.model, seed=args.seed)
        print(json.dumps(result, indent=2))
