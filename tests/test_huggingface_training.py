from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.huggingface_training import (
    build_dataset,
    build_lora_config,
    compute_metrics,
    merge_and_export,
    parse_args,
    require_cuda,
)


def test_huggingface_training_dependencies_are_declared():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    for package in ("transformers", "datasets", "peft", "accelerate", "safetensors"):
        assert package in requirements


def test_parse_args_uses_xlmr_and_lora_defaults():
    args = parse_args([])
    assert args.model_name == "FacebookAI/xlm-roberta-base"
    assert args.output_dir == Path("results/huggingface/xlm_roberta_lora")
    assert args.merged_dir == Path("results/huggingface/xlm_roberta_merged")
    assert args.max_length == 128
    assert args.batch_size == 16
    assert args.learning_rate == 2e-5
    assert args.epochs == 5
    assert args.dry_run is True


def test_require_cuda_rejects_cpu_only_environment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda()


def test_build_dataset_rejects_invalid_labels(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"uid": ["1"], "text_clean": ["hello"], "label": [3]}).to_csv(
        train, index=False
    )
    val = tmp_path / "val.csv"
    pd.DataFrame({"uid": ["2"], "text_clean": ["good"], "label": [2]}).to_csv(
        val, index=False
    )
    with pytest.raises(ValueError, match="invalid labels"):
        build_dataset(train, val)


def test_compute_metrics_reports_accuracy_and_macro_f1():
    result = compute_metrics(
        SimpleNamespace(
            predictions=np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]]),
            label_ids=np.array([0, 2]),
        )
    )
    assert set(result) >= {"accuracy", "macro_f1", "weighted_f1"}


def test_lora_defaults_target_attention_modules():
    config = build_lora_config()
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) >= {"query", "value", "key", "dense"}


def test_merge_and_export_requires_adapter(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter"):
        merge_and_export(tmp_path / "missing", tmp_path / "merged")


def test_friend_pc_guide_contains_cuda_and_transfer_commands():
    guide = Path("docs/XLMR_TRAINING.md").read_text(encoding="utf-8")
    assert "nvidia-smi" in guide
    assert "python src/huggingface_training.py --train" in guide
    assert "xlmr-model.zip" in guide


def test_project_zip_script_excludes_virtual_environment():
    script = Path("scripts/package_hinglish_training.ps1").read_text(
        encoding="utf-8"
    )
    assert ".venv" not in script
