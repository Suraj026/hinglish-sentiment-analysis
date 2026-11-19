# XLM-RoBERTa LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible GPU-only training entry point that fine-tunes `FacebookAI/xlm-roberta-base` with LoRA and exports a standalone merged model plus evaluation artifacts.

**Architecture:** Add one Hugging Face dataset/training module that reads the existing processed train and validation CSV files, applies XLM-R preprocessing and stratified class weighting, trains with a Hugging Face `Trainer`, and exports both the LoRA adapter and a merged `safetensors` model. Add a smoke-test script that verifies CUDA, dataset schemas, model loading, inference, and export paths without starting training.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Datasets, PEFT, Accelerate, scikit-learn, safetensors.

**Spec:** User request: implement XLM-RoBERTa LoRA fine-tuning code and friend-PC commands, without executing training.

## Global Constraints

- Base model: `FacebookAI/xlm-roberta-base`
- Training method: parameter-efficient LoRA
- Final artifact: merged standalone model in `results/huggingface/xlm_roberta_merged/`
- Training datasets: existing `data/processed/train.csv` and `data/processed/val.csv`
- Labels: 0 = negative, 1 = neutral, 2 = positive
- Training may require CUDA, but all code must expose CPU-safe smoke tests
- No training run will execute on this PC
- No hardcoded predictions or benchmark metrics

## Review Focus

- Missing CUDA: fail before dataset/model work with an explicit GPU requirement and retain an inference-only smoke-test mode.
- Invalid split schema or duplicate IDs: reject with the repository's existing split contract.
- Mixed-case or missing text: reject missing values and preserve Unicode Roman-script and Devanagari text.
- Wrong export format: verify `config.json`, tokenizer files, and `model.safetensors` are present.
- Friend-PC portability: verify the merged model loads with `AutoTokenizer` and `AutoModelForSequenceClassification` on a clean local path.

---

### Task 1: Add Hugging Face Training Dependencies

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_huggingface_training.py`

**Interfaces:**
- Consumes: none
- Produces: dependencies available for `src/huggingface_training.py`

- [ ] **Step 1: Add dependency assertions to a test module**

```python
def test_huggingface_training_dependencies_are_declared():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    for package in ("transformers", "datasets", "peft", "accelerate", "safetensors"):
        assert package in requirements
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_huggingface_training.py::test_huggingface_training_dependencies_are_declared -q`

Expected: FAIL because the new package declarations are absent.

- [ ] **Step 3: Add exact dependency names to `requirements.txt`**

Add `transformers`, `datasets`, `peft`, `accelerate`, and `safetensors` without a GPU-specific torch extra.

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/test_huggingface_training.py::test_huggingface_training_dependencies_are_declared -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_huggingface_training.py
git commit -m "feat: add Hugging Face training dependencies"
```

### Task 2: Add Dataset and CLI Contract

**Files:**
- Create: `src/huggingface_training.py`
- Test: `tests/test_huggingface_training.py`

**Interfaces:**
- Consumes: `data/processed/train.csv`, `data/processed/val.csv`
- Produces:
  - `build_dataset(train_path: Path, val_path: Path) -> tuple[object, object]`
  - `parse_args() -> argparse.Namespace`
  - `require_cuda() -> torch.device`

- [ ] **Step 1: Add schema and CLI tests**

```python
def test_parse_args_uses_xlmr_and_lora_defaults():
    args = parse_args([])
    assert args.model_name == "FacebookAI/xlm-roberta-base"
    assert args.output_dir == Path("results/huggingface/xlm_roberta_lora")
    assert args.merged_dir == Path("results/huggingface/xlm_roberta_merged")
    assert args.max_length == 128
    assert args.batch_size == 16
    assert args.learning_rate == 2e-5
    assert args.epochs == 5
```

```python
def test_require_cuda_rejects_cpu_only_environment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda()
```

```python
def test_build_dataset_rejects_invalid_labels(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"uid": ["1"], "text_clean": ["hello"], "label": [3]}).to_csv(train, index=False)
    val = tmp_path / "val.csv"
    pd.DataFrame({"uid": ["2"], "text_clean": ["good"], "label": [2]}).to_csv(val, index=False)
    with pytest.raises(ValueError, match="invalid labels"):
        build_dataset(train, val)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_huggingface_training.py -k "parse_args or require_cuda or build_dataset" -q`

Expected: FAIL because the module is missing.

- [ ] **Step 3: Implement the argument and dataset contract in `src/huggingface_training.py`**

Load both CSV files with `keep_default_na=False`, validate them through `src.contracts.validate_split_frame`, convert labels to Hugging Face integer features, and ensure text values are strings. The module may define helper signatures beyond these interfaces where needed.

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_huggingface_training.py -k "parse_args or require_cuda or build_dataset" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/huggingface_training.py tests/test_huggingface_training.py
git commit -m "feat: add XLM-R training dataset contract"
```

### Task 3: Add LoRA Model, Tokenization, and Metrics

**Files:**
- Modify: `src/huggingface_training.py`
- Test: `tests/test_huggingface_training.py`

**Interfaces:**
- Produces:
  - `build_lora_model(model_name: str, num_labels: int) -> tuple[object, object]`
  - `compute_metrics(eval_prediction: object) -> dict[str, float]`

- [ ] **Step 1: Add model and metrics tests**

```python
def test_compute_metrics_reports_accuracy_and_macro_f1():
    result = compute_metrics(SimpleNamespace(
        predictions=np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]]),
        label_ids=np.array([0, 2]),
    ))
    assert set(result) >= {"accuracy", "macro_f1", "weighted_f1"}
```

```python
def test_lora_defaults_target_attention_modules():
    config = build_lora_config()
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) >= {"query", "value", "key", "dense"}
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_huggingface_training.py -k "compute_metrics or lora_defaults" -q`

Expected: FAIL because LoRA/config functions are missing.

- [ ] **Step 3: Implement tokenizer/model/LoRA/metrics**

Use `AutoTokenizer.from_pretrained(model_name)`, `AutoModelForSequenceClassification.from_pretrained(..., num_labels=3, id2label=..., label2id=...)`, and PEFT `LoraConfig`. Target modules must include XLM-R attention `query`, `key`, `value`, and `dense`.

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_huggingface_training.py -k "compute_metrics or lora_defaults" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/huggingface_training.py tests/test_huggingface_training.py
git commit -m "feat: add XLM-R LoRA model and metrics"
```

### Task 4: Add Training and Portable Export

**Files:**
- Modify: `src/huggingface_training.py`
- Test: `tests/test_huggingface_training.py`

**Interfaces:**
- Consumes: `Dataset` splits and CLI arguments
- Produces:
  - `train(args: argparse.Namespace) -> dict[str, object]`
  - `merge_and_export(adapter_dir: Path, merged_dir: Path) -> Path`

- [ ] **Step 1: Add export tests**

```python
def test_merge_and_export_requires_adapter(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter"):
        merge_and_export(tmp_path / "missing", tmp_path / "merged")
```

```python
def test_training_cli_does_not_auto_start(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["src/huggingface_training.py"])
    assert parse_args([]).dry_run is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_huggingface_training.py -k "merge_and_export or training_cli" -q`

Expected: FAIL because export/`dry_run` behavior is missing.

- [ ] **Step 3: Implement training/export**

Default to a `--dry-run` CLI that loads and validates data but does not train. Add `--train` for explicit GPU execution. Use class-balanced weighted loss or report that full balanced retraining is configured without silently changing validation metrics. Save the best adapter, merge with `merge_and_unload()`, and export to `safetensors` in the merged directory.

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_huggingface_training.py -k "merge_and_export or training_cli" -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/huggingface_training.py tests/test_huggingface_training.py
git commit -m "feat: add XLM-R LoRA training and merge export"
```

### Task 5: Add Friend-PC Setup and Transfer Commands

**Files:**
- Create: `scripts/train_xlmr_friend_pc.ps1`
- Create: `scripts/export_xlmr_model.ps1`
- Create: `docs/XLMR_TRAINING.md`
- Test: `tests/test_huggingface_training.py`

**Interfaces:**
- Produces:
  - PowerShell setup command for the friend's Windows PC
  - PowerShell export command that creates a ZIP artifact

- [ ] **Step 1: Add command-document assertions**

```python
def test_friend_pc_guide_contains_cuda_and_transfer_commands():
    guide = Path("docs/XLMR_TRAINING.md").read_text(encoding="utf-8")
    assert "nvidia-smi" in guide
    assert "python src/huggingface_training.py --train" in guide
    assert "xlmr-model.zip" in guide
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_huggingface_training.py::test_friend_pc_guide_contains_cuda_and_transfer_commands -q`

Expected: FAIL because the guide is absent.

- [ ] **Step 3: Add setup/training/export documentation and scripts**

The guide must cover Git clone, Python virtual environment creation, CUDA-enabled PyTorch installation for the friend's GPU, dependency installation, `nvidia-smi`, dry-run validation, explicit training, ZIP export, and transfer instructions.

- [ ] **Step 4: Run tests and verify the full suite passes**

Run: `pytest tests/test_huggingface_training.py -q`

Expected:  PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_xlmr_friend_pc.ps1 scripts/export_xlmr_model.ps1 docs/XLMR_TRAINING.md tests/test_huggingface_training.py
git commit -m "docs: add GPU friend-PC XLM-R training workflow"
```
