# XLM-RoBERTa LoRA GPU Training

This guide trains `FacebookAI/xlm-roberta-base` on a friend's Windows NVIDIA-GPU PC. It does not start training on the no-GPU PC.

## 1. Create the project ZIP on your PC

Run this from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_hinglish_training.ps1
```

This creates `hinglish-training.zip` and includes only `src/`, `data/`, `scripts/`, `docs/`, `requirements.txt`, and `README.md`. It does not include `.venv`, `.git`, `frontend/node_modules`, or build outputs.

Send `hinglish-training.zip` to the friend's GPU PC.

## 2. Extract and open the project on the friend's PC

```powershell
Expand-Archive -Path hinglish-training.zip -DestinationPath hinglish-sentiment-analysis
cd hinglish-sentiment-analysis
```

Do not copy or reuse your local `.venv`. Create a new virtual environment on the GPU PC.

## 3. Verify the GPU PC

```powershell
nvidia-smi
py -3.10 --version
```

Use one of these CUDA-compatible PyTorch installations:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For a GPU package requiring CUDA 12.1 instead, use:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Confirm PyTorch can see the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

## 4. Install the project

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Validate without training

```powershell
python src/huggingface_training.py
```

This validates the train/validation schemas and prints a dry-run result. It does not download the training base model or write checkpoints.

## 6. Fine-tune with LoRA

```powershell
python src/huggingface_training.py --train
```

For the repository helper that creates the environment, validates dependencies, checks the GPU, and starts training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_xlmr_friend_pc.ps1 -Mode train
```

Default training settings:

- Base: `FacebookAI/xlm-roberta-base`
- Method: LoRA
- Labels: 0 negative, 1 neutral, 2 positive
- Sequence length: 128
- Batch size: 16
- Learning rate: `2e-5`
- Epochs: 5
- Seed: 42
- LoRA rank: 16

The script saves the best adapter under `results/huggingface/xlm_roberta_lora/` and the merged standalone model under `results/huggingface/xlm_roberta_merged/`.

## 7. Export for your PC

```powershell
powershell -ExecutionPolicy Bypass -File scripts/export_xlmr_model.ps1
```

This creates `xlmr-model.zip`.

Transfer `xlmr-model.zip` using Google Drive, OneDrive, USB, or Hugging Face Hub. Do not transfer only the LoRA adapter unless you also transfer the base model and adapter separately.

## 8. Load on your no-GPU PC

```powershell
Expand-Archive -Path xlmr-model.zip -DestinationPath xlmr-model
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; p='xlmr-model'; t=AutoTokenizer.from_pretrained(p); m=AutoModelForSequenceClassification.from_pretrained(p); print(m.config.id2label)"
```
