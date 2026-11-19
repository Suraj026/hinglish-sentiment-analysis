# XLM-RoBERTa LoRA GPU Training

This guide trains `FacebookAI/xlm-roberta-base` on a friend's Windows NVIDIA-GPU PC. It does not start training on the no-GPU PC.

## 1. Verify the GPU PC

```powershell
nvidia-smi
py -3.10 --version
git clone https://github.com/Suraj026/hinglish-sentiment-analysis.git
cd hinglish-sentiment-analysis
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

## 2. Install the project

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Validate without training

```powershell
python src/huggingface_training.py
```

This validates the train/validation schemas and prints a dry-run result. It does not download the training base model or write checkpoints.

## 4. Fine-tune with LoRA

```powershell
python src/huggingface_training.py --train
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

## 5. Export for your PC

```powershell
powershell -ExecutionPolicy Bypass -File scripts/export_xlmr_model.ps1
```

This creates `xlmr-model.zip`.

Transfer `xlmr-model.zip` using Google Drive, OneDrive, USB, or Hugging Face Hub. Do not transfer only the LoRA adapter unless you also transfer the base model and adapter separately.

## 6. Load on your no-GPU PC

```powershell
Expand-Archive -Path xlmr-model.zip -DestinationPath xlmr-model
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; p='xlmr-model'; t=AutoTokenizer.from_pretrained(p); m=AutoModelForSequenceClassification.from_pretrained(p); print(m.config.id2label)"
```
