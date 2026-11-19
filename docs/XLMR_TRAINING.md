# XLM-RoBERTa LoRA GPU Training

This guide trains `FacebookAI/xlm-roberta-base` on Google Colab or a friend's Windows NVIDIA-GPU PC. It does not start training on the no-GPU PC.

## Google Colab T4 workflow

Open `notebooks/train_xlmr_colab.ipynb` in Google Colab.

1. Select **Runtime > Change runtime type > T4 GPU**.
2. Mount Google Drive when prompted.
3. Run the cells in order.
4. The final model ZIP is copied to:

   ```text
   /MyDrive/xlmr-model.zip
   ```

Manual Colab setup is:

```python
!nvidia-smi
assert torch.cuda.is_available()
!git clone https://github.com/Suraj026/hinglish-sentiment-analysis.git
%cd hinglish-sentiment-analysis
!pip install -r requirements.txt
!python src/huggingface_training.py
!python src/huggingface_training.py --train --epochs 3 --batch-size 16 --gradient-accumulation-steps 2 --max-length 128 --learning-rate 2e-5
```

After training, create the transferable ZIP from a Colab cell:

```python
import shutil
shutil.make_archive('/content/xlmr-model', 'zip', root_dir='results/huggingface/xlm_roberta_merged')
from google.colab import files
files.download('/content/xlmr-model.zip')
```

On the T4 profile, these settings normally complete 3 epochs within about 30–45 minutes, plus 5–10 minutes for model download and merged export. Colab availability and session throttling can change this.

## 1. Create the project ZIP on your PC

Run this from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_hinglish_training.ps1
```

This creates `hinglish-training.zip` and includes only `src/`, `data/`, `scripts/`, `docs/`, `notebooks/`, `requirements.txt`, and `README.md`. It does not include `.venv`, `.git`, `frontend/node_modules`, or build outputs.

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
- Effective batch size: 32
- Gradient accumulation: 2
- Learning rate: `2e-5`
- Epochs: 3
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
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch transformers sentencepiece
```

Run local CPU inference:

```powershell
python scripts/predict_xlmr.py --model-dir xlmr-model --text "Bahut badhiya service" --text "This service is disappointing" --json
```

For a simple text-only result:

```powershell
python scripts/predict_xlmr.py --model-dir xlmr-model --text "Yaar ye movie bahut acchi thi"
```

The predictor loads the merged `xlmr-model/` directory. It accepts Roman-script Hinglish and English and can run on CPU. Training is not repeated locally.
