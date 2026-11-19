param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("dry-run", "train")]
    [string]$Mode = "dry-run"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.10 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if ($Mode -eq "train") {
    & .\.venv\Scripts\python.exe -m pip show torch | Select-String "Version"
    & nvidia-smi
    & .\.venv\Scripts\python.exe src/huggingface_training.py --train
}
else {
    & .\.venv\Scripts\python.exe src/huggingface_training.py
}
