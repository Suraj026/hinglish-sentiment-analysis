$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$modelDir = "results/huggingface/xlm_roberta_merged"
$zipPath = Join-Path $PWD "xlmr-model.zip"

if (-not (Test-Path $modelDir)) {
    throw "Merged model not found: $modelDir"
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path "$modelDir\*" -DestinationPath $zipPath
Write-Output "Created $zipPath"
