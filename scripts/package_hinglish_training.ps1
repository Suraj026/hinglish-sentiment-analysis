$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$zipPath = Join-Path $PWD "hinglish-training.zip"
$stageDir = Join-Path $env:TEMP ("hinglish-training-" + [guid]::NewGuid())
$packageFiles = @(
    "src",
    "data",
    "scripts",
    "docs",
    "requirements.txt",
    "README.md"
)

New-Item -ItemType Directory -Path $stageDir | Out-Null
foreach ($item in $packageFiles) {
    if (Test-Path $item) {
        Copy-Item -Path $item -Destination $stageDir -Recurse -Force
    }
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path "$stageDir\*" -DestinationPath $zipPath
Remove-Item $stageDir -Recurse -Force
Write-Output "Created $zipPath"
