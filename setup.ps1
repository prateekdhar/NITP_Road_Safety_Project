param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE) { throw 'Python 3.12 or a compatible Python installation is required.' }
}
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -u download_dataset.py
if ($LASTEXITCODE) { throw 'Dataset download failed.' }
& .\.venv\Scripts\python.exe -u prepare_dataset.py
if ($LASTEXITCODE) { throw 'Dataset preparation failed.' }
