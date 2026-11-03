param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$pidFile = Join-Path $PSScriptRoot 'training.pid'
if (Test-Path $pidFile) {
    $oldId = [int](Get-Content $pidFile)
    if (Get-Process -Id $oldId -ErrorAction SilentlyContinue) { throw "Training process $oldId is already running. Inspect it before starting another run." }
}
if (!(Test-Path 'data\processed\audit.json')) { throw 'Dataset preparation is not complete.' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$arguments = @('-u', 'train.py')
if ($Resume) {
    $checkpoint = (Resolve-Path -LiteralPath $Resume).Path
    $arguments += @('--resume', ('"' + $checkpoint + '"'))
}
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "training-$stamp.log" -RedirectStandardError "training-$stamp.err.log" -PassThru
$process.Id | Set-Content $pidFile
Write-Output "Training launched: PID $($process.Id); logs training-$stamp.log and training-$stamp.err.log"
