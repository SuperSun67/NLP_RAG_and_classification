$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$outputPath = Join-Path $PSScriptRoot 'runs\wongnai'
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$pidPath = Join-Path $outputPath 'process_id.txt'
if (Test-Path -LiteralPath $pidPath) {
    $previousId = [int](Get-Content -LiteralPath $pidPath)
    if (Get-Process -Id $previousId -ErrorAction SilentlyContinue) {
        throw "Training process $previousId is already running."
    }
}
if (Test-Path -LiteralPath (Join-Path $outputPath 'run_config.json')) {
    throw 'A previous run exists. Use train.py with a different --output directory.'
}
$process = Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') `
    -ArgumentList '-X', 'utf8', '-u', 'train.py' `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $outputPath 'train.log') `
    -RedirectStandardError (Join-Path $outputPath 'stderr.log')
$process.Id | Set-Content -LiteralPath $pidPath
Write-Output "Training process started: $($process.Id)"
Write-Output "Progress: $outputPath\status.json"
