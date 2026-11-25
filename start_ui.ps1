param(
    [int]$Port = 8000,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not $Background) {
    & $python -X utf8 app.py --port $Port --device $Device
    exit $LASTEXITCODE
}
$logPath = Join-Path $PSScriptRoot 'runs\ui'
New-Item -ItemType Directory -Force -Path $logPath | Out-Null
$pidPath = Join-Path $logPath "process-$Port.txt"
if (Test-Path -LiteralPath $pidPath) {
    $previousId = [int](Get-Content -LiteralPath $pidPath)
    if (Get-Process -Id $previousId -ErrorAction SilentlyContinue) {
        throw "UI process $previousId is already running."
    }
}
$process = Start-Process -FilePath $python -ArgumentList '-X', 'utf8', '-u', 'app.py', '--port', $Port, '--device', $Device `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logPath "stdout-$Port.log") `
    -RedirectStandardError (Join-Path $logPath "stderr-$Port.log")
$process.Id | Set-Content -LiteralPath $pidPath
Write-Output "UI starting at http://127.0.0.1:$Port (PID $($process.Id))"
