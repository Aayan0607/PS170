param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python environment missing. Follow setup in README.md.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) {
    throw 'Frontend build missing. Run npm.cmd ci and npm.cmd run build in frontend.'
}
Write-Host "Nirikshan Netra: http://127.0.0.1:$Port"
& $pythonPath -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
