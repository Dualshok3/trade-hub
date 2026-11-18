# Startet die lokale Gemma-4-Engine (crit, candidate-0014) im Notbetrieb auf http://127.0.0.1:8000/v1.
#
# Die Engine wurde mit crit 0.17.1 plus lokalen Gemma-Anpassungen gebaut. Dieser Stand ist im crit-Repo als Tag
# `gemma4-engine-c0014` gesichert. Das Skript entpackt ihn einmalig nach %LOCALAPPDATA%\crit-gemma-runtime und
# startet `crit serve` mit dem venv des crit-Repos. Mit crit 0.19 startet candidate-0014 nicht (siehe PLAN.md 4.1.2).
#
# Aufruf:  powershell -ExecutionPolicy Bypass -File scripts\start-engine.ps1
# Modell-ID in der API: run-20261109T095346Z-1b210ccf/candidate-0014

param(
    [string]$Crit = "C:\python_projects\applications\crit",
    [string]$Tag = "gemma4-engine-c0014",
    [string]$Run = "run-20261109T095346Z-1b210ccf",
    [string]$Candidate = "candidate-0014",
    [string]$StepLibrary = "D:\crit-workspace\build\core\crit_core_steps.dll",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Runtime = Join-Path $env:LOCALAPPDATA "crit-gemma-runtime\$Tag"

if (-not (Test-Path (Join-Path $Runtime "core\python\crit_core"))) {
    Write-Host "Entpacke $Tag nach $Runtime ..."
    New-Item -ItemType Directory -Force $Runtime | Out-Null
    $archive = Join-Path $env:TEMP "$Tag.tar"
    git -C $Crit archive --format=tar -o $archive $Tag
    tar -xf $archive -C $Runtime
    Remove-Item $archive
}

$env:PYTHONPATH = @("cli", "api", "kernels", "models", "core\python", "inference" | ForEach-Object { Join-Path $Runtime $_ }) -join ";"
$env:CRIT_CORE_STEP_LIBRARY = $StepLibrary

& (Join-Path $Crit ".venv\Scripts\python.exe") -m crit.cli serve `
    --workspace (Join-Path $Crit "runs") --run $Run --candidate $Candidate --project $Crit --port $Port
