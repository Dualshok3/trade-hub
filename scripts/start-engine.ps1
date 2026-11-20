# Startet die lokale Gemma-4-Engine (crit) auf http://127.0.0.1:8000/v1.
#
# Standard: candidate-0024 auf crit 0.21, zertifiziert am 20.11.2026, 200k Kontext. Das Startskript im Run setzt
# die MSVC-Umgebung und Cores eigene Laufzeit von crit 0.21 und braucht das crit-Checkout im aktuellen Stand
# (Branch feat/package-tool-format). Es läuft im Vordergrund, bis es beendet wird.
#   Modell-ID in der API: run-20261109T095346Z-1b210ccf/candidate-0024
#
# Rückfall: -Fallback startet candidate-0014 (crit 0.17.1 plus lokale Gemma-Anpassungen, gesichert als Tag
# `gemma4-engine-c0014`). Nie beide gleichzeitig starten, es gibt nur eine GPU.
#   Modell-ID in der API: run-20261109T095346Z-1b210ccf/candidate-0014
#
# Aufruf:  powershell -ExecutionPolicy Bypass -File scripts\start-engine.ps1 [-Fallback]
# Hinweis: GET /v1/models bricht derzeit ab (alte 0.17-Kandidaten im Run). Die Modell-ID fest konfigurieren
# (configs/agent.toml) und chat/completions direkt aufrufen.

param(
    [switch]$Fallback,
    [string]$Crit = "C:\python_projects\applications\crit",
    [string]$Run = "run-20261109T095346Z-1b210ccf",
    [string]$Candidate = "candidate-0024",
    # nur für -Fallback:
    [string]$Tag = "gemma4-engine-c0014",
    [string]$FallbackCandidate = "candidate-0014",
    [string]$StepLibrary = "D:\crit-workspace\build\core\crit_core_steps.dll",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

if (-not $Fallback) {
    $start = Join-Path $Crit "runs\$Run\start-$Candidate.cmd"
    if (-not (Test-Path $start)) { throw "Startskript fehlt: $start" }
    & cmd /c $start
    exit $LASTEXITCODE
}

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
    --workspace (Join-Path $Crit "runs") --run $Run --candidate $FallbackCandidate --project $Crit --port $Port
