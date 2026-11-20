# Ein geplanter Live-Takt: Snapshot, Entscheidungen der LLM-Agenten und Vergleichs-Depots, Ausführung.
#
# Wird von der Windows-Aufgabenplanung gestartet (Aufgabe "trade-hub tick", Mo–Fr 09:30, 13:30, 15:45).
# Ist die lokale Engine nicht erreichbar, fällt der Takt für ALLE Agenten aus, damit der Vergleich fair bleibt
# (PLAN.md 3.5). Jeder Lauf schreibt ein Log nach data\logs\.
#
# Aufruf von Hand:  powershell -ExecutionPolicy Bypass -File scripts\run-tick.ps1 [-Agents gemma,opus,deepseek]

param(
    [string]$Agents = "gemma,opus,deepseek",
    [switch]$DryRun,  # nur zum Testen: zeigt die Depots statt einen Takt auszulösen
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Uv = "$env:USERPROFILE\.local\bin\uv.exe",
    [string]$Health = "http://127.0.0.1:8000/health"
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $Repo
$logDir = Join-Path $Repo "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("tick-" + (Get-Date -Format "yyyyMMdd-HHmm") + ".log")

function Write-Log([string]$msg) {
    $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "  " + $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

Write-Log "Takt startet, Agenten: $Agents"
try {
    $h = Invoke-RestMethod -Uri $Health -TimeoutSec 10
    if (-not $h.ready) { throw "Engine nicht bereit" }
    Write-Log ("Engine bereit: " + $h.engine.engine.candidate_id)
} catch {
    Write-Log "AUSGEFALLEN: Engine nicht erreichbar ($($_.Exception.Message)). Kein Agent handelt in diesem Takt."
    exit 2
}

# Windows PowerShell 5.1 schreibt mit *>> UTF-16; deshalb Zeile für Zeile als UTF-8 anhängen.
# Breite fest, damit rich die Tabellen nicht auf 80 Zeichen umbricht.
$env:COLUMNS = "160"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$cmd = if ($DryRun) { @("run", "trading-agent", "depots") } else { @("run", "trading-agent", "tick", "--agents", $Agents) }
& $Uv @cmd 2>&1 | ForEach-Object { Add-Content -Path $log -Value "$_" -Encoding utf8 }
$code = $LASTEXITCODE
Write-Log "Takt beendet, Exit-Code $code"
exit $code
