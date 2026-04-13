# pause_for_game.ps1 -- free the GPU for gaming.
#
# Pauses train_service (signal-based, keeps heartbeat), disables both
# Task Scheduler entries so no tick fires smoke_train / restarts the
# service during play. Watchdog stays enabled (CPU only).
#
# Resume with scripts/resume_from_game.ps1.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

New-Item -ItemType Directory -Force -Path ".soma-loop/signals" | Out-Null
New-Item -ItemType File -Force -Path ".soma-loop/signals/pause" | Out-Null
Write-Host "train_service: pause signal sent"

$scheduled = @("SOMA Loop Tick", "SOMA Loop Watchdog")
foreach ($name in $scheduled) {
    $exists = schtasks /Query /TN $name 2>$null
    if ($LASTEXITCODE -eq 0) {
        if ($name -eq "SOMA Loop Tick") {
            schtasks /Change /TN $name /DISABLE | Out-Null
            Write-Host "$name disabled"
        } else {
            Write-Host "$name left enabled (CPU-only)"
        }
    }
}

Write-Host ""
Write-Host "GPU freed. Resume with: pwsh -File scripts/resume_from_game.ps1"
