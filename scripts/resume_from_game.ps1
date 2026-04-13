# resume_from_game.ps1 -- return the GPU to SOMA after gaming.
#
# Reverses pause_for_game.ps1:
#   - signals/resume -> train_service unpauses
#   - re-enables the loop tick task
#
# If train_service has exited (CrashBackoff during a prior NaN episode,
# operator killed it, etc.), this script detects a stale / missing
# train_heartbeat.json and starts a fresh detached service so the loop
# has something to train. The scheduler's Phase-10 restart would also
# handle this at the next tick, but starting manually is ~37 min faster.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

# Clear any stale pause signal, then emit resume.
Remove-Item -ErrorAction SilentlyContinue ".soma-loop/signals/pause"
New-Item -ItemType Directory -Force -Path ".soma-loop/signals" | Out-Null
New-Item -ItemType File -Force -Path ".soma-loop/signals/resume" | Out-Null
Write-Host "train_service: resume signal sent"

# Detect dead service so we can respawn it.
$hb = ".soma-loop/state/train_heartbeat.json"
$needsRestart = $true
if (Test-Path $hb) {
    try {
        $data = Get-Content $hb -Raw | ConvertFrom-Json
        $age = (Get-Date -UFormat %s) - [double]$data.ts
        if ($age -lt 30 -and $data.status -ne "shutdown") {
            $needsRestart = $false
        }
    } catch {
        # malformed heartbeat -> fall through to restart
    }
}
if ($needsRestart) {
    # Clear any CrashBackoff markers so the fresh service starts cleanly.
    Remove-Item -ErrorAction SilentlyContinue ".soma-loop/state/train_permanent_failure.json"
    Remove-Item -ErrorAction SilentlyContinue ".soma-loop/state/train_crash.json"
    $logDir = ".soma-loop/logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir "train_service.log"
    Start-Process python -ArgumentList "-u", "scripts/train_service.py" `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $log `
        -RedirectStandardError $log `
        -WindowStyle Hidden
    Write-Host "train_service: respawned (was stale/dead)"
} else {
    Write-Host "train_service: still alive, just unpaused"
}

schtasks /Change /TN "SOMA Loop Tick" /ENABLE | Out-Null
Write-Host "SOMA Loop Tick re-enabled"

Write-Host ""
Write-Host "GPU back with SOMA. Next tick:"
schtasks /Query /TN "SOMA Loop Tick" /FO LIST 2>$null | Select-String "Next Run" | ForEach-Object { Write-Host "  $_" }
