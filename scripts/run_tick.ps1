# run_tick.ps1 — invoke `claude --print` with the tick prompt.
# Windows Task Scheduler entrypoint; mirrors run_tick.sh.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = ".soma-loop/logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = [int][double]::Parse((Get-Date -UFormat %s))
$log = Join-Path $logDir "tick-$timestamp.log"

# Clear last-tick failure markers so a retry starts from a blank slate.
Remove-Item -ErrorAction SilentlyContinue ".soma-loop/state/tick_failure.json"

$allowed = "Bash(git:*) Bash(python:*) Bash(ruff:*) Bash(mypy:*) Bash(pytest:*) Bash(taskkill:*) Bash(kill:*) Bash(curl:*) Read Edit Write Grep Glob Skill"
$disallowed = "Agent WebFetch WebSearch NotebookEdit"

$promptFile = "docs/loop_tick_prompt.md"
if (-not (Test-Path $promptFile)) {
    Write-Error "FATAL: $promptFile missing"
    exit 2
}

$prompt = Get-Content -Raw $promptFile

claude --print --dangerously-skip-permissions `
    --allowedTools $allowed `
    --disallowedTools $disallowed `
    --prompt $prompt *> $log

# Detect Claude CLI failure patterns (G71).
if (Select-String -Path $log -Pattern "rate limit|authentication|expired|quota exceeded" -Quiet) {
    New-Item -ItemType Directory -Force -Path ".soma-loop/state" | Out-Null
    $err = @{ ts = $timestamp; log = $log } | ConvertTo-Json -Compress
    Set-Content -Path ".soma-loop/state/claude_cli_error.json" -Value $err
}

# Exit code derives from sentinel files the tick may have produced.
if (Test-Path ".soma-loop/state/tick_failure.json") { exit 2 }
if ((Test-Path ".soma-loop/STOP") -or (Test-Path ".soma-loop/state/baseline_broken.json")) { exit 1 }
exit 0
