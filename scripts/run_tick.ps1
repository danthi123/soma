# run_tick.ps1 — invoke `claude --print` with the tick prompt.
# Windows Task Scheduler entrypoint; mirrors run_tick.sh.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Windows Task Scheduler runs powershell with a stripped PATH that is
# missing git, so git rev-parse subprocesses inside the test suite and
# scripts fail with FileNotFoundError. Prepend the standard Git for
# Windows locations before doing anything else.
$gitCandidates = @(
    "C:\Program Files\Git\cmd",
    "C:\Program Files\Git\bin",
    "C:\Program Files\Git\mingw64\bin"
)
foreach ($g in $gitCandidates) {
    if ((Test-Path $g) -and ($env:PATH -notlike "*$g*")) {
        $env:PATH = "$g;$env:PATH"
    }
}

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

$prompt | claude --print --dangerously-skip-permissions `
    --allowedTools $allowed `
    --disallowedTools $disallowed `
    *> $log

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
