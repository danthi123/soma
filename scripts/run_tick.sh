#!/usr/bin/env bash
# run_tick.sh — invoke `claude --print` with the autonomous-loop tick prompt.
# Derives the exit code from sentinel files written by the tick session.

set -u

cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOG_DIR=".soma-loop/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/tick-$(date +%s).log"

# Clear last-tick failure markers so a retry starts from a blank slate.
rm -f .soma-loop/state/tick_failure.json

ALLOWED="Bash(git:*) Bash(python:*) Bash(ruff:*) Bash(mypy:*) Bash(pytest:*) Bash(taskkill:*) Bash(kill:*) Bash(curl:*) Read Edit Write Grep Glob Skill"
DISALLOWED="Agent WebFetch WebSearch NotebookEdit"

PROMPT_FILE="docs/loop_tick_prompt.md"
if [ ! -f "$PROMPT_FILE" ]; then
  echo "FATAL: $PROMPT_FILE missing" >&2
  exit 2
fi

cat "$PROMPT_FILE" | claude --print --dangerously-skip-permissions \
  --allowedTools "$ALLOWED" \
  --disallowedTools "$DISALLOWED" \
  > "$LOG" 2>&1 || true

# Detect Claude CLI failure patterns (G71).
if grep -qiE "rate limit|authentication|expired|quota exceeded" "$LOG"; then
  mkdir -p .soma-loop/state
  cat > .soma-loop/state/claude_cli_error.json <<EOF
{"ts": $(date +%s), "log": "$LOG"}
EOF
fi

# Exit code derives from sentinel files the tick may have produced.
if [ -f .soma-loop/state/tick_failure.json ]; then
  exit 2
fi
if [ -f .soma-loop/STOP ] || [ -f .soma-loop/state/baseline_broken.json ]; then
  exit 1
fi
exit 0
