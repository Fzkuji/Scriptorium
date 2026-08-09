#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/e/Scriptorium
CODE="$REPO/code"
PYTHON=/home/qi2/.venvs/scriptorium/bin/python
KEY_FILE="$REPO/provider-api-key.txt"
RUN_ROOT=/home/qi2/scriptorium-runs
OUTPUT="$RUN_ROOT/longmemeval-ms-index105-first3-token15k-wsl-ext4-micro-r1"
LOG_PREFIX="$OUTPUT.launch-001"

test -x "$PYTHON"
test -s "$KEY_FILE"
if [[ -e "$OUTPUT" ]]; then
  echo "refusing existing output: $OUTPUT" >&2
  exit 1
fi

nohup "$PYTHON" "$CODE/scripts/diagnostics/run_writer_micro_smoke.py" \
  --data "$CODE/benchmarks/longmemeval/data/longmemeval_s_cleaned.json" \
  --output-dir "$OUTPUT" \
  --api-key-file "$KEY_FILE" \
  --base-url https://www.packyapi.ai \
  --model deepseek-v4-flash \
  --item-index 105 \
  --sessions 3 \
  --writer-input-token-cap 15000 \
  --max-turns 120 \
  --shell-backend posix-bash \
  >"$LOG_PREFIX.stdout" \
  2>"$LOG_PREFIX.stderr" \
  </dev/null &
pid=$!
printf '%s\n' "$pid" >"$LOG_PREFIX.pid"
echo "pid=$pid"
