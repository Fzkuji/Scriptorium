#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/e/Scriptorium
CODE="$REPO/code"
PYTHON=/home/qi2/.venvs/scriptorium/bin/python
KEY_FILE="$REPO/provider-api-key.txt"
RUN_ROOT=/home/qi2/scriptorium-runs

test -x "$PYTHON"
test -s "$KEY_FILE"

for worker in 1 2 3 4 5; do
  config="$CODE/scripts/configs/longmemeval_ms_20_index105_124_token15k_t120_wsl_ext4_worker${worker}.json"
  run="$RUN_ROOT/longmemeval-ms-20-index105-124-worker${worker}-wsl-ext4-r1"
  if [[ -e "$run" ]]; then
    echo "refusing existing output: $run" >&2
    exit 1
  fi
  mkdir -p "$run"
  nohup "$PYTHON" "$CODE/scripts/runners/run_longmemeval.py" \
    --config "$config" \
    --api-key-file "$KEY_FILE" \
    >"$run/launch-001.stdout" \
    2>"$run/launch-001.stderr" \
    </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$run/launch-001.pid"
  echo "worker=$worker pid=$pid"
done
