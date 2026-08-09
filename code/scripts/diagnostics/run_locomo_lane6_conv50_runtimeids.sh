#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/Scriptorium/code
run_dir=/home/qi2/scriptorium-runs/locomo-6way-lane6-conv50-token15k-t120-runtimeids-r1
mkdir -p "$run_dir"

exec /mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_conversation.py \
  --config scripts/configs/locomo_6way_lane6_conv50_token15k_t120_runtimeids_r1.json \
  > "$run_dir/launch-002.stdout" \
  2> "$run_dir/launch-002.stderr"
