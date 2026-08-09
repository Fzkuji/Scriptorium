#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/Scriptorium/code
python=/mnt/e/Scriptorium/.venv/bin/python

launch() {
  local lane="$1"
  local config="$2"
  local run_dir="$3"
  local log_id="$4"

  if [[ -n "${ONLY_LANE:-}" && "$ONLY_LANE" != "$lane" ]]; then
    return
  fi

  mkdir -p "$run_dir"
  nohup "$python" scripts/runners/run_conversation.py \
    --config "$config" \
    < /dev/null \
    > "$run_dir/$log_id.stdout" \
    2> "$run_dir/$log_id.stderr" &
  printf '%s\t%s\n' "$lane" "$!"
}

launch lane1 \
  scripts/configs/locomo_5way_lane1_conv26_token15k_t120_runtimeids_r1.json \
  /home/qi2/scriptorium-runs/locomo-5way-lane1-conv26-token15k-t120-runtimeids-r1 \
  launch-002
launch lane2 \
  scripts/configs/locomo_5way_lane4_conv43_token15k_t120_runtimeids_r1.json \
  /home/qi2/scriptorium-runs/locomo-5way-lane4-conv43-token15k-t120-runtimeids-r1 \
  launch-002
launch lane3 \
  scripts/configs/locomo_6way_lane3_conv47_token15k_t120_runtimeids_r1.json \
  /home/qi2/scriptorium-runs/locomo-6way-lane3-conv47-token15k-t120-runtimeids-r1 \
  launch-001
launch lane4 \
  scripts/configs/locomo_6way_lane4_conv48_token15k_t120_runtimeids_r1.json \
  /home/qi2/scriptorium-runs/locomo-6way-lane4-conv48-token15k-t120-runtimeids-r1 \
  launch-001
launch lane5 \
  scripts/configs/locomo_6way_lane5_conv49_token15k_t120_runtimeids_r1.json \
  /home/qi2/scriptorium-runs/locomo-6way-lane5-conv49-token15k-t120-runtimeids-r1 \
  launch-001
launch lane6 \
  scripts/configs/locomo_6way_lane6_conv50_token15k_t120_runtimeids_r1.json \
  /home/qi2/scriptorium-runs/locomo-6way-lane6-conv50-token15k-t120-runtimeids-r1 \
  launch-001
