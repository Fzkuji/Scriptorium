#!/usr/bin/env bash
# Build memory for one or more LoCoMo conversations, evaluate, and summarize.
#
# Usage:
#   scripts/run_experiment.sh CONFIG [SAMPLE_ID ...]
#
# With no sample ids the config's own sample_id is used. Each sample is retried
# once, and a sample that still fails is skipped rather than ending the run.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
RUNNER="$REPO/code/scripts/nativemem/run_locomo.py"

CONFIG="${1:-}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "usage: $0 CONFIG [SAMPLE_ID ...]" >&2
    [[ -n "$CONFIG" ]] && echo "config not found: $CONFIG" >&2
    exit 2
fi
shift
SAMPLES=("$@")

OUT_BASE="$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1])).get('output_dir',''))" "$CONFIG")"
if [[ -z "$OUT_BASE" ]]; then
    echo "config has no output_dir: $CONFIG" >&2
    exit 2
fi

run_one() {
    local args=("--config" "$CONFIG") out="$OUT_BASE"
    if [[ -n "${1:-}" ]]; then
        args+=("--sample-id" "$1" "--output-dir" "${OUT_BASE}-$1")
        out="${OUT_BASE}-$1"
    fi
    for attempt in 1 2; do
        echo "===== ${1:-default} attempt $attempt $(date +%H:%M:%S)"
        "$PYTHON" "$RUNNER" "${args[@]}"
        # Trust the artifact, not the exit code: a runner can fail after
        # writing partial output, and a wrapper's $? is easy to read wrong.
        if [[ -f "$out/eval_full.json" ]]; then
            echo "===== ${1:-default} ok"
            return 0
        fi
        echo "===== ${1:-default} failed (attempt $attempt)" >&2
    done
    echo "===== ${1:-default} skipped after 2 attempts" >&2
    return 1
}

DONE=()
if [[ ${#SAMPLES[@]} -eq 0 ]]; then
    run_one "" && DONE+=("$OUT_BASE")
else
    for s in "${SAMPLES[@]}"; do
        run_one "$s" && DONE+=("${OUT_BASE}-$s")
    done
fi

if [[ ${#DONE[@]} -gt 0 ]]; then
    echo
    echo "===== summary"
    "$PYTHON" "$REPO/code/scripts/analysis/analyze_run.py" "${DONE[@]}"
fi
