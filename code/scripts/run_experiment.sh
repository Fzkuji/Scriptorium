#!/usr/bin/env bash
# Build memory, evaluate, and summarize — for one run or a sweep.
#
# Usage:
#   scripts/run_experiment.sh CONFIG                      one run, as configured
#   scripts/run_experiment.sh CONFIG --samples A B C      one run per conversation
#   scripts/run_experiment.sh CONFIG --caps 4096 8192     one run per input size
#
# --caps sweeps writer_input_token_cap, which is how much conversation the
# Writer sees at once. Each variant is retried once; a variant that still fails
# is skipped rather than ending the sweep. Success is decided by whether
# eval_full.json holds judged records, not by an exit code.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
RUNNER="$REPO/code/scripts/runners/run_conversation.py"
SCORE="$REPO/code/baselines/read_score.py"
SUMMARY="$REPO/code/scripts/analysis/analyze_run.py"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

CONFIG="${1:-}"
[[ -z "$CONFIG" || ! -f "$CONFIG" ]] && usage
shift

MODE="" ; VALUES=()
case "${1:-}" in
    --samples|--caps) MODE="$1"; shift; VALUES=("$@") ;;
    "") ;;
    *) echo "unknown option: $1" >&2; usage ;;
esac

OUT_BASE="$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1])).get('output_dir',''))" "$CONFIG")"
if [[ -z "$OUT_BASE" ]]; then
    echo "config has no output_dir: $CONFIG" >&2
    exit 2
fi

run_one() {  # $1 = label ("" for none), $2.. = extra runner args
    local label="$1"; shift
    local out="$OUT_BASE" ; local args=("--config" "$CONFIG" "$@")
    if [[ -n "$label" ]]; then
        out="${OUT_BASE}-${label}"
        args+=("--output-dir" "$out")
    fi
    for attempt in 1 2; do
        echo "===== ${label:-run} attempt $attempt $(date +%H:%M:%S)"
        "$PYTHON" "$RUNNER" "${args[@]}"
        # A judged record, not a file: the unified evaluator writes its
        # output before judging anything, so an existing file proves nothing.
        if "$PYTHON" "$SCORE" "$out/eval_full.json" >/dev/null 2>&1; then
            echo "===== ${label:-run} ok"
            DONE+=("$out")
            return 0
        fi
        echo "===== ${label:-run} failed (attempt $attempt)" >&2
    done
    echo "===== ${label:-run} skipped after 2 attempts" >&2
    return 1
}

DONE=()
case "$MODE" in
    --samples) for v in "${VALUES[@]}"; do run_one "$v" --sample-id "$v"; done ;;
    --caps)    for v in "${VALUES[@]}"; do run_one "cap$v" --writer-input-token-cap "$v"; done ;;
    *)         run_one "" ;;
esac

if [[ ${#DONE[@]} -gt 0 ]]; then
    echo
    echo "===== summary"
    "$PYTHON" "$SUMMARY" "${DONE[@]}"
fi
