#!/usr/bin/env bash
# Score other memory systems under the same condition as ours.
#
# Usage:
#   baselines/run_comparison.sh SAMPLE SYSTEM...      e.g. 0 mem0 zep naive
#   baselines/run_comparison.sh --list                what can be run
#
# Each system retrieves with its own API, then every system's retrieved
# memories go through one shared answerer and one shared judge. The answerer
# receives only the question and the retrieved memories, so no system can see
# the dataset's gold answer.
#
# Results land in results/comparison-s<SAMPLE>/<system>/. A system that fails
# is reported and skipped; the rest still run.
set -uo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$CODE/../.venv/bin/python}"

# Paths here contain spaces, so name the files rather than word-splitting them.
systems() {
    local f
    for f in "$CODE"/baselines/adapters/run_*.py; do
        f="${f##*/}"; f="${f#run_}"; echo "${f%.py}"
    done
}

usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }

if [[ "${1:-}" == "--list" ]]; then
    echo "systems:"; systems | sed 's/^/  /'
    echo
    echo "Each needs its own dependencies installed; see baselines/README.md."
    exit 0
fi

SAMPLE="${1:-}"
[[ -z "$SAMPLE" || $# -lt 2 ]] && usage
shift

OUT="$CODE/results/comparison-s${SAMPLE}"
mkdir -p "$OUT"
OK=() FAILED=()

for system in "$@"; do
    adapter="$CODE/baselines/adapters/run_${system}.py"
    if [[ ! -f "$adapter" ]]; then
        echo "no such system: $system (try --list)" >&2
        FAILED+=("$system"); continue
    fi

    dir="$OUT/$system"; mkdir -p "$dir"
    echo "===== $system: retrieving  $(date +%H:%M:%S)"
    "$PYTHON" "$adapter" --sample "$SAMPLE" --output "$dir/retrieved.json"
    if [[ ! -s "$dir/retrieved.json" ]]; then
        echo "===== $system: retrieval produced nothing, skipping" >&2
        FAILED+=("$system"); continue
    fi

    echo "===== $system: answering and judging"
    (cd "$CODE" && "$PYTHON" -m scripts.evaluation.evaluate \
        --benchmark locomo \
        --input "$dir/retrieved.json" \
        --output "$dir/eval.json" \
        --metrics judge f1)
    # evaluate.py writes partial results as it goes, so a file that exists is
    # not a file that was scored. Require actual judged records.
    if "$PYTHON" "$CODE/baselines/read_score.py" "$dir/eval.json" >/dev/null 2>&1; then
        OK+=("$system")
    else
        echo "===== $system: no judged records, skipping" >&2
        FAILED+=("$system")
    fi
done

echo
echo "===== comparison, LoCoMo sample $SAMPLE"
for system in "${OK[@]+"${OK[@]}"}"; do
    read -r score count < <("$PYTHON" "$CODE/baselines/read_score.py" "$OUT/$system/eval.json")
    printf "  %-12s %5s  (n=%s)\n" "$system" "$score" "$count"
done
[[ ${#FAILED[@]} -gt 0 ]] && echo "  failed: ${FAILED[*]}" >&2
[[ ${#OK[@]} -eq 0 ]] && exit 1
exit 0
