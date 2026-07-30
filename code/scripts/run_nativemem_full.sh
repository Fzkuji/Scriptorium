#!/bin/bash
# Run NativeMem (v4 two-layer + precise-grep) over all 10 LoCoMo conversations.
# Each sample builds fresh memory + retrieves, output per-sample questions.json.
# Usage: bash scripts/run_nativemem_full.sh <model_tag> <builder_model> <builder_base> <key>
# e.g.:  bash scripts/run_nativemem_full.sh 54mini gpt-5.4-mini http://localhost:8199/v1 x
# No `set -e`: one sample failing (transient API error) must NOT kill the
# whole 10-sample run. Failed samples are retried, then skipped if still bad.
cd "$(dirname "$0")/.."

TAG="${1:-54mini}"
BMODEL="${2:-gpt-5.4-mini}"
BBASE="${3:-http://localhost:8199/v1}"
BKEY="${4:-x}"
OUTDIR="results/v4full-${TAG}-locomo"
mkdir -p "$OUTDIR"

export NATIVEMEM_PROMPT=v4
export BUILDER_MODEL="$BMODEL"
export BUILDER_BASE="$BBASE"
export ALIYUN_KEY="$BKEY"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

for s in 0 1 2 3 4 5 6 7 8 9; do
  out="$OUTDIR/sample${s}_questions.json"
  if [ -f "$out" ]; then
    echo "[full] sample $s already done, skip"
    continue
  fi
  echo "===== [full] building+retrieving sample $s ($(date +%H:%M:%S)) ====="
  for attempt in 1 2; do
    if python3 src/adapters/run_nativemem.py --sample "$s" --output "$out"; then
      break
    fi
    echo "[full] sample $s attempt $attempt failed; $([ $attempt -lt 2 ] && echo retrying || echo giving up)"
    rm -rf "$OUTDIR/memory_sample${s}"  # clean partial build before retry
    sleep 10
  done
done
echo "===== [full] all 10 samples done ====="
