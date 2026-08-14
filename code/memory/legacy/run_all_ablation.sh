#!/bin/bash
# Run all ablation experiments on sample 0 via ChatGPT proxy
# Each variant: build wiki + verify 50 QA
set -e
cd "$(dirname "$0")"

OUTDIR="ablation_results"
MODEL="gpt-5.5"
SAMPLE=0
VERIFY=50

mkdir -p "$OUTDIR"

echo "=== Ablation Experiments ==="
echo "Model: $MODEL, Sample: $SAMPLE, QA: $VERIFY"
echo ""

# 1. No Memory (lower bound) - fast, no wiki build needed
echo "[1/5] Running: no_memory"
python3 run_ablation.py --variant no_memory --sample $SAMPLE --verify $VERIFY --outdir "$OUTDIR" --proxy --model $MODEL 2>&1 | tee "$OUTDIR/log_no_memory.txt"
echo ""

# 2. Full Context (upper bound) - no wiki build needed
echo "[2/5] Running: full_context"
python3 run_ablation.py --variant full_context --sample $SAMPLE --verify $VERIFY --outdir "$OUTDIR" --proxy --model $MODEL 2>&1 | tee "$OUTDIR/log_full_context.txt"
echo ""

# 3. w/o Copy (LLM rephrases instead of copying original text)
echo "[3/5] Running: no_copy"
python3 run_ablation.py --variant no_copy --sample $SAMPLE --verify $VERIFY --outdir "$OUTDIR" --proxy --model $MODEL 2>&1 | tee "$OUTDIR/log_no_copy.txt"
echo ""

# 4. w/o RSP (chronological storage instead of retrieval-simulated placement)
echo "[4/5] Running: no_rsp"
python3 run_ablation.py --variant no_rsp --sample $SAMPLE --verify $VERIFY --outdir "$OUTDIR" --proxy --model $MODEL 2>&1 | tee "$OUTDIR/log_no_rsp.txt"
echo ""

# 5. Flat Memory (single file, no hierarchy)
echo "[5/5] Running: flat"
python3 run_ablation.py --variant flat --sample $SAMPLE --verify $VERIFY --outdir "$OUTDIR" --proxy --model $MODEL 2>&1 | tee "$OUTDIR/log_flat.txt"
echo ""

echo "=== All ablation experiments complete ==="
echo "Results in $OUTDIR/"
