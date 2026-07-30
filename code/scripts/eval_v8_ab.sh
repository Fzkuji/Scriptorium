#!/bin/bash
# 方案A(两模型:检索喂事件行+原文→独立answerer) vs 方案B(单模型:检索+直接答)
# 对比。build 一次，两方案复用同一记忆库(--memory-dir)，只差检索/答题方式。
cd "$(dirname "$0")/.."

SAMPLE="${1:-0}"
export NATIVEMEM_PROMPT=v8
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
ALIYUN_KEY="sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

# --- 1) build 一次(方案A的默认检索也顺便产出) ---
OUTA="results/v8ab-A-s${SAMPLE}"
echo "===== [AB] build + 方案A检索 ($(date +%H:%M:%S)) ====="
rm -rf "$OUTA"
python3 src/adapters/run_nativemem.py --sample "$SAMPLE" \
  --output "$OUTA/sample${SAMPLE}_questions.json"
MEMDIR="$OUTA/memory_sample${SAMPLE}"

# --- 2) 方案B:复用同一记忆库，单模型检索+答题 ---
OUTB="results/v8ab-B-s${SAMPLE}"
echo "===== [AB] 方案B检索(复用记忆库，单模型) ($(date +%H:%M:%S)) ====="
rm -rf "$OUTB"; mkdir -p "$OUTB"
NATIVEMEM_V8_SINGLE=1 python3 src/adapters/run_nativemem.py --sample "$SAMPLE" \
  --memory-dir "$MEMDIR" \
  --output "$OUTB/sample${SAMPLE}_questions.json"

# --- 3) 两方案分别 answer+judge ---
for tag in A B; do
  OUT="results/v8ab-${tag}-s${SAMPLE}"
  echo "===== [AB] 方案${tag} judge ($(date +%H:%M:%S)) ====="
  python3 scripts/eval_full.py "$OUT" qwen3.6-flash "$ALIYUN_BASE" "$ALIYUN_KEY" "$OPENROUTER_API_KEY"
done
echo "===== [AB] 完成 ($(date +%H:%M:%S)) ====="
