#!/bin/bash
# v8 单样本完整评测：build+检索 sample0 → answerer(阿里云) → judge(gpt-4o-mini)
# 目的：拿 v8 的真实 LLM-judge 分 + 耗时，判断值不值得上全 10 样本。
# 用法: bash scripts/eval_v8_one.sh [sample]   默认 sample0
cd "$(dirname "$0")/.."

SAMPLE="${1:-0}"
OUT="results/v8eval-s${SAMPLE}"
export NATIVEMEM_PROMPT=v8
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"

# --- 1) build + 检索(全部 session、全部问题) ---
echo "===== [eval] build+检索 sample $SAMPLE ($(date +%H:%M:%S)) ====="
rm -rf "$OUT"
python3 src/adapters/run_nativemem.py --sample "$SAMPLE" \
  --output "$OUT/sample${SAMPLE}_questions.json"
echo "[eval] build+检索完成 ($(date +%H:%M:%S))"

# --- 2) answerer(阿里云 qwen3.6-flash) + judge(gpt-4o-mini via OpenRouter) ---
# eval_full.py 合并 RUN_DIR 下所有 sample*_questions.json，这里只有一个样本
ALIYUN_KEY="sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
echo "===== [eval] answer + judge ($(date +%H:%M:%S)) ====="
python3 scripts/eval_full.py \
  "$OUT" \
  qwen3.6-flash \
  "$ALIYUN_BASE" \
  "$ALIYUN_KEY" \
  "$OPENROUTER_API_KEY"
echo "===== [eval] 完成 ($(date +%H:%M:%S)) ====="
