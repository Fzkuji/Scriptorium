#!/bin/bash
# chunk size 扫描：固定方案B(单模型)+固定 sample0，只扫 NATIVEMEM_CHUNK_TURNS ∈ {6,10,15,20}。
# 目的：找覆盖率(overall分)开始下降之前的最大 chunk —— 覆盖率不掉的前提下 build 最省。
# 每个 chunk 值 build 一次库 + 方案B 检索答题 + judge。build 成本记在 _build_stats 里。
cd "$(dirname "$0")/.."

SAMPLE=0
export NATIVEMEM_PROMPT=v8
export NATIVEMEM_STORE_MODE=oneshot
export NATIVEMEM_V8_SINGLE=1
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
ALIYUN_KEY="sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

for CK in 6 10 15 20; do
  DIR="results/v8chunk-c${CK}"
  echo "===== [SWEEP] chunk=$CK : build + 方案B检索答题 ($(date +%H:%M:%S)) ====="
  rm -rf "$DIR"; mkdir -p "$DIR"
  NATIVEMEM_CHUNK_TURNS=$CK python3 src/adapters/run_nativemem.py --sample "$SAMPLE" \
    --output "$DIR/sample${SAMPLE}_questions.json"

  echo "===== [SWEEP] chunk=$CK : judge ($(date +%H:%M:%S)) ====="
  python3 scripts/eval_full.py "$DIR" qwen3.6-flash "$ALIYUN_BASE" "$ALIYUN_KEY" "$OPENROUTER_API_KEY"
done
echo "===== [SWEEP] 完成 ($(date +%H:%M:%S)) ====="
