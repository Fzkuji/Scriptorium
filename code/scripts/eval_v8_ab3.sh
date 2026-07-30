#!/bin/bash
# 方案A(两模型:检索喂事件行+原文→独立answerer) vs 方案B(单模型:检索+直接答)
# 在 3 个样本(0/1/2)上对比，取合并总分，减少单样本波动。
# 每个样本 build 一次，A/B 复用同一记忆库。
cd "$(dirname "$0")/.."

export NATIVEMEM_PROMPT=v8
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
ALIYUN_KEY="sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

DIRA="results/v8ab3-A"; DIRB="results/v8ab3-B"
rm -rf "$DIRA" "$DIRB"; mkdir -p "$DIRA" "$DIRB"

for S in 0 1 2; do
  echo "===== [AB3] sample $S : build + 方案A检索 ($(date +%H:%M:%S)) ====="
  python3 src/adapters/run_nativemem.py --sample "$S" \
    --output "$DIRA/sample${S}_questions.json"
  MEMDIR="$DIRA/memory_sample${S}"

  echo "===== [AB3] sample $S : 方案B检索(复用记忆库) ($(date +%H:%M:%S)) ====="
  NATIVEMEM_V8_SINGLE=1 python3 src/adapters/run_nativemem.py --sample "$S" \
    --memory-dir "$MEMDIR" \
    --output "$DIRB/sample${S}_questions.json"
done

# 各方案合并 3 样本 judge（eval_full 自动合并 RUN_DIR 下所有 sample*_questions.json）
for tag in A B; do
  DIR="results/v8ab3-${tag}"
  echo "===== [AB3] 方案${tag} 合并3样本 judge ($(date +%H:%M:%S)) ====="
  python3 scripts/eval_full.py "$DIR" qwen3.6-flash "$ALIYUN_BASE" "$ALIYUN_KEY" "$OPENROUTER_API_KEY"
done
echo "===== [AB3] 完成 ($(date +%H:%M:%S)) ====="
