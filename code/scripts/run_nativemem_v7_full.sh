#!/bin/bash
# NativeMem v7（agent 自组织 + 三节奏）跑全 10 个 LoCoMo 样本。
# 每个样本：fresh build（模型用 bash 自组织记忆库）+ 检索，输出 per-sample questions.json。
# 用当前默认策略（真实阈值，不人为压低），先看生成效果，judge 打分后续再说。
# 用法: bash scripts/run_nativemem_v7_full.sh [tag]
#   tag 默认 qwen36flash；模型固定 qwen3.6-flash（阿里云，key 硬编码在 nativemem.py）
# 无 set -e：单样本失败（偶发 API 502）不能拖垮整轮；失败重试后仍坏则跳过。
cd "$(dirname "$0")/.."

TAG="${1:-qwen36flash}"
OUTDIR="results/v7full-${TAG}-locomo"
mkdir -p "$OUTDIR"

export NATIVEMEM_PROMPT=v7
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
export NATIVEMEM_REBALANCE=on          # rhythm2 每 session 末轻量再平衡
export NATIVEMEM_SUMMARY_PAGE=off       # 消融结论：弱模型不留摘要页更好
# 其余用代码默认真实阈值：BIG_FILE=150 / SMALL=8 / WIDE=20 / REORG_GROWTH=2.0 / FILE_DELTA=8
# （不人为压低，让文件自然长到超标才拆，才是真实评测形态）

echo "===== [v7full] 开始，输出 -> $OUTDIR （$(date +%H:%M:%S)）====="
for s in 0 1 2 3 4 5 6 7 8 9; do
  out="$OUTDIR/sample${s}_questions.json"
  if [ -f "$out" ]; then
    echo "[v7full] sample $s 已完成，跳过"
    continue
  fi
  echo "===== [v7full] build+retrieve sample $s ($(date +%H:%M:%S)) ====="
  for attempt in 1 2; do
    if python3 src/adapters/run_nativemem.py --sample "$s" --output "$out"; then
      break
    fi
    echo "[v7full] sample $s attempt $attempt 失败；$([ $attempt -lt 2 ] && echo 重试 || echo 放弃)"
    rm -rf "$OUTDIR/memory_sample${s}"  # 清掉半成品再重试
    sleep 10
  done
done
echo "===== [v7full] 全 10 样本完成（$(date +%H:%M:%S)）====="
echo "=== 每样本记忆库文件数 ==="
for s in 0 1 2 3 4 5 6 7 8 9; do
  d="$OUTDIR/memory_sample${s}"
  [ -d "$d" ] || continue
  nmd=$(find "$d" -name '*.md' -not -path '*/raw/*' 2>/dev/null | wc -l | tr -d ' ')
  ndir=$(find "$d" -mindepth 1 -type d -not -path '*/raw/*' 2>/dev/null | wc -l | tr -d ' ')
  echo "  sample $s: ${nmd} md 文件, ${ndir} 子目录"
done
