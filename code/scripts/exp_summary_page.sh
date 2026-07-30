#!/usr/bin/env bash
# SUMMARY_PAGE on/off 对比实验（机制验证版，小样本 + 调低 BIG_FILE 强制触发拆分）
# 唯一变量：NATIVEMEM_SUMMARY_PAGE=on|off。其余全同。
# 产物对比：拆出的子目录/文件结构、有无同名摘要页、3 问检索跳数与命中。
set -euo pipefail
cd "$(dirname "$0")/.."

export NATIVEMEM_PROMPT=v7
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
export NATIVEMEM_REBALANCE=on          # 开 rhythm2，才会走到 summary-page 分支
export NATIVEMEM_BIG_FILE="${NATIVEMEM_BIG_FILE:-4}"   # 调低阈值：单人文件超 4 条即超标 → 强制触发拆子目录
export NATIVEMEM_REORG_GROWTH=off      # 只测 rhythm2 的拆分，不叠加 rhythm3 增长触发

SAMPLE="${SAMPLE:-0}"
MAXS="${MAXS:-2}"
QLIM="${QLIM:-3}"

run_one () {
  local mode="$1"
  local out="results/exp-sumpage-$mode"
  echo "########## SUMMARY_PAGE=$mode ##########"
  rm -rf "$out"
  NATIVEMEM_SUMMARY_PAGE="$mode" python src/adapters/run_nativemem.py \
    --sample "$SAMPLE" --max-sessions "$MAXS" --questions-limit "$QLIM" \
    --output "$out/result.json"
  echo "=== [$mode] 记忆库结构（含子目录）==="
  find "$out/memory_sample$SAMPLE" -not -path '*/raw/*' \( -type f -o -type d \) | sort
  echo "=== [$mode] 检索跳数（steps）==="
  python -c "
import json
r=json.load(open('$out/result.json'))
for x in r[1:]:
    st=x.get('retrieval',{})
    print(f\"  {x['question_id']}: steps={st.get('steps')} k={st.get('k')} mem={len(x.get('memories',[]))}\")
"
}

run_one off
run_one on

echo
echo "########## 对比小结 ##########"
for mode in off on; do
  out="results/exp-sumpage-$mode/memory_sample$SAMPLE"
  # -mindepth 1 排除根目录本身；|| true 防止无子目录时管道在 pipefail 下退出 1
  ndir=$(find "$out" -mindepth 1 -type d -not -path '*/raw/*' 2>/dev/null | grep -vc '/raw$' || true)
  ndir=${ndir:-0}
  nmd=$(find "$out" -name '*.md' -not -path '*/raw/*' | wc -l | tr -d ' ')
  # 同名摘要页 = 存在一个 X.md 且同层有一个目录 X/
  sumpage=0
  while read -r d; do
    [ -z "$d" ] && continue
    base=$(basename "$d")
    parent=$(dirname "$d")
    [ -f "$parent/$base.md" ] && sumpage=$((sumpage+1))
  done < <(find "$out" -mindepth 1 -type d -not -path '*/raw/*' 2>/dev/null)
  echo "[$mode] 子目录数=$ndir  md文件数=$nmd  同名摘要页数=$sumpage"
done
