#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export NATIVEMEM_PROMPT=v7
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
OUT="results/smoke-v7"
rm -rf "$OUT"
python src/adapters/run_nativemem.py \
  --sample 0 --max-sessions 2 --questions-limit 3 \
  --output "$OUT/result.json"
echo "=== 生成的记忆库结构 ==="
find "$OUT/memory_sample0" -type f | sort
echo "=== 抽查一个文件 ==="
find "$OUT/memory_sample0" -name '*.md' | head -1 | xargs cat
echo "=== 检索结果 ==="
cat "$OUT/result.json"
