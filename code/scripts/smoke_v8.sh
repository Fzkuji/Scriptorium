#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export NATIVEMEM_PROMPT=v8
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
OUT="results/smoke-v8"
rm -rf "$OUT"
python src/adapters/run_nativemem.py --sample 0 --max-sessions 3 --questions-limit 3 \
  --output "$OUT/result.json"
echo "=== 记忆库结构 ==="
find "$OUT/memory_sample0" -type f | sort
echo "=== timeline 抽查 ==="
find "$OUT/memory_sample0/timeline" -name '*.md' | head -1 | xargs cat
echo "=== topics 抽查 ==="
find "$OUT/memory_sample0/topics" -name '*.md' | head -1 | xargs cat
echo "=== 检索结果(应含回原文的原始句子) ==="
cat "$OUT/result.json"
