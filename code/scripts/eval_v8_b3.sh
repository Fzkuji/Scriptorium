#!/bin/bash
# 方案B(单模型)3样本确认跑:v8自组织升级(地图+整理+结构感知)后的多样本分。
# 每样本 build 一次 + 方案B检索答题,合并 judge。
cd "$(dirname "$0")/.."

export NATIVEMEM_PROMPT=v8
export NATIVEMEM_STORE_MODE=oneshot
export NATIVEMEM_V8_SINGLE=1
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
ALIYUN_KEY="sk-sp-D.LIEPX.T6GA.MEYCIQCmomYgcDaqBtAWGXYyua8ve/6RPImY/+0W2BuPqNXzZQIhAJwxunRW89LTfdar0B2A59wjemXSIAHYOPGBgIluBUCh"
ALIYUN_BASE="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

DIR="${B3_DIR:-results/v8selforg-b3}"
rm -rf "$DIR"; mkdir -p "$DIR"

# 样本并行：各样本 build+答题各起一个后台进程写各自日志，wait 后合并 judge。
# 样本列表 B3_SAMPLES（默认 "0 1"，不再含 2）。
SAMPLES="${B3_SAMPLES:-0 1}"
for S in $SAMPLES; do
  echo "===== [B3] sample $S : build + 方案B 后台启动 ($(date +%H:%M:%S)) ====="
  python3 src/adapters/run_nativemem.py --sample "$S" \
    --output "$DIR/sample${S}_questions.json" > "$DIR/sample${S}.log" 2>&1 &
done
wait
for S in $SAMPLES; do
  echo "----- [B3] sample $S 关键行 -----"
  grep -E "build done|wrote .* question records|SKIPPED" "$DIR/sample${S}.log" || true
done

echo "===== [B3] 合并样本 judge ($(date +%H:%M:%S)) ====="
python3 scripts/eval_full.py "$DIR" qwen3.6-flash "$ALIYUN_BASE" "$ALIYUN_KEY" "$OPENROUTER_API_KEY"
echo "===== [B3] 完成 ($(date +%H:%M:%S)) ====="
