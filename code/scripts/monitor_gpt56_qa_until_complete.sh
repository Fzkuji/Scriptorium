#!/usr/bin/env bash
set -u

REPO="/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
BUILD_ROOT="$REPO/results/formal/gpt56-chunk-curve-subscription-20260719-w32-only"
QA_BASE="$REPO/results/formal/gpt56-w32-screening-qa-20260719"
UPSTREAM="http://127.0.0.1:8206"
UPSTREAM_SHA="33fb02829090de6faf82aaba100d4a931dbee2a4e58898494118922ab519a66d"
MONITOR_DIR="$QA_BASE/monitor"
MONITOR_LOG="$MONITOR_DIR/watchdog.jsonl"
CHECKPOINTS=(60 300 600 1800 3600)

mkdir -p "$MONITOR_DIR" "$QA_BASE/retry-archives"

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

record_count() {
  find "$1" -type f -name record.json | wc -l | tr -d ' '
}

runner_pid() {
  local root="$1"
  ps -axo pid=,command= | awk -v output="$root" '
    index($0, "run_gpt56_chunk_curve_qa.py") &&
    index($0, "--output-dir " output) { print $1; exit }
  '
}

log_event() {
  local event="$1"
  local tier="$2"
  local detail="$3"
  local root="$QA_BASE/${tier}-r5"
  local pid
  pid="$(runner_pid "$root")"
  printf '{"timestamp":"%s","event":"%s","tier":"%s","records":%s,"completion":%s,"runner_pid":%s,"detail":"%s"}\n' \
    "$(timestamp)" "$event" "$tier" "$(record_count "$root")" \
    "$(test -f "$root/completion.json" && printf true || printf false)" \
    "${pid:-null}" "$detail" >> "$MONITOR_LOG"
}

archive_incomplete_questions() {
  local tier="$1"
  local root="$QA_BASE/${tier}-r5"
  local stamp archive question relative
  stamp="$(date -u +'%Y%m%dT%H%M%SZ')"
  archive="$QA_BASE/retry-archives/$tier/$stamp"
  while IFS= read -r question; do
    test -n "$question" || continue
    test -f "$question/record.json" && continue
    relative="${question#"$root"/}"
    mkdir -p "$archive/$(dirname "$relative")"
    mv "$question" "$archive/$relative"
    log_event "archived_incomplete_question" "$tier" "$relative"
  done < <(
    find "$root" -type f -name attempt_manifest.json -print \
      | sed -E 's#/attempt-[^/]+/attempt_manifest\.json$##' \
      | sort -u
  )
}

launch_runner() {
  local tier="$1"
  local root="$QA_BASE/${tier}-r5"
  local qa_run_id="gpt56-w32-screening-${tier}-r5-20260719"
  local run_ids=()
  if [[ "$tier" == "sol" ]]; then
    run_ids=(
      gpt56-locomo-conv-44-sol-w32
      gpt56-locomo-conv-48-sol-w32
      gpt56-longmemeval-s-2318644b-sol-w32
      gpt56-longmemeval-s-gpt4-6dc9b45b-sol-w32
      gpt56-beam-100k-100K-conv-1-sol-w32
      gpt56-beam-100k-100K-conv-2-sol-w32
    )
  else
    run_ids=(
      gpt56-locomo-conv-44-terra-w32
      gpt56-locomo-conv-48-terra-w32
      gpt56-longmemeval-s-2318644b-terra-w32
      gpt56-longmemeval-s-gpt4-6dc9b45b-terra-w32
      gpt56-beam-100k-100K-conv-1-terra-w32
      gpt56-beam-100k-100K-conv-2-terra-w32
    )
  fi
  (
    cd "$REPO" || exit 1
    env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      /opt/miniconda3/bin/python scripts/run_gpt56_chunk_curve_qa.py \
      --build-root "$BUILD_ROOT" \
      --output-dir "$root" \
      --run-ids "${run_ids[@]}" \
      --upstream "$UPSTREAM" \
      --upstream-code-sha256 "$UPSTREAM_SHA" \
      --python /opt/miniconda3/bin/python \
      --qa-run-id "$qa_run_id" \
      --execute --allow-model-requests \
      >> "$root/resume-watchdog.log" 2>&1
  ) &
  log_event "runner_started" "$tier" "watchdog_restart"
}

declare -A started_at
declare -A checkpoint_index
for tier in sol terra; do
  started_at[$tier]="$(date +%s)"
  checkpoint_index[$tier]=0
  log_event "monitor_started" "$tier" "existing_or_pending_runner"
done

while true; do
  all_complete=true
  for tier in sol terra; do
    root="$QA_BASE/${tier}-r5"
    if [[ -f "$root/completion.json" ]]; then
      continue
    fi
    all_complete=false
    pid="$(runner_pid "$root")"
    if [[ -z "$pid" ]]; then
      log_event "runner_stopped" "$tier" "completion_missing"
      archive_incomplete_questions "$tier"
      launch_runner "$tier"
      started_at[$tier]="$(date +%s)"
      checkpoint_index[$tier]=0
      continue
    fi

    now="$(date +%s)"
    elapsed=$((now - started_at[$tier]))
    index="${checkpoint_index[$tier]}"
    if (( index < ${#CHECKPOINTS[@]} && elapsed >= CHECKPOINTS[index] )); then
      log_event "checkpoint" "$tier" "elapsed_seconds=${CHECKPOINTS[index]}"
      checkpoint_index[$tier]=$((index + 1))
    fi
  done

  if [[ "$all_complete" == true ]]; then
    log_event "all_complete" "sol" "both_completion_files_present"
    log_event "all_complete" "terra" "both_completion_files_present"
    exit 0
  fi
  sleep 10
done
