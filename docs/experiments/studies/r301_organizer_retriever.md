# R301 fixed-entry organizer-by-retriever interaction

R301 implements the M5 experiment frozen in
`paper/refine-logs/EXPERIMENT_PLAN.md`. The executable preregistration is
`paper/refine-logs/R301_PREREGISTRATION.json`; it binds the final
`m5_interaction` section of
`paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json` at SHA-256
`b706535a61972c368547caf6611bac397959ae3fbfe71142adb8bfb95b449c22`.

The formal runner is `scripts/run_r301_organizer_retriever.py`. The independent
auditor is `scripts/audit_r301_organizer_retriever.py`. The auditor does not
import the runner and independently reconstructs source hashes, materialized
memory trees, prompt boundaries, visible-token traces, durable model-call
records, the exact matrix, scores, bootstrap interval, and decision.

## Frozen experimental unit

The formal input must be a completed, independently audited R207 artifact for
all ten LoCoMo conversations. R301 copies each canonical entry bank into its
own immutable input bundle. Every organizer condition receives the same entry
IDs, dates, summaries, inline summaries, and source IDs in the same order.

An organizer may return only an ordered list of `{entry_id, topic_path}`. The
runner rejects additions, omissions, duplication, reordering, unsafe paths,
more than 30 distinct topics, and paths deeper than four components. It then
uses `src.v8_memory.write_events` and `dedup_topic_files` to reproduce the same
deterministic materialization used by R207. No model performs file, section,
timeline, or deduplication maintenance.

The two organizer models are GPT-5.5 (`O55`) and GPT-4o-mini (`O4o`). Each
organizer has three separately recorded replicates. The two retrievers are
GPT-5.5 (`R55`) and GPT-4o-mini (`R4o`), producing the four cells `O55-R55`,
`O4o-R55`, `O55-R4o`, and `O4o-R4o`.

For every cell, replicate, conversation, and question, retrieval has two model
calls. Navigation can inspect only the gate-delivered path inventory. Selection
can inspect only the gate-delivered entries from opened paths. The runner
validates every selected path, entry ID, and source ID. Retriever output is a
trace and evidence selection, not an answer.

## Fixed answer boundary

All cells use the same fixed GPT-5.5 answerer and the shared
`controlled_locomo_answer_contract.ANSWER_PROMPT`. One 20,000-token
`tiktoken==0.12.0` `o200k_base` gate covers the retriever-visible inventory,
opened-path contents, selection trace, selected entries, and resolved source
text. Source resolution is a separate event kind. Prompt assembly receives
only `DeliveryResult.delivered_text`; gold answer and category fields are
loaded after the model response.

Each question artifact records ordered opened paths, selected entries, source
IDs, exact resolved source text, tool events, local token counts, provider
usage, latency, response IDs, requested and actual model identities, and
before/after memory-tree descriptors. It also records the R002 mapping hash and
the M4 stage trace IDs for canonical-source presence, maintenance survival,
path validity, retrieval reach, source resolution, and answer quality.

## Statistical decision

The primary outcome is fixed-answerer set F1. Replicates are averaged with
equal weight within question and cell. The point estimate is

```text
(O55-R55 - O4o-R55) - (O55-R4o - O4o-R4o)
```

The interval is a two-sided 95% percentile bootstrap with 10,000 resamples,
seed `20260714`, and LoCoMo conversation as the resampling unit. The
model-specific organization claim is supported only when both preregistered
strict inequalities hold (`O55-R55 > O4o-R55` and
`O4o-R4o > O55-R4o`) and the interaction interval lower bound is greater than
zero. Otherwise the claim is deleted and results are restricted to organizer
or retriever capacity effects. Synthetic results are always labelled
`synthetic_validation_only`.

## Durability and resume

Every organizer and question is a durable operation. Requests and responses
are fsynced before terminal ledger records. Commits bind artifact paths and
SHA-256 values. Resume accepts only committed artifacts whose hashes still
match; orphan, failed, incomplete, or modified state requires a new output
root.

Formal calls use separate exclusive proxies for GPT-5.5 and GPT-4o-mini. Proxy
events bind the logical call ID, requested and actual model, response ID,
usage, and exact log prefix. A completed launch has a hash-bound stop manifest.
On resume, the runner verifies an interrupted launch before terminating its
exact surviving process; it rejects PID reuse or inconsistent launch files.
API-key values are read from named environment variables and are not recorded.

## No-network validation

The synthetic command exercises three organizer replicates, all four cells,
the fixed answerer, the 20K gate, durable ledger, resume, analysis, and
independent audit. It authorizes no network requests:

```bash
/opt/miniconda3/bin/python3 scripts/run_r301_organizer_retriever.py \
  --synthetic-sanity \
  --output-dir results/paper-experiments-20260714/r301-synthetic-sanity
/opt/miniconda3/bin/python3 scripts/audit_r301_organizer_retriever.py \
  results/paper-experiments-20260714/r301-synthetic-sanity \
  --report results/paper-experiments-20260714/r301-synthetic-sanity-audit.json
```

## Formal command

Formal execution is rejected unless the sample set is exactly `0-9`, an R207
root is supplied, `--allow-model-requests` is present, and both provider gateway
roots are supplied. The GPT-5.5 loopback origin is derived from the validated
Flex gateway root and is bounded by an exact provider window. GPT-4o-mini uses
the separately marked OpenRouter gateway. A new output root is required after
any failed or orphaned durable operation.

```bash
export R301_OPENROUTER_GATEWAY_KEY=local-openrouter-gateway
export OPENROUTER_GATEWAY_BASE_URL="$(jq -r '.base_url' \
  "$OPENROUTER_GATEWAY_ROOT/gateway_ready.json")"
/opt/miniconda3/bin/python3 scripts/run_r301_organizer_retriever.py \
  --output-dir results/paper-experiments-20260714/r301-formal-r1 \
  --r207-root results/paper-experiments-20260714/r207-path-control-gpt55 \
  --samples 0-9 \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --gpt4o-upstream "$OPENROUTER_GATEWAY_BASE_URL" \
  --openrouter-gateway-root "$OPENROUTER_GATEWAY_ROOT" \
  --gpt4o-api-key-env R301_OPENROUTER_GATEWAY_KEY \
  --allow-model-requests
/opt/miniconda3/bin/python3 scripts/audit_r301_organizer_retriever.py \
  results/paper-experiments-20260714/r301-formal-r1 \
  --report results/paper-experiments-20260714/r301-formal-r1-audit.json
```

The formal matrix contains 60 organizer artifacts, 18,480 question-cell-
replicate artifacts, 55,500 successful model calls, and 18,480 visible-token
traces. These counts are invariant for the frozen 1,540-question all-ten input.
