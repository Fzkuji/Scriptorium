# R403 checkpoint task evaluation

`scripts/run_r403_task_eval.py` evaluates the immutable 10%, 25%, 50%, and
100% checkpoint memories produced by `scripts/run_r403_incremental_growth.py`.
It does not modify the chronological build runner or its checkpoint files.

## Frozen denominators

The denominator contract was written before the task-evaluation implementation
in `paper/refine-logs/R403_TASK_EVAL_PROTOCOL_FREEZE.json`.

- At 10%, 25%, and 50%, the task denominator contains exactly the upstream
  inventory rows with `growth_eligible=true`.
- At 100%, the task denominator contains all 1,540 LoCoMo category 1--4 rows
  with `main_qa_100_cohort=true`. This includes the seven source-recall
  exclusions.
- Old-fact retention uses `old_fact_cohort=true`; update accuracy uses
  `update_cohort=true`; source reachability uses
  `source_cohort_eligible=true`. The task runner copies these frozen labels and
  never derives them from model output.
- Each question result stores its answer, exact raw gold answer, category, and
  cohort memberships. These are inputs to the separate primary and sensitivity
  judges. Local LoCoMo F1 is diagnostic only.

## Input and future-evidence gates

Formal task execution accepts only a complete all-ten R403 build whose live
result passes `scripts/audit_r403_incremental_growth.py` twice with an unchanged
report. The source binding includes exact hashes for the run manifest, frozen
labels, every sample manifest, every checkpoint tree and memory, every
`question_inventory.jsonl`, every `task_score_binding.json`, and the independent
growth-audit report. A running R403 build lock, synthetic source, partial source,
changed source, or non-GPT-5.5 formal source is rejected.

At each checkpoint, the read-only source resolver contains only raw turns from
sessions at or before that checkpoint's `session_boundary`. The retriever can
access only that checkpoint's finalized memory directory. The task auditor
rejects any delivered `Dn:m` source identifier whose session is later than the
checkpoint boundary.

## Fixed retrieval and answer protocol

Every checkpoint uses `gpt-5.5` for retrieval and answering, the shared answer
prompt, the `dual_source` read-only tools, at most 12 retrieval rounds, and one
20,000-token visible-content gate counted with tiktoken 0.12.0 and
`o200k_base`. Per-question artifacts include answer and scoring inputs, build
and observed source reachability, M4 stage evidence, logical calls, physical
attempts, visible and provider tokens, tool/model latency, before/after memory
hashes, hash-chained retrieval and answer ledgers, and exclusive-proxy links.

Only complete per-question checkpoints can resume. An incomplete question
directory requires a new output root. The independent auditor does not import
the task runner; it reconstructs the checkpoint inventory, prefix resolver,
hashes, token accounting, ledgers, proxy events, M4 evidence, metrics, and
aggregates. Missing questions, extra questions, unfinished model calls,
unassigned proxy events, or stale artifacts fail the audit.

## No-network validation

Use `/opt/miniconda3/bin/python3`: this environment contains both
`tiktoken==0.12.0` for the frozen visible-token contract and `nltk==3.9.1` for
the diagnostic LoCoMo official F1. The BEAM-specific virtual environment does
not contain `nltk` and is not valid for this task-evaluation stage.

Create a fresh synthetic R403 build, then execute the task fixture:

```bash
/opt/miniconda3/bin/python3 scripts/run_r403_incremental_growth.py \
  --synthetic-sanity \
  --output-dir results/paper-experiments-20260714/r403-task-eval-source-synthetic-v1

/opt/miniconda3/bin/python3 scripts/run_r403_task_eval.py \
  --synthetic-sanity \
  --source-root results/paper-experiments-20260714/r403-task-eval-source-synthetic-v1 \
  --output-dir results/paper-experiments-20260714/r403-task-eval-synthetic-v5

/opt/miniconda3/bin/python3 scripts/audit_r403_task_eval.py \
  results/paper-experiments-20260714/r403-task-eval-synthetic-v5 \
  --report results/paper-experiments-20260714/r403-task-eval-synthetic-v5-reaudit.json
```

The fixture evaluates 1, 1, 2, and 4 questions at the four checkpoints,
respectively. It includes one update cohort, old-fact cohorts, and one 100%
source-recall exclusion. Its completion marker records zero model and network
requests.

## Formal preflight and execution

Preflight is local-only and currently remains blocked until the all-ten formal
growth artifact exists:

```bash
/opt/miniconda3/bin/python3 scripts/run_r403_task_eval.py \
  --preflight \
  --source-root results/gpt55-flex-benchmarks-20260714/r403-incremental-growth-gpt55-flex
```

Formal execution requires the explicit request gate and a validated GPT-5.5
Flex gateway result root. The loopback origin is derived from the root and an
exact provider window is audited. The task runner does not provide sample or
question limit flags:

```bash
/opt/miniconda3/bin/python3 scripts/run_r403_task_eval.py \
  --source-root results/gpt55-flex-benchmarks-20260714/r403-incremental-growth-gpt55-flex \
  --output-dir results/gpt55-flex-benchmarks-20260714/r403-checkpoint-task-eval-gpt55-flex \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --allow-model-requests

/opt/miniconda3/bin/python3 scripts/audit_r403_task_eval.py \
  results/gpt55-flex-benchmarks-20260714/r403-checkpoint-task-eval-gpt55-flex \
  --report results/gpt55-flex-benchmarks-20260714/r403-checkpoint-task-eval-gpt55-flex-reaudit.json
```

No formal request was made during implementation or validation.
