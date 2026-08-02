# R207 task evaluation

R207 task evaluation is a downstream, read-only comparison of the two frozen
path conditions produced by `run_r207_path_control.py`:

- `model_directed`;
- `deterministic_permutation`.

Formal execution accepts only an independently audited R207 artifact with all
ten LoCoMo samples, 1,540 category 1--4 questions, and 3,080 upstream
question-condition traces. The task runner cannot select fewer samples or
questions. Category 5 contains 446 questions and is excluded from this primary
run; any retained category-5 evaluation must use a separate artifact and score
summary.

The executable preregistration is
`paper/refine-logs/R207_TASK_EVAL_PREREGISTRATION.json`. It fixes the same
GPT-5.5 retriever, GPT-5.5 answerer, answer prompt, 12 retrieval rounds, Dn:m
source resolver, and 20,000-token visible gate for both path conditions. Gold
answers and R002 source labels are not model-visible.

Each question-condition directory contains:

- an input binding to the preregistration, R207 source, question, condition
  tree bytes/hash, upstream R207 trace, and derived stage provenance;
- an R004 visible-token trace and manifest;
- durable retrieval and answer ledgers plus fsynced request/response artifacts;
- the fixed-answer result;
- an M4 trace that retains the R207 pre-retrieval stages and adds observed
  retrieval reach and source-resolution fields;
- metrics for the first relevant file, mapped-source recall, navigation calls,
  visible/source-resolution tokens, before/after tree hash, and primary
  official LoCoMo F1;
- an independently reconstructed question audit and a no-clobber checkpoint.

Only complete checkpoints are resumable. The runner asks the independent
auditor to reconstruct every existing checkpoint before skipping it. A
partially written question directory is rejected and requires a new output
root. Formal calls require an exclusive controlled proxy, and the final audit
rejects missing or orphan proxy events and duplicate response IDs.

The local synthetic source and task-evaluation artifacts are:

- `results/paper-experiments-20260714/r207-task-eval-r207-source-synthetic`;
- `results/paper-experiments-20260714/r207-task-eval-r207-source-synthetic-audit.json`;
- `results/paper-experiments-20260714/r207-task-eval-synthetic-sanity`;
- `results/paper-experiments-20260714/r207-task-eval-synthetic-sanity-audit.json`.

The source audit has three canonical entries, both R207 path conditions, zero
question traces, zero model calls, and status `pass`. The downstream task
fixture adds one local question to each condition. Its independent audit has
two question-condition artifacts, six simulated retrieval calls, two simulated
answer calls, eight unique simulated response IDs, mapped-source recall 1.0 in
both conditions, official F1 1.0 in both conditions, unchanged memory hashes,
and zero real model or network requests. It is validation evidence only.

Reproduce the local checks with:

```bash
python scripts/run_r207_path_control.py \
  --synthetic-sanity \
  --output-dir results/paper-experiments-20260714/r207-task-eval-r207-source-synthetic-N
python scripts/audit_r207_path_control.py \
  --artifact-dir results/paper-experiments-20260714/r207-task-eval-r207-source-synthetic-N
python scripts/run_r207_task_eval.py --preregister
python scripts/run_r207_task_eval.py --preflight
python scripts/run_r207_task_eval.py \
  --synthetic-sanity \
  --r207-root results/paper-experiments-20260714/r207-task-eval-r207-source-synthetic-N \
  --output-dir results/paper-experiments-20260714/r207-task-eval-synthetic-sanity-N
python scripts/audit_r207_task_eval.py \
  results/paper-experiments-20260714/r207-task-eval-synthetic-sanity-N
```

The current formal preflight is `blocked`: the required all-ten source
`results/paper-experiments-20260714/r207-path-control-gpt55` does not exist.
The preflight records R002 subset status `passed`, `model_requests=0`, and
`network_requests=0`.

After an all-ten R207 artifact exists and passes its independent audit, the
formal command is:

```bash
/opt/miniconda3/bin/python3 scripts/run_r207_task_eval.py \
  --formal \
  --allow-model-requests \
  --r207-root results/paper-experiments-20260714/r207-path-control-gpt55 \
  --output-dir results/paper-experiments-20260714/r207-task-eval-gpt55 \
  --python /opt/miniconda3/bin/python3 \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT"
```

No formal R207 build or task-evaluation model request was made during this
implementation work.
