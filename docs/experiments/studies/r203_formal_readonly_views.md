# R203 formal read-only view/source controls

R203 evaluates the four registered read-only NativeMem conditions on the
LoCoMo category-1--4 primary set:

- `dual_source`: topics and timeline views, source resolver enabled;
- `topic_source`: topics view, source resolver enabled;
- `timeline_source`: timeline view, source resolver enabled;
- `dual_no_source`: topics and timeline views, source resolver absent.

The formal inventory is exactly 10 conversations, 1,540 questions per
condition, and 6,160 condition-question artifacts.  Source recall is reported
on the 1,533 R002-eligible questions in each condition.  Category 5 is not part
of R203.

## Local preflight

Preflight performs no model or network request:

```bash
/opt/miniconda3/bin/python3 scripts/run_r203_readonly_views.py \
  --preflight \
  --preflight-report \
  results/paper-experiments-20260714/r203-formal-preflight.json
```

It independently recomputes the LoCoMo part of R002, checks the raw dataset,
checks all code bindings, and accepts the NativeMem source only if
`scripts/audit_v88_gpt55_locomo.py` passes on a completed all-ten
`NativeMem-v8.8+calendar` GPT-5.5 Flex run.  It requires
`memory_sample0` through `memory_sample9`, all ten question outputs,
`questions_all.json`, `audit.json`, a closed Flex provider-evidence inventory,
and the exact current source hashes.

The current local source is incomplete and predates the required dynamic Flex
provider binding.  The generated preflight therefore reports concrete missing
or incompatible artifacts and remains `blocked`; it does not issue any model
request.

## Integrity-only preregistration revision

`paper/refine-logs/R203_PREREGISTRATION.json` is v2.  The conditions, samples,
comparisons, hypotheses, primary outcome, budgets, and denominators are
unchanged from v1.  The revision only clarifies stage provenance:

- the independently audited final NativeMem tree is the canonical input at the
  R203 boundary;
- R203 performs zero maintenance operations on each condition copy;
- this does not establish NativeMem builder pre-maintenance provenance.

The exact superseded v1 text remains at
`results/paper-experiments-20260714/code-integrity-superseded/R203_PREREGISTRATION.v1.json`
with SHA256
`fe71bd31beb58c1fed5b82d6990571e223482e8b8855973eb28bdd166a0f7c1d`.
The v2 preregistration, preflight, run manifest, per-question input bindings,
and auditor all retain this path and hash.

## Formal execution

Formal execution requires an explicit cost-authorized gateway root and does
not accept a provider base URL:

```bash
/opt/miniconda3/bin/python3 scripts/run_r203_readonly_views.py \
  --formal \
  --output-dir results/paper-experiments-20260714/r203-formal \
  --source-root results/gpt55-benchmarks-20260714/locomo-v88-calendar-gpt55 \
  --gateway-root /absolute/path/to/new-gpt55-flex-gateway-root \
  --preflight-report \
  results/paper-experiments-20260714/r203-formal-preflight.json \
  --python /opt/miniconda3/bin/python3 \
  --allow-model-requests
```

Before the first request, the runner creates all 40 condition memories and
verifies that the four copies for each sample have the same tree hash as the
audited source.  Retrieval and answering use the dynamic origin read from the
validated gateway root through an exclusive controlled proxy.  Each proxy
invocation captures a bounded Flex provider window, and the auditor checks
exact request-ID correspondence between the durable question ledgers, the
exclusive proxy log, and the provider request log.

The runner writes no-clobber input, stage-provenance, request, response,
ledger, result, question-audit, and checkpoint artifacts.  `--resume` accepts
only independently auditable complete checkpoints.  A partial question,
unresolved ledger operation, unsafe proxy invocation, provider-window error,
or orphan proxy event causes a failure before any additional request; a new
output root is then required.

## Independent audit

```bash
/opt/miniconda3/bin/python3 scripts/audit_r203_readonly_views.py \
  results/paper-experiments-20260714/r203-formal
```

The auditor does not import the runner.  It reconstructs the 1,540-question
inventory from the raw dataset and R002, reruns the NativeMem source audit,
recomputes every 20,000-token visible-content trace, checks source resolution,
rebuilds M4 stage evidence, verifies before/after tree hashes, checks all 6,160
checkpoints and aggregate outputs, and reconciles every model event with an
exclusive Flex gateway window.
