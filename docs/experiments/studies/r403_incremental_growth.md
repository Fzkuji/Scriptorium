# R403 chronological incremental-growth experiment

`scripts/run_r403_incremental_growth.py` implements the M3 growth artifact for
LoCoMo. Each conversation has one continuing NativeMem v8.8+calendar state.
Sessions are processed once in chronological order. The runner never invokes
`build_memory(max_sessions=...)`, never reconstructs a prefix from an empty
directory, and never applies final maintenance to the continuing state.

The checkpoints are 10%, 25%, 50%, and 100% of chronological sessions. The
boundary is `ceil(total_sessions * percent / 100)`. At each boundary, the
runner copies the current state to a staging directory, applies the existing
v8.8 final maintenance only to that copy, writes the checkpoint inventory and
binding, and atomically publishes the checkpoint directory. The continuing
tree hash before and after every checkpoint operation must be identical.

## Exact build configuration

The continuing build uses fixed six-turn segments and the existing v8.8
functions for extraction, deterministic event writing, exact deduplication,
conditional topic consolidation, and per-session non-final maintenance. Empty
extraction results remain valid and are recorded as zero-event segment
operations, matching the existing v8.8 behavior. `NATIVEMEM_V9_PIPELINE` is
forbidden.

The hard source gate contains only code that this experiment imports or
executes: `src/nativemem.py`, `src/v8_memory.py`, the NativeMem adapter,
`src/chatgpt_proxy.py`, the durable ledger, visible-token counter, both proxy
wrappers, and the R403 runner and auditor. The three standalone benchmark
runners are recorded under
`environment_observations` but do not enter the run fingerprint. The BEAM r2
preflight repair changed its observed hash from the superseded
`b965ec83029ea4c32251abbde3ec94b20e6f1c01d22a95046f7f140adad277d9` to
`72e2f088690cfffc1dd354d754a53a70f596e0b4755f26b8554d2958a3cc087b`;
that unrelated change does not invalidate R403.

## Frozen labels and checkpoint cohorts

`frozen_labels.json` is produced from the raw LoCoMo questions, answers,
conversation turns, and frozen G0 evidence mapping before any output state is
built. A question enters the growth and source cohorts only when it is
source-recall eligible, has mapped anchors, and every mapped anchor belongs to
a session at or before the checkpoint. The seven primary source exclusions
never enter those cohorts. They remain in the 100% main-QA denominator.

An old-fact label at a checkpoint means that all final-gold anchors had already
appeared by the immediately preceding checkpoint. An update label requires
anchors in at least two sessions and one pre-registered update expression in
the raw question. The artifact also preserves normalized answer-substring
history from raw turns. These labels are not inferred from model outputs.

Each checkpoint stores `question_inventory.jsonl`. Its
`task_score_binding.json` binds the memory tree, inventory, frozen labels,
source mapping, and all cohort question-ID lists. It declares the later
controlled-answer result schema but keeps `task_results_status=not_attached`.
No task score is created by the build runner.

Every inventory row also has a stable checkpoint-specific `trace_id` and
`stage_evidence` for M4. The runner records mapping completeness from the
frozen mapping, canonical source presence from committed extraction-operation
results, maintenance survival and path validity from the actual finalized
checkpoint files. Retrieval and source resolution have not occurred at this
stage, so their status is `not_observed` and their value is `null`. A
source-recall exclusion is explicit `not_applicable`, not an inferred success
or failure. The auditor independently reconstructs all six stages from the
frozen inputs, ledger, and checkpoint tree and rejects any substituted boolean.

## Durable operation and model-call ledger

Every segment, exact-dedup pass, conditional consolidation, non-final session
maintenance pass, and finalized checkpoint copy has an `operation_started`
and `operation_committed` record in `operations.jsonl`. The JSONL file is an
fsynced SHA-256 hash chain. Each commit records input identity, continuing-tree
hashes, operation latency, result identity, and actual model-call cost.

Before each completion request, the observer fsyncs a request artifact and a
`model_call_started` record. After a response arrives, it fsyncs the complete
response artifact before recording the terminal event or returning the
response to the caller. The terminal record contains response ID, actual
model, provider-reported usage, local model-visible token count, latency,
logical client attempts, physical upstream attempts, unsupported parameters,
and the exact exclusive-proxy events and prefix cutoffs.

Formal traffic uses a new `controlled_gpt55_run_proxy.py` process for every
launch. The logical-call header connects request artifacts, response artifacts,
ledger events, and proxy events. The independent auditor verifies both prefix
hashes and byte offsets, response identity and usage, and that the exclusive
logs contain no unassigned events.

A process failure after an operation or model-call start but before its
terminal record leaves an orphan. Resume rejects that output instead of
resending the request. A safe SIGINT/SIGTERM is honored after the active
operation commits. Resume accepts only a contiguous committed schedule prefix,
verifies the current continuing tree and committed checkpoint trees, and then
starts the first uncommitted operation. A sample manifest that was durably
written immediately before a process failure can be verified and linked into
the run manifest without repeating model calls.

## Independent audit

`scripts/audit_r403_incremental_growth.py` does not import the runner. It
independently recomputes raw-input labels, session boundaries, fixed-six
segment identities, operation order, hash chains, request and response hashes,
local token counts, proxy linkage, per-operation and cumulative cost,
checkpoint memory trees, source reachability, inventories, cohort bindings,
and the 100% main-QA denominator. It rejects missing or extra artifacts,
symlinks, hardlinks, special nodes, Unicode/case path collisions, source or
configuration drift, orphan records, failed operations, modified checkpoint
copies, and unassigned proxy events.

The no-network validation command is:

```bash
python3 scripts/run_r403_incremental_growth.py \
  --synthetic-sanity \
  --output-dir results/paper-experiments-20260714/r403-synthetic-sanity-repro-N
python3 scripts/audit_r403_incremental_growth.py \
  results/paper-experiments-20260714/r403-synthetic-sanity-repro-N \
  --report results/paper-experiments-20260714/r403-synthetic-sanity-repro-N-audit.json
```

The formal command is intentionally gated by `--allow-model-requests` and
requires all ten conversations:

```bash
python3 scripts/run_r403_incremental_growth.py \
  --samples 0-9 \
  --model gpt-5.5 \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --allow-model-requests \
  --output-dir results/paper-experiments-20260714/r403-incremental-growth-gpt55
python3 scripts/audit_r403_incremental_growth.py \
  results/paper-experiments-20260714/r403-incremental-growth-gpt55 \
  --report results/paper-experiments-20260714/r403-incremental-growth-gpt55-audit.json
```

The implementation and synthetic validation do not authorize or execute the
formal GPT-5.5 build, controlled answering, judging, or paper statistics.
