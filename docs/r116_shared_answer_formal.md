# R116 NativeMem shared-answer formal protocol

R116 has two formal executable modes. LoCoMo consumes the complete all-ten
`NativeMem-v8.8+calendar` main artifact. LongMemEval-S consumes the complete
500-item main artifact, including each item's immutable `checkpoint.json` and
memory tree. Neither mode rebuilds or modifies NativeMem memory.

The frozen preregistration is
`paper/refine-logs/R116_SHARED_ANSWER_PREREGISTRATION.json`. It binds both raw
datasets, the R002 evidence manifest/questions/audit, formal scope, source-run
locations, source auditors, the fixed GPT-5.5 retrieval and answer models, the
fixed answer prompt, a single 20,000-token visible-delivery gate, tiktoken
identity, retrieval rounds, answer reservation, durable ledgers, exclusive
proxy evidence, and implementation hashes.

## Source and recall policies

LoCoMo has 1,986 questions: 1,540 category 1--4 questions and 446 category 5
questions. Source recall uses the 1,533 R002-eligible primary questions and
compares gate-delivered `Dn:m` anchors with the frozen turn-level mapping.

LongMemEval-S has all 500 questions, including 30 abstention questions. R002
defines session-level gold IDs only. The runner reconstructs the released
session and turn order, maps each gate-delivered `Dn:m` anchor to the
corresponding `haystack_session_ids` entry, and computes recall against
`answer_session_ids`. It does not create a turn-level gold mapping.

Each question stores an immutable input binding, one `attempt-0001` directory,
the visible-token trace and manifest, retrieval request/response artifacts and
hash-chain ledger, answer ledger, raw result, independently derived source
recall, generic read-only audit, and a terminal checkpoint. Resume accepts
only a complete checkpoint that re-audits against the current frozen source.
An interrupted question requires a new output root, preventing an unrecorded
repeat request. A complete run also materializes hash-bound `results.json` and
`hypotheses.jsonl` files reconstructed from the per-question checkpoints.

Formal retrieval and answering use one exclusive controlled proxy per runner
invocation. Every logical call records its exact proxy event and append-only
log prefix. The independent auditor verifies requested and actual model,
response ID, provider usage, logical and physical attempts, request linkage,
prefix hashes, and the absence of orphan proxy events.

## Offline validation and preflight

These commands do not start a proxy and do not send model or network requests:

```bash
/opt/miniconda3/bin/python3 scripts/run_r116_formal.py --preregister
/opt/miniconda3/bin/python3 scripts/run_r116_formal.py --preflight
/opt/miniconda3/bin/python3 scripts/run_r116_formal.py \
  --synthetic-sanity \
  --output-dir results/paper-experiments-20260714/r116-formal-synthetic-sanity
/opt/miniconda3/bin/python3 scripts/audit_r116_formal.py \
  results/paper-experiments-20260714/r116-formal-synthetic-sanity
```

The persisted preflight report is
`results/paper-experiments-20260714/r116-formal-preflight.json`. It reports each
benchmark separately. A benchmark is ready only after its canonical main run
passes the existing independent auditor, its persisted audit and scoring input
match a live audit, and every memory/checkpoint descriptor is frozen.

## Formal launch commands

Run these only after the selected benchmark's preflight row is `ready` and a
validated OpenAI GPT-5.5 Flex gateway result root is active. The runner derives
the loopback origin from that root, acquires its consumer lock, and records an
exact provider window. It has no sample/question limit option.

```bash
/opt/miniconda3/bin/python3 scripts/run_r116_formal.py \
  --formal \
  --benchmark locomo \
  --output-dir results/paper-experiments-20260714/r116-locomo-shared-answer \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --python /opt/miniconda3/bin/python3 \
  --allow-model-requests

/opt/miniconda3/bin/python3 scripts/run_r116_formal.py \
  --formal \
  --benchmark longmemeval-s \
  --output-dir results/paper-experiments-20260714/r116-longmemeval-s-shared-answer \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --python /opt/miniconda3/bin/python3 \
  --allow-model-requests
```

Independent formal audits use:

```bash
/opt/miniconda3/bin/python3 scripts/audit_r116_formal.py \
  results/paper-experiments-20260714/r116-locomo-shared-answer
/opt/miniconda3/bin/python3 scripts/audit_r116_formal.py \
  results/paper-experiments-20260714/r116-longmemeval-s-shared-answer
```
