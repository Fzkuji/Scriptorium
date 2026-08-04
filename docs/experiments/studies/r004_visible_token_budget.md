# R004 visible-token budget gate

This implementation provides the G0.2 accounting boundary for controlled
retrieval experiments. It is separate from the active GPT-5.5 benchmark
runners and does not change the frozen NativeMem implementation.

## Accounting boundary

The budget counts rendered `tool_result` and `source_resolution` text that is
actually returned by `VisibleTokenBudgetGate` for inclusion in the model
context. Raw tool text is retained in the durable trace for audit, but callers
must use only `DeliveryResult.delivered_text` at the model boundary. A `None`
value means that no content may be delivered.

The gate records, per event:

- exact raw and delivered UTF-8 text, SHA-256, byte count, and local token
  count;
- event kind, tool/source metadata, decision, configured and remaining budget,
  cumulative visible tokens, and cumulative source-resolution tokens;
- deterministic truncation or rejection, exhaustion state, and the required
  finalizer state.

The finalizer records the before/after memory descriptors and hashes, whether
memory changed, and separately supplied provider usage. Provider prompt and
completion usage is not added to the visible retrieval budget.

Each JSONL record is hash chained and fsynced. Finalization writes a separate
manifest containing the complete trace hash and final record hash. The auditor
reparses the original raw/delivered text, reconstructs the truncation decision,
recomputes token counts and cumulative state, validates the hash chain and
manifest, and optionally compares preserved before/after memory paths.

The trace and manifest detect accidental or independent modification. They do
not provide authenticity against an actor who can rewrite both artifacts and
recompute all hashes; signed or externally retained manifest hashes are needed
for that threat model.

## Tokenizer identity

`TokenCounter.resolve()` first calls `tiktoken.encoding_for_model()`. If the
model is not in tiktoken's mapping, it uses the explicitly configured named
encoding, by default `o200k_base`, and records that fallback. The trace stores
the requested model, tiktoken version, encoding name, resolution method, and
fallback reason.

If tiktoken is not installed, execution fails unless the caller explicitly
enables the UTF-8 byte fallback. The byte fallback is deterministic and useful
for the sanity test, but its unit is bytes rather than provider tokens.

Every tokenizer identity has `provider_exact: false`. Neither tiktoken counts
nor the byte fallback may be reported as an exact provider token count. A
formal comparison can claim only a hard cap under the exact tokenizer identity
stored in its trace.

## Minimal sanity command

Run from the repository root with a new output path:

```bash
python3 scripts/token_budget/run_visible_token_budget_sanity.py \
  --output-dir /private/tmp/nativemem-r004-sanity-001 \
  --budget-tokens 32 \
  --model gpt-5.5 \
  --allow-byte-fallback
```

The command performs no model request. It creates accepted, truncated, and
post-exhaustion rejected events, changes a small synthetic memory state, writes
the trace and manifest, and invokes the independent auditor. The process exits
nonzero if the audit fails.

The saved trace can be audited again with:

```bash
python3 scripts/token_budget/audit_visible_token_budget.py \
  --trace /private/tmp/nativemem-r004-sanity-001/visible-token-trace.jsonl \
  --memory-before /private/tmp/nativemem-r004-sanity-001/memory-before \
  --memory-after /private/tmp/nativemem-r004-sanity-001/memory-after \
  --report /private/tmp/nativemem-r004-sanity-001/audit-second.json
```

## Integration interface

```python
from src.evaluation.visible_token_budget import (
    TokenCounter,
    VisibleTokenBudgetGate,
    snapshot_memory_path,
)

tokenizer = TokenCounter.resolve(
    requested_model="gpt-5.5",
    fallback_encoding="o200k_base",
)
gate = VisibleTokenBudgetGate(
    trace_path=trace_path,
    run_id=question_id,
    configured_budget_tokens=20000,
    tokenizer=tokenizer,
    memory_before=snapshot_memory_path(memory_path),
    overflow_policy="truncate",
)

result = gate.deliver_tool_result(
    event_id=tool_call_id,
    raw_text=rendered_tool_result,
    tool_name="read_memory",
    tool_call_id=tool_call_id,
)
if result.may_deliver:
    model_context.append(result.delivered_text)
if result.exhausted:
    stop_retrieval_and_finalize = True

gate.finalize(
    memory_after=snapshot_memory_path(memory_path),
    actual_model_usage=provider_usage_record,
)
```

Source expansion uses `deliver_source_resolution()` instead of
`deliver_tool_result()`, so its delivered tokens are included both in total
visible tokens and the source-resolution subtotal.

## Required integration work

The generic gate does not yet intercept any existing model call. Each new
controlled baseline or mechanism runner must place it at the final prompt
assembly boundary and must demonstrate that no alternate raw-result path can
reach the model. The integration must also:

- create one trace per independently budgeted question/retrieval;
- deliver no further content after `result.exhausted`; a finite candidate list
  may still submit later candidates so the trace records them as
  `rejected_budget_exhausted`, then execute the finalizer;
- pass real proxy/SDK model identity, response ID, logical/physical attempts,
  and provider usage when available;
- call `deliver_source_resolution()` after resolving source identifiers into
  text;
- preserve stable before/after memory snapshots if live-path verification is
  required later;
- run `audit_visible_token_budget.py` before treating a result as complete.

The active benchmark runners, `src/nativemem.py`, `src/v8_memory.py`, the
NativeMem adapter, and the shared GPT-5.5 proxy remain unchanged by R004.

For the controlled baseline launcher, the current adapter stage ends with a
per-question `memories` list and does not invoke the answerer. The gate belongs
in the later shared-answerer stage, after that list has been rendered with one
frozen serialization and immediately before prompt assembly. Adapter-internal
build/retrieval model usage remains provider usage; it is not included in the
visible evidence budget. Methods that can resolve stable source identifiers
must pass the additional rendered source text through
`deliver_source_resolution()`.

The formal controlled LoCoMo protocol uses 20,000 local
`tiktoken==0.12.0` `o200k_base` tokens for BM25, Mem0, Graphiti OSS, and the
planned controlled NativeMem adapter. Full-context is a separate
`full_context_unbounded_accounted` row: it measures every item, rejects any
truncation, and does not claim a matched cap. The earlier 6,000-token paper
claim is not part of the controlled protocol.

For R203, each condition and question needs its own trace. Every read-tool
result and source-resolution result crosses the gate separately. All four
conditions must share the same tokenizer identity, budget, and rendering. The
read-only experiment auditor must additionally require equal before/after
memory hashes. The dual-view/no-source condition must have a zero
source-resolution subtotal.

For R301, the gate belongs between the normalized evidence bundle and the fixed
GPT-5.5 answerer. It does not apply to the organizer prompt. Retriever traces
and entry IDs are rendered as tool results, while resolved source text is
recorded separately. The run ID must identify organizer/retriever cell,
organizer replicate, conversation, and question; all cells use the same
tokenizer, budget, and evidence serialization.

This boundary is implemented in `scripts/run_r301_organizer_retriever.py` and
independently reconstructed by
`scripts/audit_r301_organizer_retriever.py`. The no-network matrix validates all
four cells and three organizer replicates without authorizing external model
requests. Formal R301 execution remains gated on an independently audited
all-ten R207 canonical-bank artifact and `--allow-model-requests`.
