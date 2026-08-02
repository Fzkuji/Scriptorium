# Controlled LoCoMo answer protocol

This protocol applies one shared GPT-5.5 answerer after retrieval and enforces
R004 at the final memory-to-prompt boundary. It does not modify the frozen
NativeMem or benchmark runners.

## Formal rows

The preregistered baseline matrix contains `full_context`, `bm25`, `mem0`, and
`zep`. The `zep` row is the local Graphiti OSS adapter and is not described as
the commercial Zep platform.

- `bm25`, `mem0`, and `zep` use a hard total visible budget of 20,000 local
  `tiktoken==0.12.0` `o200k_base` tokens per question.
- `full_context` uses `full_context_unbounded_accounted`. Every memory is
  measured by the same tokenizer, but the row does not claim a matched cap.
  Any truncation rejects the run as a full-context result.
- The CLI requires an explicit `--budget-tokens 20000` or
  `--budget-tokens unbounded`; there is no implicit formal default.
- The former 6,000-token claim is not part of this protocol.

The context check is local-token accounting, not provider-exact accounting.
The caller must preregister `--model-context-limit-tokens`. Before each model
request, the runner requires:

```
local_prompt_tokens + answer_completion_reservation <= model_context_limit
```

After the response, the runner also requires provider-reported `total_tokens`
to remain within the same declared context limit and records both checks.

## Formal runbook

Use this order for the formal run:

1. Run `audit_gpt55_locomo_baselines.py` independently on `full_context`,
   `bm25`, `mem0`, and `zep`. All four reports must be `passed`, must identify
   their expected method, and must report 1,986 questions before the protocol
   freezer is called.
2. Select and record the actual provider context limit. On the currently
   assembled full-context input, the read-only local preflight with a 512-token
   completion reservation reports a maximum of 24,458 tokens (`s4_q217`:
   23,946 prompt tokens plus 512). Recompute this bound if any input, prompt,
   tokenizer, or reservation hash changes. The declared context limit must be
   at least this value and must not exceed the provider's actual limit.
3. Run `freeze_controlled_answer_protocol.py` once. Record both its file
   SHA-256 and `protocol_content_sha256`; do not edit or replace the frozen
   file.
4. Use a validated, cost-authorized OpenAI Flex gateway root. Run one formal
   row at a time with a new output directory. Each row acquires the gateway
   consumer lock and records an exact provider window. Use the same exact CLI
   to resume an interrupted row; do not delete failed question attempts or
   proxy invocation artifacts.
5. Run `audit_controlled_locomo_answers.py` on the completed row before any
   answer or score is used. The immutable `complete.json` is created only after
   the pre-final independent audit passes, then the auditor verifies it again.

The current implementation intentionally cannot create the controlled-answer
preregistration while either `mem0` or `zep` formal input is absent or fails its
baseline audit. No formal answer request belongs before that freeze.

The frozen GPT-5.5 proxy currently reports `max_output_tokens` as unsupported
and retries without it. The requested answer limit, unsupported-parameter
record, actual usage, client HTTP attempts, and inner-proxy upstream HTTP
attempts are preserved. The protocol does not claim an enforced provider-side
output limit.

## Prompt boundary

`render_memories_individually()` reproduces the current `format_memories`
order: stable chronological ordering when any date exists, otherwise input
order. Each rendered memory is submitted separately to
`VisibleTokenBudgetGate.deliver_tool_result()`.

`assemble_answer_prompt()` accepts only a question string and a sequence of
`DeliveryResult` objects. The memory block is constructed only from
`DeliveryResult.delivered_text`. It cannot accept a dataset row, raw memory
text, gold, evidence, or category. The independent auditor reconstructs every
raw rendering, every gate decision, the delivered memory block, and the final
prompt hash.

Ordinary baseline rows never perform source expansion. Their required
`cumulative_source_resolution_tokens` value is zero.

## Category-5 labels

For category-5 rows without a raw `answer`, the `gold` field in the current
retrieval-only baseline input is the raw dataset's `adversarial_answer`; it is
a distractor and is not a correct answer. Two current rows have an explicit raw
`answer`, and their legacy input `gold` contains that answer. The answer runner
does not pass either form of this field to the model.

After model completion, the runner reconstructs scoring labels from raw
LoCoMo by `question_id`:

- use raw `answer` when the QA row explicitly provides it;
- otherwise, for category 5, use `Not mentioned in the conversation`;
- preserve `adversarial_answer` separately as `adversarial_distractor`.

The auditor requires 1,540 category-1--4 rows and 446 separately identified
category-5 rows.

## Freeze and execute

Do not freeze the protocol until all four retrieval inputs pass
`audit_gpt55_locomo_baselines.py`. A freeze command is:

```bash
/opt/miniconda3/bin/python3 scripts/freeze_controlled_answer_protocol.py \
  --output results/gpt55-locomo-baselines-20260714/controlled_answer_protocol_preregistration.json \
  --full-context-input-dir results/gpt55-locomo-baselines-20260714/full_context \
  --bm25-input-dir results/gpt55-locomo-baselines-20260714/bm25 \
  --mem0-input-dir results/gpt55-locomo-baselines-20260714/mem0 \
  --zep-input-dir results/gpt55-locomo-baselines-20260714/zep \
  --model-context-limit-tokens CONTEXT_LIMIT
```

The generator refuses stale, incomplete, smoke-scope, noncanonical, or
method-mismatched inputs. It atomically creates a no-clobber artifact containing
all input hashes, the formal matrix, tokenizer identity, prompt/gate source
hashes, label policy, and context policy. The caller must record the resulting
file SHA-256 and pass it explicitly to each answer run.

A capped row uses:

```bash
/opt/miniconda3/bin/python3 scripts/run_controlled_locomo_answers.py \
  --method bm25 \
  --input-dir results/gpt55-locomo-baselines-20260714/bm25 \
  --output-dir results/gpt55-locomo-baselines-20260714/bm25-controlled-answers \
  --preregistration PATH_TO_FROZEN_PROTOCOL \
  --preregistration-sha256 FROZEN_FILE_SHA256 \
  --budget-policy hard_cap \
  --budget-tokens 20000 \
  --model-context-limit-tokens CONTEXT_LIMIT \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --python /opt/miniconda3/bin/python3 \
  --allow-model-requests
```

The full-context row uses:

```bash
/opt/miniconda3/bin/python3 scripts/run_controlled_locomo_answers.py \
  --method full_context \
  --input-dir results/gpt55-locomo-baselines-20260714/full_context \
  --output-dir results/gpt55-locomo-baselines-20260714/full-context-controlled-answers \
  --preregistration PATH_TO_FROZEN_PROTOCOL \
  --preregistration-sha256 FROZEN_FILE_SHA256 \
  --budget-policy full_context_unbounded_accounted \
  --budget-tokens unbounded \
  --model-context-limit-tokens CONTEXT_LIMIT \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --python /opt/miniconda3/bin/python3 \
  --allow-model-requests
```

Each completed question has a gate trace, gate manifest, hash-chained question
ledger, prompt hash, response/model ID, usage, logical-call count, client HTTP
attempt count, inner-proxy upstream HTTP attempt count, and exact exclusive
proxy event IDs. Resume re-audits a completed question, including its exclusive
proxy entries, before skipping it.

Every exclusive proxy invocation publishes an immutable `start.json` after its
health check and a no-clobber `manifest.json` when it stops. If the answer
process is interrupted after `start.json` is published, the next invocation
verifies the run/PID/path linkage, terminates only an exactly matching surviving
process group, fingerprints the preserved artifacts, and publishes an
`interrupted_recovered` manifest before resume auditing. The auditor rejects an
invocation directory without a completed manifest or with different start,
ready, request-log, or process-log identities.

`controlled_gpt55_run_proxy.py` reuses the transport, health, fsync, and atomic
ready-file primitives from `gpt55_run_proxy.py`. It adds question-level fields
without changing the retrieval baseline wrapper, whose byte hash is already
part of completed baseline input manifests. Both wrapper hashes are recorded
and independently audited.

The final independent audit is:

```bash
/opt/miniconda3/bin/python3 scripts/audit_controlled_locomo_answers.py \
  results/gpt55-locomo-baselines-20260714/bm25-controlled-answers
```

The no-network synthetic check is:

```bash
/opt/miniconda3/bin/python3 scripts/run_controlled_locomo_answer_sanity.py \
  --output-dir /private/tmp/controlled-locomo-answer-sanity-001 \
  --budget-tokens 16
```

It uses a fake client and fake exclusive proxy ledger. Canary checks verify
that gold, evidence, category, overflow-only raw text, and post-exhaustion raw
text do not occur in the model prompt.

## Adapter status

The frozen protocol declares the same 20,000-token hard-cap contract for
NativeMem, R203, and R301. Their separate upstream adapters are implemented and
covered by no-network fixtures plus independent auditors:

- R116 renders canonical frozen NativeMem entries without modifying the source
  memory artifact.
- R203 records one trace per condition and question, keeps read-tool and
  source-resolution events separate, verifies unchanged memory hashes, and
  retains a zero source-resolution subtotal for dual-view/no-source.
- R301 renders the retriever selection trace and selected entries as tool
  results, then resolved source text as source-resolution events. Organizer
  prompts are outside the answer budget. Run IDs bind organizer, retriever,
  replicate, conversation, and question.

All three use a boundary that passes only `DeliveryResult.delivered_text` into
`ANSWER_PROMPT`; gold answers and category labels are loaded only after the
model response. Formal artifacts remain pending.
