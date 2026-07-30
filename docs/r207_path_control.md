# R207 canonical-entry path control

`scripts/run_r207_path_control.py` installs an in-process hook around the
existing v8.8 adapter. After each final `distill_events` return and before
`write_events`, it copies the returned events into a canonical entry bank. It
also checks that the same list object, with the same serialized payload, is
passed to `write_events` exactly once. An unobserved write, a second write, an
unwritten non-empty batch, or a write exception prevents publication. The hook
is restored on both success and failure.

The extraction call remains the final v8.8+calendar call: it still receives
`known_topics`, and each extracted event still contains the model-selected
topic. Topic removal starts only after extraction, for the downstream path
experiment. `canonical_entry_bank.json` contains no topic field. Raw and
sanitized original topics are stored only in `original_placement.json`.

Each canonical entry contains `entry_id`, sample, session, chunk, ordinal,
`when`, `summary`, `summary_inline`, and naturally sorted `dia_ids`. The stable
ID hashes every field except placement. The reusable placement schema contains
only `{entry_id, topic_path}` plus file-level identity and metrics.

The two materialized conditions are `model_directed` and
`deterministic_permutation`. The control assigns the original path sequence to
entries with the cyclic offset that has the fewest fixed points. Consequently,
the conditions have exactly the same path multiset, per-topic counts, topic
count, maximum depth, and total UTF-8 path-string bytes. The manifest reports
the minimum fixed-point count and rate. Both conditions call only the same
deterministic `write_events`, timeline generation, and `dedup_topic_files`
operations. Model line, file, section, and consolidation maintenance is not
called.

The runner records the dataset and frozen evidence-mapping hashes,
selected-conversation hash, preregistered method-source hashes, hashes of the
code actually imported or executed by R207, final environment, model
configuration, bank and placement identities, and tree hashes. The standalone
LoCoMo, LongMemEval, and BEAM runners are informational
`environment_observations`; they are not part of the fingerprint. A
repository-level non-blocking lock permits one active R207 capture runner.
Publication is per-sample and atomic.

Each sample has an fsynced SHA-256 hash-chain `operations.jsonl`. Before a
completion request, the runner persists the exact model-visible request,
tokenizer identity, local visible-token count, logical call ID, and exclusive
proxy prefix. It persists the full response before the terminal ledger event.
The terminal event binds response ID, actual model, provider usage, latency,
logical client attempts, physical upstream attempts, unsupported parameters,
and the ending proxy prefix. Formal traffic always uses a per-launch managed
`controlled_gpt55_run_proxy.py` process.

Resume validates configuration, source provenance, every completed sample,
the ledger terminal state, call files, and artifact hashes. It can atomically
register a complete final or staging sample whose sample manifest was already
durable. An orphan model call, orphan operation, failed operation, incomplete
staging sample, or changed artifact requires a new output root; the runner
does not repeat the model request.

`question_traces.jsonl` preserves stable per-question, per-condition trace IDs
for later M4 assembly. It records independently observable evidence for frozen
mapping completeness, canonical entry/source presence, survival in the final
maintained build, and valid source-bearing condition paths. R207 does not run
retrieval or source resolution, so both stages are explicitly
`not_observed` with `value=null`; they are never defaulted to true.

The independent auditor recomputes source, dataset, and evidence-mapping
hashes, stable entry and trace IDs, every trace stage, capture-to-write pairing,
placement bijections, path sanitization, both path metrics, deterministic
permutation, ledger hashes, request/response files, token counts, costs, proxy
prefixes, artifact trees, and deterministic rematerialization. It rejects
unassigned proxy events, orphan or failed ledger records, symlinks, hardlinks,
Unicode/case path collisions, unsafe topic paths, unexpected files, and
incomplete manifests.

Synthetic validation does not make a model request:

```bash
python3 scripts/run_r207_path_control.py \
  --synthetic-sanity \
  --output-dir results/paper-experiments-20260714/r207-synthetic-sanity-repro-N

python3 scripts/audit_r207_path_control.py \
  --artifact-dir results/paper-experiments-20260714/r207-synthetic-sanity-repro-N
```

Formal all-conversation integration, when separately authorized, is:

```bash
python3 scripts/run_r207_path_control.py \
  --samples 0-9 \
  --model gpt-5.5 \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --request-concurrency 10 \
  --allow-model-requests \
  --output-dir results/paper-experiments-20260714/r207-path-control-gpt55

python3 scripts/audit_r207_path_control.py \
  --artifact-dir results/paper-experiments-20260714/r207-path-control-gpt55 \
  --report results/paper-experiments-20260714/r207-path-control-gpt55-audit.json
```

The implementation is formal-readiness complete but this work did not
authorize or issue formal GPT-5.5 requests. It does not implement the R301
organizer-by-retriever crossing and does not run retrieval, answering, judging,
or paired task statistics. It exposes the fixed bank, path-only placements,
durable model-call evidence, and fail-closed M4 trace inputs needed by those
later stages.
