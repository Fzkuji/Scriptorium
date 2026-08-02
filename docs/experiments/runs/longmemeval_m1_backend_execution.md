# LongMemEval-S M1 Mem0 / Graphiti execution runbook

This runbook covers backend construction and retrieval only. The resulting
`answer_input.json` files are consumed unchanged by the shared GPT-5.5 answer
stage.

No command in this document should be executed with the system Python. The
formal environment is `/opt/miniconda3/bin/python3` and the dependency contract
requires these exact versions:

- `mem0ai==2.0.10`
- `qdrant-client==1.18.0`
- `graphiti-core==0.29.2`
- `kuzu==0.11.3`
- `sentence-transformers==5.6.0`
- `tiktoken==0.12.0`
- local embedding model `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions
- builder requested and actual model `gpt-5.5`

Set the common paths from the repository root:

```bash
PY=/opt/miniconda3/bin/python3
DATASET=benchmarks/longmemeval/data/longmemeval_s_cleaned.json
BASE=results/gpt55-longmemeval-m1-baselines-20260714
PREREG=$BASE/formal_matrix_preregistration.json
```

Verify the environment without contacting a model or network service:

```bash
$PY - <<'PY'
from pathlib import Path
from scripts.longmemeval_m1_contract import strict_dependency_snapshot

print(strict_dependency_snapshot())
cache = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
if not cache.is_dir():
    raise SystemExit(f"missing local embedding cache: {cache}")
print(cache)
PY
```

The existing `backend_plans/*-smoke-item0000.json` files must not be executed:
their item-0 workspaces are identical to those in the formal plans. The formal
launcher rejects a smoke audit whose workspace overlaps any formal workspace.

Create a distinct one-item smoke plan for each backend. Replace `mem0` with
`graphiti` for the second backend:

```bash
METHOD=mem0
SMOKE_ROOT=results/gpt55-longmemeval-m1-$METHOD-smoke-20260714
SMOKE_PLAN=$SMOKE_ROOT/backend_plan.json

$PY scripts/run_longmemeval_m1_baselines.py plan-backend \
  --method "$METHOD" \
  --scope smoke \
  --item-index 0 \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --output-root "$SMOKE_ROOT" \
  --output "$SMOKE_PLAN"

$PY scripts/audit_longmemeval_m1_baselines.py plan "$SMOKE_PLAN" \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --output "$SMOKE_ROOT/backend_plan.audit.json"
```

The runner creates its own local exclusive proxy. Formal execution accepts a
validated OpenAI GPT-5.5 Flex gateway result root, not an arbitrary provider
URL. The gateway must already be active under an explicitly authorized durable
USD cap; its signed ready artifact supplies the loopback origin.

```bash
export OPENAI_GPT55_FLEX_GATEWAY_ROOT=/absolute/path/to/new-openai-flex-root

$PY scripts/run_longmemeval_m1_backends.py \
  --plan "$SMOKE_PLAN" \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --allow-model-requests

$PY scripts/audit_longmemeval_m1_backends.py \
  --plan "$SMOKE_PLAN" \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --output "$SMOKE_ROOT/$METHOD/audit.json"
```

The smoke audit must report `status=passed`, `items=1`, positive `model_calls`
and `network_calls`, `actual_builder_model=gpt-5.5`, and one unique workspace.
Only then launch the corresponding formal plan:

```bash
FORMAL_PLAN=$BASE/backend_plans/$METHOD-formal.json
SMOKE_AUDIT=$SMOKE_ROOT/$METHOD/audit.json

$PY scripts/run_longmemeval_m1_backends.py \
  --plan "$FORMAL_PLAN" \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --allow-model-requests \
  --smoke-audit "$SMOKE_AUDIT"

$PY scripts/audit_longmemeval_m1_backends.py \
  --plan "$FORMAL_PLAN" \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --output "$BASE/$METHOD/audit.json"
```

The recorded formal audit must be at `$BASE/$METHOD/audit.json`. The shared
answer runner recomputes the backend audit from the plan recorded in
`run_manifest.json`, compares it with this file, and binds the plan hash,
ordered workspace inventory, final workspace hashes, model ledgers, proxy
slices, source traces, retrieval records, and 20K budget traces.

Each item owns a separate Qdrant path and SQLite history database for Mem0, or
a separate Kuzu path and Graphiti group for Graphiti. Each item also records:

- `attempts.jsonl` with one start and one completion event;
- `model_evidence/ledger.jsonl` with a hash chain and fsync per event;
- exact request and response artifacts plus requested and actual model IDs;
- `proxy_slice.json`, binding all and only that item's calls to an append-only
  exclusive proxy prefix;
- `workspace_before.json` and checkpointed after-build/after-retrieval tree
  hashes;
- `source_trace.json` and `retrieval_raw.json` with stable dataset session IDs;
- `retrieval_budget.jsonl` and its independently hashed manifest;
- final `answer_input.json` with a hard 20,000-token visible total.

An unresolved attempt, failed model operation, mismatched actual model,
unrelated proxy event, pre-existing workspace, hash mismatch, symlink, or
hardlink causes rejection. A run with an unresolved item must use a new output
root and newly frozen plan; the executor never repeats that item in place.

No formal backend or model request is issued by the test command:

```bash
$PY -m pytest -q tests/test_longmemeval_m1_backends.py
$PY -m ruff check \
  scripts/longmemeval_m1_backend_contract.py \
  scripts/run_longmemeval_m1_backends.py \
  scripts/audit_longmemeval_m1_backends.py \
  tests/test_longmemeval_m1_backends.py \
  src/evaluation/visible_token_budget.py
```

After both backend audits exist, rerun the versioned shared-answer preflight:

```bash
$PY scripts/run_longmemeval_shared_answers.py preflight \
  --root "$BASE" \
  --dataset "$DATASET" \
  --preregistration "$PREREG" \
  --output "$BASE/shared_answer_preflight-after-backends-N.json"
```

Mem0 and Graphiti may enter formal answering only when this fresh report marks
them `ready_for_explicit_formal_answer_execution`. The source is dispatched to
`audit_longmemeval_m1_backends.audit_backend_run`; using the base input auditor
for either method is rejected.
