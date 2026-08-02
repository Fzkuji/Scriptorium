# OpenRouter GPT-4o-mini cost gateway

The LongMemEval primary judge and the O4o/R4o R301 cells must use the local
OpenRouter gateway in `src/openrouter_gpt4o_mini_gateway.py`. Direct formal
requests to `https://openrouter.ai/api/v1` are rejected by the R301 runner,
the LongMemEval scorer, and the BEAM evaluator. LoCoMo is excluded from this
gateway protocol and is locked to `scripts/eval_full.py`.

The local API accepts only `openai/gpt-4o-mini`. The provider payload is always
pinned to `openai/gpt-4o-mini-2024-07-18`, is non-streaming, and is sent only to
`https://openrouter.ai/api/v1/chat/completions`. A read-only request to
OpenRouter's `/api/v1/models` endpoint on 2026-07-14 confirmed that this exact
model ID exists. Runtime model-directory queries are not part of the gateway.
Top-level `models`, `route`, `provider`, and `fallbacks` inputs are rejected,
and the provider payload sets `provider.allow_fallbacks=false`. This prevents
OpenRouter cross-model fallback while leaving ordinary tool payloads intact.

## Start one shared gateway

Set the cap only after the user authorizes an exact USD maximum. The example
uses variables so it does not imply that USD 250, or any other value, has been
authorized.

```bash
export OPENROUTER_MAX_COST_USD='<authorized maximum>'
export OPENROUTER_GATEWAY_ROOT="$PWD/results/paper-experiments-20260714/openrouter-gateway-v1"

/opt/miniconda3/bin/python src/openrouter_gpt4o_mini_gateway.py \
  --result-root "$OPENROUTER_GATEWAY_ROOT" \
  --max-cost-usd "$OPENROUTER_MAX_COST_USD" \
  --host 127.0.0.1 \
  --port 0 \
  --api-key-env OPENROUTER_API_KEY
```

In the consumer shell, read the ephemeral marked origin from the ready
artifact instead of entering a port manually:

```bash
export OPENROUTER_GATEWAY_BASE_URL="$(jq -r '.base_url' \
  "$OPENROUTER_GATEWAY_ROOT/gateway_ready.json")"
```

The result root is exclusive to this gateway. It must not be a subscription
result root, an OpenAI Flex result root, or a benchmark output root. All formal
OpenRouter consumers should share this one root and cap. Their artifacts bind
the root marker and complete-line byte prefixes of `openrouter_requests.jsonl`.

The durable files are:

- `openrouter_gpt4o_mini_root.json`: fixed provider, endpoint, model, and root
  identity.
- `openrouter_cost_state.json`: nanodollar cap, committed cost, reservations,
  usage totals, and cost-source counts.
- `openrouter_requests.jsonl`: append-only hashes, provider identity, response
  IDs, usage, provider `usage.cost` when present, token-derived cost, and
  reservation outcome. It does not contain keys, messages, prompts, or tools.
- `gateway_ready.json`: ephemeral loopback address and local artifact paths.

Each request reserves the maximum published 128K-context/16,384-output cost
before I/O using USD 0.15/M uncached input, USD 0.075/M cached input, and USD
0.60/M output. A valid provider `usage.cost` is the committed cost; otherwise
the token-derived cost is committed. Transport failures, non-2xx responses,
missing or malformed usage, and uncertain costs retain the reservation and
stop processing. The gateway performs one physical request and does not retry
an uncertain request.

## R301

The controlled R301 proxy remains in place. Its GPT-4o upstream must be the
marked loopback gateway. The dummy local bearer value has no provider billing
authority.

```bash
export R301_OPENROUTER_GATEWAY_KEY=local-openrouter-gateway

/opt/miniconda3/bin/python scripts/run_r301_organizer_retriever.py \
  --output-dir results/paper-experiments-20260714/r301-gpt55-flex-openrouter-gated-v1 \
  --r207-root results/paper-experiments-20260714/r207-formal-v1 \
  --samples 0-9 \
  --allow-model-requests \
  --gateway-root "$OPENAI_GPT55_FLEX_GATEWAY_ROOT" \
  --gpt4o-upstream "$OPENROUTER_GATEWAY_BASE_URL" \
  --openrouter-gateway-root "$OPENROUTER_GATEWAY_ROOT" \
  --gpt4o-api-key-env R301_OPENROUTER_GATEWAY_KEY
```

R301 records the gateway start prefix in `run_manifest.json`, the end prefix
in `complete.json`, and verifies every GPT-4o response ID against that interval
during its independent artifact audit.

## LongMemEval scoring

Use a new score output for the gated formal run. The primary scorer requires
both gateway arguments and refuses direct OpenRouter.

```bash
/opt/miniconda3/bin/python scripts/score_v88_gpt55_benchmarks.py \
  --benchmark longmemeval \
  --input '<audited evaluation_input.json>' \
  --source-audit '<audited LongMemEval run audit.json>' \
  --output results/paper-experiments-20260714/scores-gated-v1/longmemeval-primary.json \
  --judge primary \
  --openrouter-gateway-root "$OPENROUTER_GATEWAY_ROOT" \
  --openrouter-gateway-base-url "$OPENROUTER_GATEWAY_BASE_URL" \
  --allow-model-requests
```

The scorer records `provider=OpenRouter` through the gateway binding while
retaining `judge_requested_model=openai/gpt-4o-mini`. It disables the HTTP
client's automatic retry for the marked-gateway primary profile.

For LoCoMo, `score_v88_gpt55_benchmarks.py` rejects both `locomo` and
`locomo-cat5`. The only permitted evaluator is the frozen
`scripts/eval_full.py`, category 1-4 only.

## BEAM evaluation

The BEAM primary profile requires the marked root and exactly one evaluator
attempt per nugget. Use a new evaluation directory.

```bash
.venv-beam-gpt55/bin/python scripts/evaluate_v88_gpt55_beam.py \
  '<audited evaluation_input.json>' \
  --output-dir results/paper-experiments-20260714/beam-evaluation-gated-v1 \
  --profile primary \
  --base-url "$OPENROUTER_GATEWAY_BASE_URL" \
  --openrouter-gateway-root "$OPENROUTER_GATEWAY_ROOT" \
  --allow-model-requests \
  --max-retries 1 \
  --resume
```

## Audit order

After all model requests have stopped and no request is in flight, audit the
shared gateway first. The audit rejects any residual uncertain reservation.

```bash
/opt/miniconda3/bin/python scripts/audit_openrouter_gpt4o_mini_gateway.py \
  --result-root "$OPENROUTER_GATEWAY_ROOT" \
  --max-cost-usd "$OPENROUTER_MAX_COST_USD" \
  --output "$OPENROUTER_GATEWAY_ROOT/audit.json"
```

Then audit each consumer artifact:

```bash
/opt/miniconda3/bin/python scripts/audit_r301_organizer_retriever.py \
  results/paper-experiments-20260714/r301-gpt55-flex-openrouter-gated-v1

/opt/miniconda3/bin/python scripts/audit_v88_gpt55_scores.py \
  --benchmark locomo \
  --input '<audited questions_all.json>' \
  --source-audit '<audited LoCoMo run audit.json>' \
  --evaluation results/paper-experiments-20260714/scores-gated-v1/locomo-primary.json \
  --judge primary

.venv-beam-gpt55/bin/python scripts/evaluate_v88_gpt55_beam.py \
  '<audited evaluation_input.json>' \
  --output-dir results/paper-experiments-20260714/beam-evaluation-gated-v1 \
  --profile primary \
  --audit-only
```

The gateway audit is independent of the gateway implementation. It recomputes
the maximum reservation, token-derived costs, state totals, response counts,
model identity, and provider-`usage.cost` commitment. A retained reservation,
tampered cost/model/count, cap mismatch, or logged sensitive request field
causes failure.
