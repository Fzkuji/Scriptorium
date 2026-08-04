# Scripts

`scripts/` holds one shell entry point plus packages of Python implementation.
Anything you run by hand is either `run_experiment.sh` or a module invoked with
`python -m`; no loose `.py` files sit at this level.

## Run an experiment

```bash
# one run, exactly as the config says
scripts/run_experiment.sh scripts/configs/my-run.json

# one run per conversation
scripts/run_experiment.sh scripts/configs/my-run.json --samples conv-50 conv-51

# one run per input size, to see how much the Writer should see at once
scripts/run_experiment.sh scripts/configs/my-run.json --caps 4096 8192 16384 32768
```

`--caps` sweeps `writer_input_token_cap`. Each variant is retried once, and a
variant that still fails is skipped rather than ending the sweep. Success is
decided by whether `eval_full.json` was written, not by an exit code. When it
finishes, the surviving runs are summarized side by side.

## Packages

| Package | What is in it |
|---|---|
| `nativemem/` | The runners behind every stored result: LoCoMo, LongMemEval, ablations |
| `model_capacity/` | Writer input-capacity calibration |
| `evaluation/` | Judges, metrics, LLM clients, and `eval_full.py` |
| `analysis/` | `analyze_run.py` and one-question analyses |
| `adapters/` | One module per external memory system, for baseline comparison |
| `gateways/` | Budget-enforcing HTTP proxies that cap what a run can spend |
| `maintenance/` | Layout check, third-party manifest |
| `configs/` | Frozen command inputs and examples |

## Experiment packages

Each formal experiment keeps its contract, runner and auditor together: the
contract defines what an artifact must contain, the runner produces it, and the
auditor re-derives it with separate code so a result is checked by something
that did not produce it.

`locomo_baselines/`, `controlled_locomo/`, `longmemeval_m1/`, `m4_statistics/`,
`beam_controls/`, `token_budget/`, `human_agreement/`, `readonly_control/`.

Within one, the prefix gives the role: `*_contract.py` fixes the shape,
`run_*.py` produces, `audit_*.py` re-derives and fails on mismatch, and
`freeze_*.py` pins inputs once audits pass.

## Locked evaluator

`evaluation/eval_full.py` is the only permitted LoCoMo evaluator and is
hash-locked by `AGENTS.md`. Verify its SHA-256 before any LoCoMo scoring run.
Do not add a second scorer or wrap it with changed semantics.

Read a module's docstring for what it does; every one has one.
