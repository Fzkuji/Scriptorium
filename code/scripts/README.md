# Scripts

`scripts/` holds one shell entry point plus packages of Python implementation.
Anything you run by hand is either `run_experiment.sh` or a module invoked with
`python -m`; no loose `.py` files sit at this level.

## Run an experiment

`run_experiment.sh` is the only thing here you invoke directly. It builds
memory, evaluates it, and prints a summary. Everything it needs comes from one
JSON config, so no credential reaches your shell history.

```bash
cp scripts/configs/locomo.example.json scripts/configs/my-run.json
# edit: output_dir, sample_id, base_url, api_key_file, judge_api_key_file, prices
```

`api_key_file` and `judge_api_key_file` must point **outside** the repository.
Relative paths inside a config resolve against the config file.

```bash
# one run, exactly as the config says
scripts/run_experiment.sh scripts/configs/my-run.json

# one run per conversation
scripts/run_experiment.sh scripts/configs/my-run.json --samples conv-50 conv-51

# one run per input size
scripts/run_experiment.sh scripts/configs/my-run.json --caps 4096 8192 16384 32768
```

A sweep writes each variant beside the configured `output_dir`, suffixed with
its label — `--caps 4096` lands in `<output_dir>-cap4096`. A variant is retried
once, and one that still fails is skipped rather than ending the sweep. Success
is decided by whether `eval_full.json` was written, not by an exit code, because
the runner can exit 0 having produced nothing. Surviving runs are summarized
side by side at the end.

### Measuring how much input a model handles

`--caps` sweeps `writer_input_token_cap`, the amount of conversation the Writer
sees in one call. This is the experiment behind the capacity numbers: accuracy
and build time at each size, on the same conversation, with everything else
fixed.

```bash
scripts/run_experiment.sh scripts/configs/my-run.json --caps 4096 8192 16384 32768
```

Four sizes on one LoCoMo conversation took roughly three hours on
deepseek-v4-flash. Read `judge_score` accuracy from each `eval_full.json` and
wall time from each `build.json`; the summary at the end prints both.

Do not read the cost column across providers. Gateways count tokens
differently — the same conversation recorded 4.11M input tokens on one and 26M
on another — so cost is comparable only within a single provider. See the cost
section in the top-level `README.md`.

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
