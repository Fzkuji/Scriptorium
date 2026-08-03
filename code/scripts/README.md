# Scripts

Executable entry points. Nothing here is imported by `src/`; the dependency
runs the other way.

Paths in stored results and HTML reports point at these filenames, so scripts
are not moved or renamed once a formal run has referenced them. New work goes
into the packages below rather than into another top-level file.

## Start here

| Command | Purpose |
|---|---|
| `nativemem/run_locomo.py --config …` | Build memory for one LoCoMo conversation and evaluate it |
| `nativemem/run_longmemeval.py --config …` | Run one resumable LongMemEval-S shard |
| `nativemem/run_ablation.py` | Paired ablation conditions |
| `model_capacity/calibrate_writer.py --config …` | Measure the Writer's reliable input size for a model |
| `analyze_run.py results/<run> […]` | Summarize run directories: scores, memory statistics, build cost |
| `verify_portable_layout.py` | Check the repository layout and symlinks |

Runners accept `--config PATH` for a JSON file of defaults; explicit flags
override it. See `configs/locomo.example.json`.

## Packages

- `nativemem/` — benchmark runners. `common/` holds shared IO, signal handling
  and config loading; `locomo/`, `longmemeval/` and `ablation/` hold one
  benchmark each.
- `model_capacity/` — Writer capacity calibration.
- `evaluation/` — evaluation implementation. `src/evaluation` is a
  compatibility link to this directory for the hash-locked evaluator.
- `gateways/` — budget-enforcing HTTP proxies for paid providers. They commit
  and reserve spend per request so a run cannot exceed its cap.
- `configs/` — frozen command inputs and example configs.

## Locked evaluator

`eval_full.py` is hash-locked by `AGENTS.md` and is the only permitted LoCoMo
evaluator. Do not edit, wrap with changed semantics, or add a second scorer.
Verify its SHA-256 before any LoCoMo scoring run.

## The rest of the top level

The remaining files support specific formal experiments and follow a naming
convention:

- `*_contract.py` — a frozen, strict input/output contract shared by a runner
  and its auditor. These define what an artifact must contain.
- `run_*.py` — execute one experiment or stage.
- `audit_*.py` — independently re-derive an artifact and fail on mismatch.
  Auditors are deliberately separate from runners so a result is checked by
  code that did not produce it.
- `freeze_*.py` — pin inputs after audits pass, so later stages cannot drift.
- `analyze_*.py`, `score_*.py` — post-hoc analysis over stored results.

Read a file's module docstring for what it does; every script has one.
