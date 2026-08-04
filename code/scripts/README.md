# Scripts

Executable entry points. Nothing here is imported by `src/`; the dependency
runs the other way.

Stored results and HTML reports mention some of these scripts by path, but those
are records rather than live callers: with every script moved, only two are
still executed by their original path, and only those keep a forwarding module
at the top level.

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

## Experiment directories

Each formal experiment keeps its contract, runner and auditor together, because
those three files only make sense as a set: the contract defines what an
artifact must contain, the runner produces it, and the auditor re-derives it
independently.

| Directory | Experiment |
|---|---|
| `locomo_baselines/` | Retrieval-only LoCoMo baselines |
| `controlled_locomo/` | Controlled LoCoMo answering, its budget proxies and audits |
| `longmemeval_m1/` | LongMemEval-S M1 baselines, backends and shared answers |
| `m4_statistics/` | Preregistered paired M4 statistics and failure analysis |
| `beam_controls/` | R115 BEAM controlled rows |
| `token_budget/` | R004/G0.2 visible-token budget checks |
| `human_agreement/` | R501 annotation packet and agreement scoring |

Within a directory the prefixes tell you the role: `*_contract.py` freezes the
shape of an artifact, `run_*.py` produces one, `audit_*.py` re-derives it and
fails on mismatch, and `freeze_*.py` pins inputs once audits pass. Auditors are
deliberately separate from runners so a result is checked by code that did not
produce it.

Import these as `scripts.<group>.<name>`. Two scripts —
`run_visible_token_budget_sanity.py` and `audit_visible_token_budget.py` — are
still launched by their original top-level path, so those keep a forwarding
module there; nothing else does.

## Still at the top level

Eleven files, in four kinds.

**Run these directly.**

| Script | What it does |
|---|---|
| `analyze_run.py` | Summarize run directories: scores, memory statistics, cost |
| `verify_portable_layout.py` | Check the repository layout and symlinks |
| `generate_third_party_manifest.py` | Regenerate `third_party/manifest.json` after changing a checkout |

**Analyses for one past question.** Each still has its study document and its
stored results, so they are kept, but nothing calls them — run them by hand
when revisiting that question.

| Script | Question |
|---|---|
| `analyze_nativemem_ablation.py` | Paired LoCoMo and BEAM ablation |
| `judge_showdown.py` | v8.8 against v9.0c, decided by `docs/experiments/studies/v9_research_protocol.md` |
| `evaluate_temporal_filter_retrieval.py` | Retrieval without dates against an oracle date window |
| `audit_readonly_nativemem_control.py` | R116 and R203 read-only view controls |

**The locked LoCoMo path.**

| Script | What it does |
|---|---|
| `eval_full.py` | The only permitted LoCoMo evaluator, hash-locked by `AGENTS.md` |
| `run_locked_locomo_eval_with_evidence.py` | Runs that evaluator while recording HTTP evidence |

**Forwarding modules**, for the two scripts still launched by their original
path: `run_visible_token_budget_sanity.py` and `audit_visible_token_budget.py`.
Both really live in `token_budget/`.

Read a file's module docstring for what it does; every script has one.
