# NativeMem

NativeMem uses model-managed Markdown files as external memory. The repository
contains the current implementation, benchmark runners, evaluation code,
stored results, paper source, and third-party reproduction metadata.

## Install

Python 3.12 is the supported runtime.

```bash
git clone <repository-url> model-aligned-wiki
cd model-aligned-wiki
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a local virtual environment, installs
`requirements-dev.txt`, and checks the repository layout. API credentials are
not stored in this repository. Current NativeMem commands receive credentials,
models, endpoints and budgets through explicit CLI or function parameters. The
capacity calibration command reads its credential from the file named by
`api_key_file` in its local config.

The complete local research directory also contains benchmark data, stored
results, and third-party checkouts that are intentionally not committed to the
main Git repository. To transfer the complete working state to another
computer, copy the entire `model-aligned-wiki` directory. Do not copy `.venv`
or `.venv-*`; run `./setup.sh` on the destination computer instead.

## Layout

```text
code/
  src/                        reusable NativeMem implementation
  scripts/                    runners, adapters, evaluation and analysis
    configs/                  frozen command inputs
    model_capacity/           Writer capacity calibration
    nativemem/                LoCoMo, LongMemEval and ablation runners
  tests/                      tests grouped by implementation responsibility
  benchmarks/                 local benchmark datasets
  figures/                    generated figures
  gold_memory/                curated memory fixtures
  results/                    formal runs, analysis and capacity artifacts
  third_party/                external framework checkouts
paper/                        independent paper Git repository
docs/
  Model-Aligned-Wiki.html     documentation entry and research overview
  related-work/               paper Related Work, survey, and evidence
  method/                     current method, design and reports
  experiments/                plans, protocols, results, studies, and runs
  prompts/                    prompt references
  archive/                    superseded historical documents
  internal/                   research analysis and development records
```

Root-level `src`, `scripts`, `tests`, `benchmarks`, `figures`, `gold_memory`,
`results`, and `third_party` are relative compatibility symlinks. Executable
code is classified under `scripts/`; generated analysis is stored under
`results/analysis/`. There is no separate code-level `experiments/` directory.
`code/src/evaluation` is a compatibility link to `code/scripts/evaluation`
for the hash-locked `code/scripts/eval_full.py`; evaluation implementation is
maintained only under `code/scripts/evaluation`.

## Basic commands

Run commands from the repository root:

```bash
# Layout and transfer check
python scripts/verify_portable_layout.py

# Current NativeMem tests
pytest -q \
  tests/management \
  tests/markdown \
  tests/retrieval \
  tests/runtime \
  tests/scripts

# Calibrate one model and Writer protocol
python -m scripts.model_capacity.calibrate_writer \
  --config scripts/configs/model_capacity.example.json

# Inspect benchmark runner parameters
python scripts/nativemem/run_locomo.py --help
python scripts/nativemem/run_longmemeval.py --help
```

The documentation entry is [`docs/Model-Aligned-Wiki.html`](docs/Model-Aligned-Wiki.html).
Experiment commands and evaluation rules are documented in
[`docs/experiments/protocols/unified_evaluation_protocol.md`](docs/experiments/protocols/unified_evaluation_protocol.md)
and [`docs/experiments/experiment.html`](docs/experiments/experiment.html).

## Data and external repositories

Benchmark datasets remain under `code/benchmarks` and are accessed through the
root `benchmarks` link. Their upstream repositories and licenses remain inside
the dataset directories.

`code/third_party/manifest.json` records the remote URL and exact commit for
each external Git repository. Regenerate it after changing a checkout:

```bash
python scripts/generate_third_party_manifest.py
```

Third-party frameworks keep their own dependency files. Their old virtual
environments are not portable and are replaced by frozen package inventories
under `code/third_party/environments`.

## Results

All existing formal result directory names and internal paths remain unchanged
under `code/results`; the root `results` link preserves stored HTML links.
`code/results/STORAGE.md` records storage totals, verified duplicates, and any
archived content. Do not compare scores unless the builder, answerer, judge,
prompt, benchmark subset, and evaluation protocol are the same.
