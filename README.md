# NativeMem

NativeMem uses a model-managed file system as long-term memory. The repository
contains the implementation, benchmark adapters, experiment scripts, stored
results, paper source, and third-party reproduction metadata.

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
not stored in this repository. Set the variables required by the selected
runner before starting an experiment; each runner prints its effective model,
endpoint, and relevant environment variables.

The complete local research directory also contains benchmark data, stored
results, and third-party checkouts that are intentionally not committed to the
main Git repository. To transfer the complete working state to another
computer, copy the entire `model-aligned-wiki` directory. Do not copy `.venv`
or `.venv-*`; run `./setup.sh` on the destination computer instead.

## Layout

```text
Model-Aligned-Wiki.html       project overview
code/
  src/                        NativeMem implementation
  scripts/                    experiment and audit commands
  tests/                      repository tests
  benchmarks/                 local benchmark datasets
  experiments/                experiment definitions
  figures/                    generated figures
  gold_memory/                curated memory fixtures
  results/                    existing experiment artifacts
  third_party/                external framework checkouts
paper/                        independent paper Git repository
docs/                         technical documents and experiment plan
```

Root-level `src`, `scripts`, `tests`, `benchmarks`, `experiments`, `figures`,
`gold_memory`, `results`, and `third_party` are relative compatibility
symlinks. Existing commands and stored result links continue to work after the
reorganization.

## Basic commands

Run commands from the repository root:

```bash
# Layout and transfer check
python scripts/verify_portable_layout.py

# Core V11 tests
pytest -q \
  tests/test_v11_memory.py \
  tests/test_reanswer_longmemeval_existing_memory.py \
  tests/test_run_v88_gpt55_longmemeval.py \
  tests/test_run_v11_gpt55_frontier_longmemeval.py \
  tests/test_run_v11_memory_organizer_pilot.py

# Example memory build
python src/nativemem.py --sample 0 --outdir results/run --validate
```

Experiment commands and evaluation rules are documented in
[`docs/unified_evaluation_protocol.md`](docs/unified_evaluation_protocol.md) and
[`docs/experiment-plan.html`](docs/experiment-plan.html).

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

All existing result directory names and internal paths remain unchanged under
`code/results`; the root `results` link preserves older scripts and HTML links.
`code/results/STORAGE.md` records storage totals, verified duplicates, and any
archived content. Do not compare scores unless the builder, answerer, judge,
prompt, benchmark subset, and evaluation protocol are the same.
