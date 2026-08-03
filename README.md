# Scriptorium

Scriptorium uses model-managed Markdown files as external memory. The repository
contains the current implementation, benchmark runners, evaluation code,
stored results, paper source, and third-party reproduction metadata.

The code runs in two modes over one shared implementation:

- **Experiment.** Benchmark runners start isolated Claude Code subprocesses
  with explicit models, endpoints and budgets, for LoCoMo, LongMemEval and
  ablations.
- **Interactive.** A local stdio MCP server exposes one memory workspace to a
  Claude Code session you are already using. See
  [`docs/integrations/claude-code.md`](docs/integrations/claude-code.md).

Both paths share `code/src/management`, `markdown`, `retrieval` and `runtime`.
There is no separate interactive memory format.

## Use from Claude Code

```bash
pip install git+https://github.com/Fzkuji/scriptorium.git
scriptorium init ~/memory
claude mcp add --scope user scriptorium -- \
  scriptorium mcp --workspace ~/memory
```

The session gains six tools: `memory_status`, `memory_list`, `memory_read`,
`memory_grep`, `memory_search` and `memory_update`. Only `memory_update`
writes, and it accepts a unified diff restricted to `topics/**/*.md` and
`core.md`; no shell is exposed. New evidence and the topic edit citing it
commit as one transaction, after which the runtime rebuilds the timeline,
recent-events and relations views.

`scriptorium validate --workspace ~/memory` checks a workspace without
modifying it.

## Install for development

Python 3.12 is the supported runtime.

```bash
git clone https://github.com/Fzkuji/scriptorium.git
cd scriptorium
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a local virtual environment, installs
`requirements-dev.txt`, and checks the repository layout. API credentials are
not stored in this repository. Current commands receive credentials, models,
Anthropic-compatible endpoints and budgets through explicit CLI or function parameters. The
capacity calibration command reads its credential from the file named by
`api_key_file` in its local config.

Writer, Manager, verification and query trajectories run through the Claude
Agent SDK. Each trajectory uses an isolated temporary Claude configuration,
does not read the user's `~/.claude` settings or subscription session, and does
not persist an SDK session. The adapter passes credentials only to the child
Claude Code process; it does not modify the parent process environment.

The complete local research directory also contains benchmark data, stored
results, and third-party checkouts that are intentionally not committed to the
main Git repository. To transfer the complete working state to another
computer, copy the entire `scriptorium` directory. Do not copy `.venv`
or `.venv-*`; run `./setup.sh` on the destination computer instead.

## Layout

```text
code/
  scriptorium/       installable facade, CLI and MCP server
  src/                        reusable Scriptorium implementation
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
  integrations/               Claude Code setup and tool reference
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

# Current Scriptorium tests
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
