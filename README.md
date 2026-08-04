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
pip install git+https://github.com/Fzkuji/Scriptorium.git
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

Several `--workspace NAME=PATH` arguments serve layered memory — typically a
per-repository `project` layer plus one `global` layer shared by every
project. Reads span all layers with qualified paths
(`global:topics/person.md`); a write lands in the layer named by
`memory_update`, defaulting to the first workspace given. See
[`docs/integrations/claude-code.md`](docs/integrations/claude-code.md).

`scriptorium validate --workspace ~/memory` checks a workspace without
modifying it.

## Install for development

Python 3.12 is the supported runtime.

```bash
git clone https://github.com/Fzkuji/Scriptorium.git
cd Scriptorium
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
computer, copy the entire repository directory. Do not copy `.venv`
or `.venv-*`; run `./setup.sh` on the destination computer instead.

## Layout

```text
code/
  scriptorium/                installable facade, CLI and MCP server
  src/                        reusable Scriptorium implementation
  scripts/                    run_experiment.sh plus our own method's runners
    nativemem/                LoCoMo, LongMemEval and ablation runners
    model_capacity/           Writer capacity calibration
    evaluation/               judges, metrics, and the locked evaluator
    analysis/                 summaries over stored runs
    configs/                  frozen command inputs
  baselines/                  systems compared against
    run_comparison.sh         score other systems under our own condition
    adapters/                 one runnable module per system
    third_party/              their checkouts, restored from manifest.json
  tests/                      tests grouped by implementation responsibility
  benchmarks/                 local benchmark datasets
  figures/                    generated figures
  gold_memory/                curated memory fixtures
  results/                    formal runs, analysis and capacity artifacts
paper/                        independent paper Git repository
docs/
  Model-Aligned-Wiki.html     documentation entry and research overview
  integrations/               Claude Code setup and tool reference
  related-work/               paper Related Work, survey, and evidence
  method/                     current method, design and reports
  experiments/                plans, protocols, results, studies, and runs
  prompts/                    prompt references
  internal/                   research analysis and development records
```

Root-level `src`, `scripts`, `tests`, `benchmarks`, `figures`, `gold_memory`,
`results`, `baselines`, and `third_party` are relative compatibility symlinks. Executable
code is classified under `scripts/`; generated analysis is stored under
`results/analysis/`. There is no separate code-level `experiments/` directory.
`code/src/evaluation` is a compatibility link to `code/scripts/evaluation`
for the hash-locked `code/scripts/evaluation/eval_full.py`; evaluation implementation is
maintained only under `code/scripts/evaluation`.

## Running an experiment

Every runner takes its settings from a JSON config, so a run is one command and
no credential reaches your shell history. Copy the example and edit it:

```bash
cp scripts/configs/locomo.example.json scripts/configs/my-run.json
```

Point `api_key_file` and `judge_api_key_file` at files **outside** the
repository. Relative paths inside a config resolve against the config file, so
a config can be moved together with its inputs.

```bash
# 1. Measure how much input this model's Writer handles reliably
python -m scripts.model_capacity.calibrate_writer \
  --config scripts/configs/model_capacity.example.json

# 2. Build memory for one conversation and evaluate it
python scripts/nativemem/run_locomo.py --config scripts/configs/my-run.json

# 3. Same run, build only, to inspect the memory before spending on answers
python scripts/nativemem/run_locomo.py --config scripts/configs/my-run.json \
  --build-only
```

Any config value can be overridden on the command line, which is convenient for
sweeps:

```bash
for sample in conv-50 conv-51 conv-52; do
  python scripts/nativemem/run_locomo.py \
    --config scripts/configs/my-run.json \
    --sample-id "$sample" \
    --output-dir "results/formal/sweep-$sample"
done
```

`run_longmemeval.py` accepts `--config` the same way. Run
`python scripts/nativemem/run_locomo.py --help` for the full option list.

A run writes `status.json`, `build.json`, `call_log.json`, `performance.json`
and `eval_full.json` into its output directory, and is resumable: rerunning the
same command skips completed work.

### Reading the cost numbers

`performance.json` reports two figures, and neither is a bill.

`estimated_cost_usd` is the token counts in `call_log.json` multiplied by the
prices you passed on the command line. It is only as good as those inputs, so
two things will silently distort it:

- **Cache rates.** Providers bill a cache read far below fresh input —
  deepseek-v4-flash on packyapi charges $0.005/M against $0.25/M. Pass
  `--cache-read-usd-per-million`; omit it and cache reads are priced as input,
  which on a cache-heavy run overstates cost several fold.
- **Provider token accounting.** Different gateways count the same work
  differently. The same conversation recorded 4.11M input tokens on one
  provider and 26M on another, so costs from two providers are not comparable
  no matter how correct the prices are.

`anthropic_equivalent_cost_usd` comes from the Claude Agent SDK, which prices
every trajectory at Anthropic's rates even when `base_url` points elsewhere. It
is never what you were billed.

For a real figure, read the provider's own usage dashboard.

## Other commands

```bash
# Tests
pytest -q \
  tests/management tests/markdown tests/retrieval tests/runtime tests/scripts
```

The documentation entry is [`docs/Model-Aligned-Wiki.html`](docs/Model-Aligned-Wiki.html).
Experiment commands and evaluation rules are documented in
[`docs/experiments/protocols/unified_evaluation_protocol.md`](docs/experiments/protocols/unified_evaluation_protocol.md)
and [`docs/experiments/experiment.html`](docs/experiments/experiment.html).

## Data and external repositories

Benchmark datasets remain under `code/benchmarks` and are accessed through the
root `benchmarks` link. Their upstream repositories and licenses remain inside
the dataset directories.

`code/baselines/third_party/manifest.json` records the remote URL and exact commit for
each external Git repository. Regenerate it after changing a checkout:

```bash
python baselines/generate_third_party_manifest.py
```

Third-party frameworks keep their own dependency files. Their old virtual
environments are not portable and are replaced by frozen package inventories
under `code/baselines/third_party/environments`.

## Results

All existing formal result directory names and internal paths remain unchanged
under `code/results`; the root `results` link preserves stored HTML links.
`code/results/STORAGE.md` records storage totals, verified duplicates, and any
archived content. Do not compare scores unless the builder, answerer, judge,
prompt, benchmark subset, and evaluation protocol are the same.
