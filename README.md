# Scriptorium

**Agent memory you can read.** The model keeps its own notes as Markdown files
— one topic per file, every sentence footnoted to the message it came from — so
memory opens in an editor, diffs in Git, and can always be traced back to what
was actually said.

```markdown
# Calvin's music career

## Craft and drive

Calvin writes new tunes, does studio sessions, and loves collaborating with
other artists.[^e-310b5c4c8e] ^36d94ab9

Calvin hit a creative block in May 2023. Dave advised taking a break, and
Calvin planned to follow that advice.[^e-078d330831][^e-fca34f2ddb] ^2b157171

[^e-310b5c4c8e]: Time: `2023-05-08`; Sources: [locomo/thread_4f2a…/msg_9c11…](../../sources/locomo/thread_4f2a….md#source-9c11…)
```

That file was written by the model, not by a template. Each `^id` is a block
another note can link to; each `[^e-…]` footnote carries the date and the
archived message it rests on. Nothing is stored that you cannot open.

## Install

```bash
pip install git+https://github.com/Fzkuji/Scriptorium.git
claude mcp add --scope user scriptorium -- \
  scriptorium mcp --workspace project=.memory --workspace global=~/memory
```

Start a new Claude Code session and memory is there. No init step, no database,
no service to run: a missing workspace is created on first use, `project=`
lands at the repository root of wherever you opened the session, and `global=`
is the same directory in every project. One `--workspace PATH` serves a single
unlayered memory instead.

The session gains six tools — `memory_status`, `memory_list`, `memory_read`,
`memory_grep`, `memory_search`, `memory_update`. Only the last one writes, and
it takes a unified diff restricted to `topics/**/*.md` and `core.md`. No shell
is exposed. Full setup and tool reference:
[`docs/integrations/claude-code.md`](docs/integrations/claude-code.md).

## How it works

**One write is one transaction.** New evidence and the note citing it commit
together, checked against the revision you read. A patch that cites a source it
did not supply, links a block that does not exist, or breaks the topic format
is refused whole, and the workspace is left byte-identical.

**Evidence is append-only.** `sources/**` holds what was said, written only by
the runtime. A note can be rewritten freely; what it rests on cannot.

**Views are derived, not authored.** `timeline/`, `recent_events.jsonl` and
`relations.json` are rebuilt after every successful write, so the model never
maintains an index by hand.

**Retrieval reads files.** BM25 over blocks and sources by default, with an
embedding backend if one is installed — and if it is not, embedding search says
so rather than silently returning something else.

**Layers, not copies.** A per-repository memory and one global memory are read
as a single memory with qualified paths (`global:topics/person.md`); each layer
stays a complete workspace that moves with its repository.

## Measured

Building memory for one LoCoMo conversation (conv-50, 30 sessions) with
deepseek-v4-flash, answering its 158 primary questions from that memory alone,
judged by gpt-4o-mini:

| Writer window | LoCoMo J | build time | build calls |
|---|---|---|---|
| 4K | 94.3 | 120 min | 577 |
| 8K | 93.0 | 61 min | 348 |
| 16K | **95.6** | 39 min | 215 |
| 32K | 93.0 | 39 min | 232 |

The Writer window is how much conversation the model sees in one call. Across
an eight-fold range accuracy moves by 2.6 points — inside the run-to-run spread
of this single conversation — while build cost triples at the small end. The
method does not need a long-context model; it needs more calls when the window
is small.

This is one conversation, not a leaderboard entry. Scores are only comparable
when the builder, answerer, judge, prompt and question set all match — see
[`docs/experiments/protocols/unified_evaluation_protocol.md`](docs/experiments/protocols/unified_evaluation_protocol.md).

## Two ways to run, one implementation

- **Interactive.** The stdio MCP server above, against a workspace you keep.
- **Experiment.** Benchmark runners that build memory in an isolated Claude
  Code subprocess with explicit models, endpoints and budgets, for LoCoMo,
  BEAM and LongMemEval.

Both go through the same `code/memory` — the same write transaction, the same
Markdown rules, the same retrieval. There is no separate interactive format,
and nothing in the interactive path is a simplified version of the measured
one.

`scriptorium validate --workspace ~/memory` parses every topic, checks source
and block links, and rebuilds the derived views in a scratch copy without
touching the original.

## Install for development

Python 3.12 is the supported runtime.

```bash
git clone https://github.com/Fzkuji/Scriptorium.git
cd Scriptorium
./setup.sh
source .venv/bin/activate
cd code && pytest -q tests
```

`setup.sh` creates the virtual environment at `~/.venvs/scriptorium`, links it
as `.venv`, installs `requirements-dev.txt`, and checks the repository layout.
The environment lives outside the repository because a syncing folder (iCloud
Drive, Dropbox) marks files hidden, and Python skips a hidden `.pth`, which
disables an editable install without saying so. Set `VENV_DIR` to put it
somewhere else. `setup.sh` refuses to rebind an existing environment to a
different interpreter, because the compiled packages inside it belong to the
one it was built with.

API credentials are never stored in this repository. Runners receive
credentials, models, endpoints and budgets through explicit CLI or config
parameters, and a config names a key **file** outside the repository rather
than a key.

Writer, Manager, verification and query trajectories run through the Claude
Agent SDK. Each uses an isolated temporary Claude configuration, does not read
your `~/.claude` settings or subscription session, and does not persist an SDK
session. Credentials reach the child process only.

The complete local research directory also holds benchmark data, stored results
and third-party checkouts that are deliberately not committed. To move the
working state to another computer, copy the whole directory, but not `.venv`
or `.venv-*`; run `./setup.sh` there instead.

## Layout

```text
code/
  scriptorium/                installable facade, CLI and MCP server
  memory/                     the implementation both paths share
  scripts/                    run_experiment.sh plus our own method's runners
    runners/                  one package per benchmark family
      conversation/           one conversation end to end: LoCoMo and BEAM
      longmemeval/            LongMemEval's many-sample queue
      beam/                   BEAM conversation conversion
      ablation/               ablation variants
      common/                 config parsing, atomic writes, hashing
    model_capacity/           Writer capacity calibration
    evaluation/               judges, metrics, and the locked evaluator
    analysis/                 summaries over stored runs
    configs/                  frozen command inputs
  baselines/                  systems compared against
    run_comparison.sh         score other systems under our own condition
    adapters/                 one runnable module per system
    third_party/              their checkouts, restored from manifest.json
  tests/                      grouped by implementation responsibility
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

Everything runnable lives under `code/`, and only there — run the commands
below from that directory. Generated analysis is stored under
`code/results/analysis/`; there is no code-level `experiments/` directory.
`code/memory/evaluation` is a compatibility link to `code/scripts/evaluation`,
required because the hash-locked `code/scripts/evaluation/eval_full.py` imports
its answerer as `memory.evaluation.answerer`; the implementation is maintained
only under `code/scripts/evaluation`.

## Running an experiment

Run these from `code/`. Every runner takes its settings from a JSON config, so
a run is one command and no credential reaches your shell history. Copy the
example and edit it:

```bash
cd code
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
python scripts/runners/run_conversation.py --config scripts/configs/my-run.json

# 3. Same run, build only, to inspect the memory before spending on answers
python scripts/runners/run_conversation.py --config scripts/configs/my-run.json \
  --build-only
```

Any config value can be overridden on the command line, which is convenient for
sweeps:

```bash
for sample in conv-50 conv-51 conv-52; do
  python scripts/runners/run_conversation.py \
    --config scripts/configs/my-run.json \
    --sample-id "$sample" \
    --output-dir "results/formal/sweep-$sample"
done
```

`run_longmemeval.py` accepts `--config` the same way. Run
`python scripts/runners/run_conversation.py --help` for the full option list.

### BEAM, for conversations long enough to strain a Writer

A LoCoMo conversation is about 23K tokens, so a Writer window above that never
binds. BEAM ships whole conversations of 100K, 500K and 1M tokens, each probed
by twenty questions across ten categories — long enough that every window size
binds, and enough questions per build to be worth the build.

```bash
# Convert one conversation into what the runner reads
python -m scripts.runners.beam.convert --size 100K --conversation 1 \
  --output benchmarks/beam/converted/beam100K-1.json

# Build, answer and judge; --benchmark picks the evaluator
python scripts/runners/run_conversation.py --config my-beam-run.json \
  --benchmark beam \
  --data benchmarks/beam/converted/beam100K-1.json \
  --sample-id beam100K-1 \
  --writer-input-token-cap 16384
```

Set `verify_every_sessions` and `local_reorg_every_sessions` to 1: BEAM has a
handful of very large sessions, so LoCoMo's every-fifth-session cadence would
never fire.

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

## Comparing against other systems

The paper's main table cites each system's own reported numbers, produced under
different answerers, prompts and judges. To measure them under one shared
condition instead:

```bash
baselines/run_comparison.sh --list
baselines/run_comparison.sh 0 mem0 zep naive
```

Each system retrieves with its own API; every system's retrieved memories then
go through the same answerer and the same judge, so the only thing that differs
between rows is retrieval. Gold answers cannot leak into a prompt, because the
answerer's entry point is `generate_answer(question, memories)` — the dataset's
answer is not one of its arguments.

## Data and external repositories

Benchmark datasets live under `code/benchmarks`; their upstream repositories
and licenses stay inside the dataset directories.

`code/baselines/third_party/manifest.json` records the remote URL and exact
commit for each external checkout. Regenerate it after changing one, so a
reported baseline number stays traceable to the code that produced it:

```bash
python baselines/generate_third_party_manifest.py
```

Third-party frameworks keep their own dependency files. Their old virtual
environments are not portable and are replaced by frozen package inventories
under `code/baselines/third_party/environments`.

## Results

Formal result directories and their internal paths under `code/results` keep
their names, so stored HTML links continue to resolve.
`code/results/STORAGE.md` records storage totals, verified duplicates and
archived content. Do not compare scores unless the builder, answerer, judge,
prompt, benchmark subset and evaluation protocol are the same.
