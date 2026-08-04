# NativeMem Source Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current V11 implementation the only NativeMem implementation, keep only reusable core code under `code/src/`, add reusable Writer-capacity calibration, and remove the ambiguous `code/experiments/` directory.

**Architecture:** The current `nativemem_versions/v11` modules move directly under `code/src/` and lose their version labels. Executable adapters, evaluation programs, gateways, benchmark runners, analysis and calibration commands live under `code/scripts/`; tests and results remain sibling directories. Writer calibration tests complete-session batch boundaries and emits a JSON token budget consumed by `BuildConfig` through explicit parameters.

**Tech Stack:** Python 3.12, dataclasses, pathlib, argparse, OpenAI-compatible clients, tiktoken, pytest, Git.

## Global Constraints

- `code/src/` contains only reusable NativeMem core code.
- No environment variable may configure NativeMem core, calibration, or current benchmark runners.
- A Writer batch never splits a session.
- Files already under `code/results/`, `code/benchmarks/`, `code/third_party/`, `paper/`, and generated memory directories are not deleted or renamed; migrated analysis files may be added under `code/results/analysis/`.
- `scripts/eval_full.py` remains unchanged with SHA-256 `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`.
- Old implementations remain recoverable through Git commit `31cc34c`.
- Do not add a new LoCoMo evaluator or report calibration output as a LoCoMo score.

---

### Task 1: Promote the current core implementation into `src/`

**Files:**
- Create: `code/src/__init__.py`
- Rename: `code/src/nativemem_versions/v11/adapter.py` → `code/src/build.py`
- Rename: `code/src/nativemem_versions/v11/conversation.py` → `code/src/conversation.py`
- Rename: `code/src/nativemem_versions/v11/memory/` → `code/src/management/`
- Rename: `code/src/nativemem_versions/v11/topic_markdown/` → `code/src/markdown/`
- Rename: `code/src/nativemem_versions/v11/retrieval/` → `code/src/retrieval/`
- Rename: `code/src/nativemem_versions/v11/online_runtime.py` → `code/src/runtime/online.py`
- Rename: `code/src/nativemem_versions/v11/runtime_state.py` → `code/src/runtime/state.py`
- Rename: `code/src/nativemem_versions/v11/derived_views.py` → `code/src/runtime/derived_views.py`
- Rename: `code/src/nativemem_versions/v11/reconciliation.py` → `code/src/management/reconciliation.py`
- Rename: `code/src/memory_bm25.py` → `code/src/retrieval/bm25.py`
- Rename: `code/src/memory_embedding.py` → `code/src/retrieval/embedding.py`
- Create: `code/src/runtime/__init__.py`
- Create: `code/tests/test_core_layout.py`
- Modify: imports in all moved core files

**Interfaces:**
- Produces: `src.BuildConfig`, `src.build_memory`, `src.MemoryConfig`, `src.MemoryWorkspace`, `src.QueryConfig`, `src.collect_answer`.
- Produces: implementation packages `src.management`, `src.markdown`, `src.retrieval`, and `src.runtime`.

- [ ] **Step 1: Write the failing public-import test**

```python
def test_current_core_is_imported_without_a_version_namespace():
    from src import (
        BuildConfig,
        MemoryConfig,
        MemoryWorkspace,
        QueryConfig,
        build_memory,
        collect_answer,
    )

    assert BuildConfig.__module__ == "src.build"
    assert MemoryConfig.__module__.startswith("src.management")
    assert MemoryWorkspace.__module__.startswith("src.management")
    assert QueryConfig.__module__ == "src.retrieval.config"
    assert callable(build_memory)
    assert callable(collect_answer)
```

- [ ] **Step 2: Run the test and verify that the direct API is absent**

Run: `pytest -q code/tests/test_core_layout.py`

Expected: FAIL because `code/src/__init__.py` does not export the current implementation.

- [ ] **Step 3: Move the current modules and create the public API**

Create `code/src/__init__.py` with these direct exports:

```python
from .build import BuildConfig, build_memory
from .management import MemoryConfig, MemoryWorkspace
from .retrieval import QueryConfig, collect_answer

__all__ = [
    "BuildConfig",
    "MemoryConfig",
    "MemoryWorkspace",
    "QueryConfig",
    "build_memory",
    "collect_answer",
]
```

Use relative imports after each move. Remove `V11` from module docstrings and error messages. Delete the compatibility-only `management/model.py` and import its concrete functions from `management/agent.py`, `management/provider.py`, and `management/model_reconciliation.py`.

- [ ] **Step 4: Run core tests through their original paths**

Run:

```bash
pytest -q \
  code/tests/test_core_layout.py \
  code/tests/test_v11_memory.py \
  code/tests/test_v11_topic_markdown.py \
  code/tests/test_v11_reconciliation.py \
  code/tests/test_v11_derived_views.py \
  code/tests/test_v11_runtime_state.py \
  code/tests/test_v11_current_runtime.py
```

Expected: PASS after imports are updated to the new `src.*` paths.

- [ ] **Step 5: Commit the promoted core**

```bash
git add code/src code/tests/test_core_layout.py code/tests/test_v11_*.py
git commit -m "refactor: promote NativeMem core into src"
```

### Task 2: Keep executable infrastructure outside `src/`

**Files:**
- Rename: `code/src/adapters/` → `code/scripts/adapters/`
- Rename: `code/src/evaluation/` → `code/scripts/evaluation/`
- Rename: `code/src/eval_standard.py` → `code/scripts/evaluation/standard.py`
- Rename: `code/src/run_judge.py` → `code/scripts/evaluation/run_judge.py`
- Rename: provider and gateway modules at `code/src/*.py` → `code/scripts/gateways/`
- Create: `code/tests/test_code_layout.py`
- Modify: imports under `code/scripts/` and `code/tests/`

The gateway move covers `anthropic_openai_compat.py`, `chatgpt_proxy.py`, `codebuddy_proxy.py`, `openai_gpt55_flex_gateway.py`, `openai_gpt55_flex_gateway_evidence.py`, `openrouter_gateway_evidence.py`, and `openrouter_gpt4o_mini_gateway.py`.

**Interfaces:**
- Consumes: the direct `src` API from Task 1.
- Produces: `scripts.adapters`, `scripts.evaluation`, and `scripts.gateways` executable support packages.

- [ ] **Step 1: Write the layout test**

```python
from pathlib import Path


def test_src_contains_only_core_packages():
    root = Path(__file__).resolve().parents[1] / "src"
    assert not (root / "adapters").exists()
    assert not (root / "evaluation").exists()
    for filename in (
        "chatgpt_proxy.py",
        "openai_gpt55_flex_gateway.py",
        "openrouter_gpt4o_mini_gateway.py",
    ):
        assert not (root / filename).exists()
```

- [ ] **Step 2: Run the test and verify that executable infrastructure remains in `src/`**

Run: `pytest -q code/tests/test_code_layout.py`

Expected: FAIL on the existing `src/adapters` and `src/evaluation` directories.

- [ ] **Step 3: Move executable infrastructure and update imports mechanically**

Replace current imports as follows throughout active scripts and retained tests:

```text
src.adapters       -> scripts.adapters
src.evaluation     -> scripts.evaluation
src.chatgpt_proxy  -> scripts.gateways.chatgpt_proxy
src.openai_gpt55_flex_gateway -> scripts.gateways.openai_gpt55_flex_gateway
src.openrouter_gpt4o_mini_gateway -> scripts.gateways.openrouter_gpt4o_mini_gateway
```

Provider-specific sibling imports become relative imports inside `scripts/gateways/`.

- [ ] **Step 4: Run layout, adapter, gateway and evaluator tests**

Run:

```bash
pytest -q \
  code/tests/test_code_layout.py \
  code/tests/test_chatgpt_proxy.py \
  code/tests/test_openai_gpt55_flex_gateway.py \
  code/tests/test_openrouter_gpt4o_mini_gateway.py \
  code/tests/test_evaluation_judges.py \
  code/tests/test_m4_reliability.py
```

Expected: PASS.

- [ ] **Step 5: Commit the infrastructure move**

```bash
git add code/src code/scripts code/tests
git commit -m "refactor: separate core from executable infrastructure"
```

### Task 3: Consolidate NativeMem runners and remove old implementations

**Files:**
- Rename: `code/scripts/v11_common/` → `code/scripts/nativemem/common/`
- Rename: `code/scripts/v11_locomo/` → `code/scripts/nativemem/locomo/`
- Rename: `code/scripts/v11_longmemeval/` → `code/scripts/nativemem/longmemeval/`
- Rename: `code/scripts/v11_ablation/` → `code/scripts/nativemem/ablation/`
- Rename: `code/scripts/run_v11_locomo.py` → `code/scripts/nativemem/run_locomo.py`
- Rename: `code/scripts/run_v11_gpt55_frontier_longmemeval.py` → `code/scripts/nativemem/run_longmemeval.py`
- Rename: `code/scripts/reanswer_longmemeval_existing_memory.py` → `code/scripts/nativemem/reanswer_longmemeval.py`
- Delete: `code/src/nativemem.py`, `code/src/v8_memory.py`, `code/src/v10_memory.py`, `code/src/nativemem_versions/`, `code/src/legacy/`, `code/scripts/adapters/run_nativemem.py`, and the old NativeMem version tests and scripts defined below
- Rename retained current tests into `code/tests/management/`, `code/tests/markdown/`, `code/tests/retrieval/`, `code/tests/runtime/`, and `code/tests/scripts/`
- Modify: `code/scripts/maintenance/verify_portable_layout.py`

**Interfaces:**
- Consumes: `src` and `scripts.*` packages from Tasks 1–2.
- Produces: version-free NativeMem commands under `scripts/nativemem/`.

- [ ] **Step 1: Extend the layout test with removed-version assertions**

```python
def test_no_active_version_router_or_old_native_memory_modules():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/nativemem.py",
        "src/v8_memory.py",
        "src/v10_memory.py",
        "src/nativemem_versions",
        "src/legacy",
        "scripts/adapters/run_nativemem.py",
    ):
        assert not (root / relative).exists()
```

- [ ] **Step 2: Run the layout test and verify that old implementations still exist**

Run: `pytest -q code/tests/test_code_layout.py`

Expected: FAIL on one or more old paths.

- [ ] **Step 3: Move current runners and delete deterministic legacy dependants**

Delete any script or test whose active implementation imports one of these removed paths and does not also exercise the direct current core:

```text
src.v8_memory
src.v10_memory
src.nativemem_versions.v8
src.nativemem_versions.v10
src.adapters.run_nativemem
```

Delete tests named `test_v7_*`, `test_v8_*`, `test_v9_*`, and `test_v10_*`. Keep the locked `code/scripts/eval_full.py`, all general baseline adapters, and current LoCoMo/LongMemEval runners.

- [ ] **Step 4: Verify there are no active legacy imports**

Run:

```bash
if rg -n "v8_memory|v10_memory|nativemem_versions|adapters\.run_nativemem" \
  code/src code/scripts code/tests --glob '*.py'; then
  exit 1
fi
```

Expected: no matches and exit status 0.

- [ ] **Step 5: Run the retained runner and core tests**

Run:

```bash
pytest -q \
  code/tests/management \
  code/tests/markdown \
  code/tests/retrieval \
  code/tests/runtime \
  code/tests/scripts
```

Expected: PASS.

- [ ] **Step 6: Commit the version removal**

```bash
git add -A code/src code/scripts code/tests
git commit -m "refactor: remove superseded NativeMem versions"
```

### Task 4: Remove `code/experiments/` and classify its contents

**Files:**
- Rename: `code/experiments/gpt56-chunk-curve/analyze.py` → `code/scripts/analysis/analyze_gpt56_chunk_curve.py`
- Rename: experiment manifests and run matrices → `code/scripts/configs/<study-name>/`
- Rename: analysis files and artifacts → `code/results/analysis/<study-name>/`
- Delete: `code/experiments/*/code/`, caches, `.pyc`, and `.DS_Store`
- Delete: root `experiments` symlink
- Modify: `README.md`, `code/scripts/maintenance/verify_portable_layout.py`, and current references outside `docs/internal/`
- Test: `code/tests/test_code_layout.py`

**Interfaces:**
- Produces: executable analysis under `scripts/analysis`, reproducibility inputs under `scripts/configs`, and outputs under `results/analysis`.

- [ ] **Step 1: Extend the layout test**

```python
def test_experiments_directory_is_not_part_of_the_code_layout():
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "code" / "experiments").exists()
    assert not (repository / "experiments").exists()
    assert (repository / "code" / "scripts" / "configs").is_dir()
    assert (repository / "code" / "results" / "analysis").is_dir()
```

- [ ] **Step 2: Run the test and verify that `experiments/` still exists**

Run: `pytest -q code/tests/test_code_layout.py`

Expected: FAIL.

- [ ] **Step 3: Move the tracked inputs and outputs, then delete frozen code**

Use these destinations:

```text
gpt56-chunk-curve/experiment_manifest.json -> scripts/configs/gpt56-chunk-curve/
gpt56-chunk-curve/run_matrix.jsonl         -> scripts/configs/gpt56-chunk-curve/
gpt56-session-groups/*                     -> scripts/configs/gpt56-session-groups/
v11-ablation-gpt56-sol/*                   -> scripts/configs/nativemem-ablation-gpt56-sol/
gpt56-chunk-curve/analysis/*               -> results/analysis/gpt56-chunk-curve/
gpt56-chunk-curve/artifacts/*              -> results/analysis/gpt56-chunk-curve/artifacts/
gpt56-chunk-curve/raw/README.md             -> results/analysis/gpt56-chunk-curve/raw/README.md
```

- [ ] **Step 4: Run layout and analysis tests**

Run:

```bash
pytest -q \
  code/tests/test_code_layout.py \
  code/tests/test_analyze_gpt56_session_group_sizes.py \
  code/tests/test_analyze_gpt56_session_group_results.py
```

Expected: PASS after paths are updated.

- [ ] **Step 5: Commit the directory removal**

```bash
git add -A README.md code/experiments code/scripts code/results code/tests experiments
git commit -m "refactor: classify experiment inputs and outputs"
```

### Task 5: Add reusable Writer-capacity calibration

**Files:**
- Create: `code/src/runtime/tokenization.py`
- Create: `code/src/runtime/capacity.py`
- Modify: `code/src/build.py`
- Modify: `code/src/management/api.py`
- Create: `code/scripts/model_capacity/calibrate_writer.py`
- Create: `code/scripts/configs/model_capacity/default_samples.json`
- Create: `code/tests/model_capacity/test_capacity.py`
- Create: `code/tests/model_capacity/test_calibrate_writer.py`

**Interfaces:**
- Produces: `TokenCounter.resolve(model: str) -> TokenCounter` and `TokenCounter.count(text: str) -> int`.
- Produces: `WriterCapacity.from_json(path: Path) -> WriterCapacity` with `safe_writer_request_tokens: int`.
- Produces: `pack_complete_sessions(sessions, *, budget, request_tokens) -> list[list[dict]]`.
- Produces: `select_safe_capacity(outcomes: dict[int, dict]) -> dict`.
- Produces: `calibrate_writer(client, model, samples, candidate_tokens, output_dir) -> dict`.
- `BuildConfig.calibration_path: Path | None` consumes `calibration.json`; `None` retains the conservative one-session batch.

- [ ] **Step 1: Write complete-session packing tests**

```python
def test_pack_complete_sessions_never_splits_or_reorders():
    sessions = [
        {"id": "s1", "tokens": 900},
        {"id": "s2", "tokens": 1_100},
        {"id": "s3", "tokens": 2_500},
    ]

    batches = pack_complete_sessions(
        sessions,
        budget=2_100,
        request_tokens=lambda rows: sum(row["tokens"] for row in rows),
    )

    assert [[row["id"] for row in batch] for batch in batches] == [
        ["s1", "s2"],
        ["s3"],
    ]
```

Add a second test asserting that one oversized session is returned alone instead of split.

- [ ] **Step 2: Run packing tests and verify that the API is absent**

Run: `pytest -q code/tests/model_capacity/test_capacity.py`

Expected: FAIL because `src.runtime.capacity` does not exist.

- [ ] **Step 3: Implement the minimal token and capacity primitives**

```python
@dataclass(frozen=True)
class WriterCapacity:
    model: str
    prompt_hash: str
    safe_writer_request_tokens: int

    @classmethod
    def from_json(cls, path: Path) -> "WriterCapacity":
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = int(payload["safe_writer_request_tokens"])
        if value <= 0:
            raise ValueError("safe_writer_request_tokens must be positive")
        return cls(str(payload["model"]), str(payload["prompt_hash"]), value)
```

Implement greedy `pack_complete_sessions`; if the current batch is non-empty and adding the next session exceeds the budget, emit the current batch first. An oversized single session forms a one-session batch.

- [ ] **Step 4: Make Writer request rendering reusable**

Add `render_writer_task(sessions: list[dict]) -> str` to `management/api.py` and use it both for request estimation and `write_sessions`. Count the canonical JSON serialization of system message, user task, tool schema, and current workspace structure with `TokenCounter`; label it as a local estimate, not provider-exact usage.

- [ ] **Step 5: Integrate calibration with `BuildConfig`**

```python
@dataclass(frozen=True)
class BuildConfig:
    calibration_path: Path | None = None
    local_reorg_every_sessions: int = 5
    verify_writes: bool = True
    verify_every_sessions: int = 1
    final_manage: bool = True
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
```

When `calibration_path` is absent, process one complete session per Writer call. When present, load `WriterCapacity`, validate that its model and current Writer prompt hash match, then greedily pack complete sessions using its token budget.

- [ ] **Step 6: Write calibration-selection tests with a fake client**

Use two fixture samples and fake candidate outcomes:

```python
outcomes = {
    2_000: {"passed": True, "fact_recall": 1.0},
    4_000: {"passed": True, "fact_recall": 0.9},
    8_000: {"passed": False, "fact_recall": 0.6},
}

result = select_safe_capacity(outcomes)
assert result["safe_writer_request_tokens"] == 4_000
```

Also assert that a failing boundary is repeated once and that the output records calls, input/output tokens, elapsed seconds, selected session boundaries and failure reasons.

- [ ] **Step 7: Implement the calibration command**

The command accepts:

```text
--provider-config PATH
--samples PATH
--model MODEL
--output-dir PATH
--candidate-tokens 2000 4000 8000 16000
```

The provider config contains `base_url` and `api_key_file`; the key is read from that file and is never copied into results. Each candidate uses a fresh memory directory. Score only the curated required facts in `default_samples.json`; do not call or modify the LoCoMo evaluator.

- [ ] **Step 8: Run calibration and build tests**

Run:

```bash
pytest -q \
  code/tests/model_capacity \
  code/tests/management \
  code/tests/runtime \
  code/tests/scripts
```

Expected: PASS.

- [ ] **Step 9: Commit calibration**

```bash
git add code/src code/scripts/model_capacity code/scripts/configs/model_capacity code/tests
git commit -m "feat: calibrate Writer capacity by complete sessions"
```

### Task 6: Update the method and repository documentation

**Files:**
- Modify: `docs/method/nativemem-method.html`
- Modify: `docs/method/designs/memory_management_design.md`
- Modify: `docs/method/README.md`
- Modify: `README.md`
- Delete: `docs/method/versions/`
- Test: `code/tests/test_document_pages.py`

**Interfaces:**
- Documents: Writer capacity calibration as a deployment/configuration stage, not a fourth memory view or retrieval method.
- Documents: `src/`, `scripts/`, `tests/`, `results/`, `benchmarks/`, and `third_party/` responsibilities.

- [ ] **Step 1: Add documentation assertions**

```python
def test_method_documents_writer_capacity_calibration():
    text = (REPOSITORY_ROOT / "docs/method/nativemem-method.html").read_text(
        encoding="utf-8"
    )
    assert "Writer Capacity Calibration" in text
    assert "safe_writer_request_tokens" in text
    assert "完整 session" in text
    assert "新的记忆视图" in text
```

- [ ] **Step 2: Run the documentation test and verify the section is absent**

Run: `pytest -q code/tests/test_document_pages.py::test_method_documents_writer_capacity_calibration`

Expected: FAIL.

- [ ] **Step 3: Add the method section**

Place the section beside memory construction/runtime configuration. It must state:

```text
The calibration estimates effective Writer capacity under the current model,
Writer prompt and tool schema. It uses one or two small calibration samples,
tests increasing token budgets only at complete-session boundaries, and selects
the largest jointly passing budget before the first confirmed failure. Runtime
then greedily packs complete sessions under safe_writer_request_tokens. This is
a deployment/configuration procedure, not a new memory view or retrieval method.
```

Include the measured outputs: fact recall, format validity, calls, tokens, time, candidate boundaries and failure reason. Do not claim a raw context-window limit.

- [ ] **Step 4: Update code-layout documentation**

Document that `src/` contains core code, `scripts/` contains executable commands and configs, `tests/` contains tests, and `results/` contains run outputs. Remove active references to `code/experiments/`, version routing, and `nativemem_versions/v11`.

- [ ] **Step 5: Run document and link tests**

Run:

```bash
pytest -q code/tests/test_document_pages.py code/tests/test_portable_layout.py
python scripts/maintenance/verify_portable_layout.py
```

Expected: PASS with no missing local links.

- [ ] **Step 6: Commit documentation**

```bash
git add -A README.md docs code/tests/test_document_pages.py code/tests/test_portable_layout.py
git commit -m "docs: document Writer capacity calibration and code layout"
```

### Task 7: Verify and commit the canonical release state

**Files:**
- No planned source changes; this task verifies the files changed in Tasks 1–6.

**Interfaces:**
- Validates all outputs from Tasks 1–6.

- [ ] **Step 1: Verify the locked evaluator hash**

Run:

```bash
test "$(shasum -a 256 code/scripts/eval_full.py | awk '{print $1}')" = \
  "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"
```

Expected: exit status 0.

- [ ] **Step 2: Verify no obsolete code paths remain**

Run:

```bash
if rg -n "v8_memory|v10_memory|nativemem_versions|adapters\.run_nativemem" \
  code/src code/scripts code/tests --glob '*.py'; then
  exit 1
fi
test ! -e code/experiments
test ! -e experiments
```

Expected: no matches and exit status 0.

- [ ] **Step 3: Run the complete retained test suite**

Run: `pytest -q code/tests`

Expected: PASS.

- [ ] **Step 4: Run repository integrity checks**

Run:

```bash
python scripts/maintenance/verify_portable_layout.py
python -c "from src import BuildConfig, MemoryWorkspace, QueryConfig; print('imports ok')"
git diff --check
git status --short
```

Expected: layout check passes, import prints `imports ok`, diff check has no output, and status contains only intended final changes.

- [ ] **Step 5: Commit any final verification fixes**

```bash
git add -A
git commit -m "refactor: finalize canonical NativeMem source layout"
```
