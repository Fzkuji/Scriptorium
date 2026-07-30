# NativeMem Run Matrix and Target Figures

Date: 2026-06-29

This file is the execution-level specification for the NativeMem experiments. It defines the ideal paper figures, exact datasets, model roles, method rows, run counts, budgets, recorded fields, and stop/go gates.

## 0. What The Ideal Result Package Should Look Like

The final paper should have this result package. Do not treat these as obtained results; they are the target outputs to generate from audited runs.

| Target | Paper Location | Ideal visual / table | Required evidence | Success pattern |
|---|---|---|---|---|
| T1 | Main Table 1 | Main benchmark table with rows as methods and columns as LoCoMo-All, LoCoMo-Comparable, LongMemEval, BEAM-1M, BEAM-10M. | Full per-question outputs, shared judge outputs, bootstrap CIs, method provenance. | NativeMem is best or statistically tied for best; controlled file-system rows show a clear ladder. |
| T2 | Main Figure 2 | Total-visible-token curve: x-axis is total visible token cap, y-axis is LJ. One line per method. | LoCoMo-Balanced-500 budget sweep plus full-LoCoMo check at selected caps. | NativeMem curve is above LLM-Topical Files and LLM-Reranked Verbatim at the same token cap, especially at 6K and 10K. |
| T3 | Main Table 2 | Protocol robustness table: Final-6K, Total-visible-6K, Total-visible-10K, Total-visible-20K, Call-limit-5. | Same question set, same judge, same trace schema. | NativeMem remains ahead of file baselines under stricter budgets or reaches similar LJ with fewer total visible tokens. |
| T4 | Main Figure 3 | Factor ladder bar chart: Verbatim-BM25+LLM -> LLM-Reranked Verbatim -> LLM-Topical Files -> NativeMem-NoRSP -> NativeMem. | LoCoMo full, five builds for LLM file methods. | Each added factor has an interpretable non-negative increment; RSP and verification are smaller but stable increments. |
| T5 | Main Table 3 | 2x2 RSP x Verification ablation. | LoCoMo full and LongMemEval full; five builds if used in main paper. | RSP and verification each improve score; combined method is best. |
| T6 | Main / Appendix Figure | Path diagnostic: first-path-hit versus LJ, plus writer/retriever mismatch matrix. | Retrieval traces with first content-bearing file and target evidence files. | Higher first-path hit correlates with higher LJ; mismatched writer/retriever degrades predictably. |
| T7 | Appendix Table | Cost and artifact audit table. | Generated memory folders, traces, per-question outputs, scripts, configs, checksums. | Every aggregate number can be regenerated from trace files. |
| T8 | Appendix Table | Human validation table on LoCoMo-Balanced-100. | Two human annotators, blind method labels, adjudication log. | Human ranking agrees with LJ; report Pearson, Spearman, and binned agreement. |

## 1. Dataset Inventory

Use two LoCoMo views because local artifacts contain both all categories and the memory-benchmarks comparable subset.

| Dataset ID | Source file / source | Size | Categories / types | Use |
|---|---|---:|---|---|
| D1: LoCoMo-All | `code/locomo/data/locomo10.json` | 10 conversations, 1,986 QA | cat1 282, cat2 321, cat3 96, cat4 841, cat5 446 | Main NativeMem evaluation, ablations, failure analysis. |
| D2: LoCoMo-Comparable | `code/memory-benchmarks/results/platform/locomo_results.json` metadata | 1,540 QA | categories 1-4, excludes adversarial cat5 | Comparison with memory-benchmarks / platform-style baselines. |
| D3: LoCoMo-Balanced-500 | Derived from D1 | 500 QA | 100 per category where possible; cat3 has only 96, so use 96 cat3 and distribute 4 to cat5 | Budget sweep and call-limit curves. |
| D4: LoCoMo-Human-100 | Derived from D1 | 100 QA | 20 per category | Human validation. |
| D5: LongMemEval-Full | `code/longmemeval/data/longmemeval_s_cleaned.json` | 500 QA | KU 78, MS 133, SSA 56, SSP 30, SSU 70, TR 133 | Main generalization benchmark. |
| D6: BEAM-1M | `code/memory-benchmarks/results/platform/beam_1m_results.json` metadata | 700 QA | 10 question types, 1M chat size | Scale stress. |
| D7: BEAM-10M | `code/memory-benchmarks/results/platform/beam_10m_results.json` metadata | 200 QA | 10 question types, 10M chat size | Extreme-scale stress, appendix if cost is high. |

Sampling rule for derived subsets:

- Use fixed JSONL files under `artifacts/splits/`.
- Store `split_id`, `source_dataset`, `question_id`, `conversation_id`, category/type, and SHA-256 checksum.
- Do not resample after first use. If a split changes, create a new split id.

LoCoMo category-5 scoring rule:

- Category 5 is adversarial / unanswerable unless the QA object has an explicit `answer` field.
- If `answer` is present, use `answer` as the reference.
- If `answer` is absent, the reference is `Not mentioned in the conversation`; `adversarial_answer` is a distractor and must not be used as the gold answer.
- Report category-5 metrics separately because this rule differs from ordinary short-answer scoring.

## 2. Model Inventory

| Model ID | Role | Required? | Where used | Notes |
|---|---|---|---|---|
| M1: GPT-5.5 | Primary writer, retriever, answerer, judge | MUST | NativeMem, LLM-operated baselines, primary LJ | This is the ideal paper's main model. |
| M2: GPT-4.1 | Secondary judge | MUST | Judge independence | Judge only unless API budget permits more. |
| M3: Claude-4.5 | Secondary judge | MUST | Judge independence | Judge only. |
| M4: GPT-4o-mini | Cross-model retriever/writer and legacy comparability | SHOULD | Writer/retriever mismatch, older baseline comparison | Useful because many earlier memory papers use GPT-4o-mini. |
| M5: Qwen-2.5-72B-Instruct | Open-weight scope test | SHOULD | Open model NativeMem run and cross-model matrix | Report as scope evidence, not as main result if below GPT-5.5. |
| M6: Qwen3.6-flash | Smoke-test / historical local run | OPTIONAL | Cheap pipeline checks only | Do not use as primary paper evidence. |
| M7: Baseline-native models | Baseline reproduction | MUST when using native baseline rows | Mnemis, TiMem, Mem0, memory-benchmarks outputs | Record exact model string from baseline config. |

Training policy:

- NativeMem has zero training runs.
- LLM-Reranked Verbatim, LLM-Topical Files, NativeMem-NoRSP, and ablations also have zero training runs.
- Learned-policy baselines such as DeltaMem/HORMA should use released checkpoints or reported outputs unless a separate reproduction plan is created. Do not mix newly trained learned baselines with no-training NativeMem rows without provenance notes.
- The repeated stochastic unit for NativeMem is `memory_build_id`, not training seed.

## 3. Method Rows

Main self-run rows:

| Method ID | Method | Builds / seeds | Datasets | Required for |
|---|---|---:|---|---|
| S1 | Raw Transcript / Truncation | 1 deterministic run | D1, D5, D6, D7 where feasible | Lower anchor. |
| S2 | BM25 chunks | 1 deterministic run | D1, D5, D6, D7 | Sparse retrieval anchor. |
| S3 | Chrono Markdown | 1 deterministic run | D1, D5 | Simple structured baseline. |
| S4 | Verbatim-BM25+LLM | 1 deterministic run | D1, D5, D6, D7 | Copy-only plus BM25 baseline. |
| S5 | LLM-Reranked Verbatim | 1 deterministic run at temperature 0 | D1, D5, D6, D7 | Strong non-file LLM retrieval baseline. |
| S6 | LLM-Topical Files | 5 memory builds for main tables; 3 builds for robustness sweeps | D1, D5, D6, D7 | Strong file-system baseline. |
| S7 | NativeMem-NoRSP | 5 memory builds for main tables; 3 builds for robustness sweeps | D1, D5, D6, D7 | Separates RSP from file substrate and repair. |
| S8 | NativeMem | 5 memory builds for main tables; 3 builds for robustness sweeps | D1, D5, D6, D7 | Final method. |

Native/provenance rows:

| Method ID | Method | Runs | Datasets | Provenance rule |
|---|---|---:|---|---|
| B1 | Mnemis | 1 native run or official prediction set | D1, D5 | Mark as native rerun, adapter rerun, or shared-judge rescoring. |
| B2 | TiMem | 1 native run or official prediction set | D1, D5 | Same provenance rule. |
| B3 | Mem0 / memory-benchmarks platform | 1 native run or official output | D2, D5, D6, D7 | Use D2 when platform output excludes adversarial category. |
| B4 | A-Mem | 1 native run if stable | D1 | Appendix if incomplete. |

Ablation rows:

| Method ID | Ablation | Builds | Dataset | Main reason |
|---|---|---:|---|---|
| A1 | w/o Evidence Copy | 3 builds | D1, D5 if budget allows | Tests content fidelity. |
| A2 | w/o Verification | 5 builds if in main 2x2 table | D1, D5 | Tests repair loop. |
| A3 | RSP off, Verification off | 5 builds if in main 2x2 table | D1, D5 | Required 2x2 cell. |
| A4 | RSP on, Verification off | 5 builds if in main 2x2 table | D1, D5 | Required 2x2 cell. |
| A5 | Random Placement | 3 builds | D1 | Placement lower bound. |
| A6 | Chronological Placement | 3 builds | D1 | Temporal organization baseline. |
| A7 | Abstract Names | 3 builds | D1 | Naming specificity test. |
| A8 | No Cross-Links | 3 builds | D1 | Link utility test. |
| A9 | Flat Single File | 1-3 builds | D1 | Structure deletion study. |

## 4. Fixed Parameters

| Parameter | Main setting | Robustness settings | Notes |
|---|---|---|---|
| Final evidence cap | 6K tokens | 2K, 4K, 6K, 10K, 20K in budget sweep | Applies to final answer context. |
| Intermediate observation cap | 20K tokens | Included inside total-visible cap for robustness | Applies to file-browsing observations. |
| Total-visible cap | Not used in main table | 4K, 6K, 10K, 20K on D3; 6K, 10K, 20K on D1 check | `retrieval observations + final evidence`. |
| Tool-call limit | No hard main limit except safety max 15 rounds | 3, 5, 8 retrieval turns on D3; call-limit-5 on D1 check | Count only retrieval tool-call rounds before final answer. |
| Reranker candidate pool | 50 chunks | 50 fixed | Record reranker-visible tokens. |
| Verbatim chunk size | 512 tokens | 512 fixed | Use 64-token overlap unless official baseline requires otherwise. |
| Chunk overlap | 64 tokens | 64 fixed | Record tokenizer and chunk hash. |
| Retrieval temperature | 0 | 0 | Deterministic retrieval and reranking. |
| Answer temperature | 0 | 0 | Shared answerer. |
| Judge temperature | 0 | 0 | Primary and secondary judges. |
| Memory construction temperature | 0.2 | 0.2 | Applies to LLM-operated file-memory methods. |
| Max write/repair output | 2,048 tokens | 2,048 tokens | Match paper appendix. |
| Max answer output | 1,024 tokens | 1,024 tokens | Match paper appendix. |
| Builds for main LLM file methods | 5 | 3 for budget sweeps | Final main table uses 5. |

## 5. Required Recorded Fields

Each question-level JSON record must include:

```json
{
  "run_id": "R003",
  "dataset_id": "D1",
  "split_id": "locomo_all_v1",
  "question_id": "conv0_q128",
  "conversation_id": "conv0",
  "category_or_type": "temporal",
  "method_id": "S8",
  "method_name": "NativeMem",
  "build_id": "build_03",
  "writer_model": "gpt-5.5",
  "retriever_model": "gpt-5.5",
  "answer_model": "gpt-5.5",
  "judge_model": "gpt-5.5",
  "tokenizer": "o200k_base_or_pinned_equivalent",
  "final_evidence_cap": 6000,
  "intermediate_observation_cap": 20000,
  "total_visible_cap": null,
  "tool_call_limit": null,
  "retrieval_calls": 0,
  "tool_calls": [],
  "files_opened": [],
  "first_content_file": null,
  "target_evidence_files": [],
  "first_path_hit": null,
  "retrieval_visible_tokens": 0,
  "final_evidence_tokens": 0,
  "answer_prompt_tokens": 0,
  "answer_completion_tokens": 0,
  "judge_prompt_tokens": 0,
  "judge_completion_tokens": 0,
  "total_visible_tokens": 0,
  "total_billable_tokens": 0,
  "storage_amortized_tokens": 0,
  "latency_retrieval_s": 0.0,
  "latency_answer_s": 0.0,
  "latency_judge_s": 0.0,
  "gold_answer": "",
  "system_answer": "",
  "judge_score": 0,
  "f1": 0.0,
  "bleu1": 0.0,
  "failure_tag": null,
  "trace_path": "retrieval_traces/D1/S8/build_03/conv0_q128.json",
  "memory_folder_sha256": "",
  "config_sha256": ""
}
```

Dataset-level / run-level metadata must include:

- model exact names and provider endpoints;
- prompt versions and prompt hashes;
- tokenizer version;
- random seed / build id;
- git commit hash for our code and baseline code;
- baseline provenance: `official_rerun`, `adapter_rerun`, `shared_judge_rescore`, or `reported_only`;
- cost accounting rule;
- excluded questions and reasons, if any.

## 6. Run Counts

The counts below are the final target. Staging runs may use fewer builds, but paper tables should not use staging-only results.

### 6.1 Main LoCoMo

Dataset: D1 LoCoMo-All, 1,986 QA.

| Family | Methods | Builds | Question-level records |
|---|---:|---:|---:|
| Deterministic anchors | S1-S5 | 1 each | 5 * 1,986 = 9,930 |
| LLM file methods | S6-S8 | 5 each | 3 * 5 * 1,986 = 29,790 |
| Native/provenance baselines | B1-B3 | 1 each if available | 3 * 1,986 = 5,958 |
| Main total target | 11 method rows | mixed | 45,678 records |

Also run D2 LoCoMo-Comparable for platform-style rows when the baseline excludes adversarial questions:

- S4-S8 on D2: `5 deterministic/file rows`, with S6-S8 using 5 builds.
- B3 Mem0 / memory-benchmarks platform on D2.
- Purpose: fair comparison to platform results with 1,540 questions.

### 6.2 Main LongMemEval

Dataset: D5 LongMemEval-Full, 500 QA.

| Family | Methods | Builds | Question-level records |
|---|---:|---:|---:|
| Deterministic anchors | S2, S4, S5 | 1 each | 3 * 500 = 1,500 |
| LLM file methods | S6-S8 | 5 each | 3 * 5 * 500 = 7,500 |
| Native/provenance baselines | B1-B3 | 1 each if available | 3 * 500 = 1,500 |
| Main total target | 9 method rows | mixed | 10,500 records |

### 6.3 BEAM

Datasets: D6 BEAM-1M with 700 QA; D7 BEAM-10M with 200 QA.

| Dataset | Required methods | Builds | Question-level records |
|---|---|---:|---:|
| D6 BEAM-1M | S2, S4, S5, S6, S7, S8, B3 | S6-S8 use 3 builds first; 5 if final main table uses BEAM | 1-build target: 4 * 700 + 3 * 3 * 700 = 9,100; 5-build target: 13,300 |
| D7 BEAM-10M | S2, S4, S5, S6, S7, S8, B3 | Same as D6 | 1-build target: 4 * 200 + 3 * 3 * 200 = 2,600; 5-build target: 3,800 |

Use BEAM-10M as appendix unless cost and runtime are stable.

### 6.4 Protocol Robustness

Use two levels:

1. Full sweep on D3 LoCoMo-Balanced-500.
2. Selected check on D1 LoCoMo-All.

| Setting | Dataset | Methods | Builds | Caps / limits | Records |
|---|---|---|---:|---|---:|
| Total-visible sweep | D3 | S5-S8 | S6-S8 use 3 builds; S5 one run | 4K, 6K, 10K, 20K | `(1 + 3*3) * 4 * 500 = 20,000` |
| Call-limit sweep | D3 | S6-S8 | 3 builds each | 3, 5, 8 turns | `3 * 3 * 3 * 500 = 13,500` |
| Full check | D1 | S5-S8 | S6-S8 use 3 builds; S5 one run | total-visible 6K, 10K, 20K; call-limit 5 | `((1+3*3)*3 + 3*3) * 1,986 = 77,454` |

If the full check is too expensive, keep D3 curves in main paper and move D1 selected check to appendix with a smaller balanced 1,000-question split. If doing that, state the subset explicitly.

### 6.5 Mechanism Ablations

| Table | Dataset | Cells / rows | Builds | Records |
|---|---|---|---:|---:|
| 2x2 RSP x Verification | D1 | 4 cells | 5 each | `4 * 5 * 1,986 = 39,720` |
| 2x2 RSP x Verification | D5 | 4 cells | 5 each | `4 * 5 * 500 = 10,000` |
| Additional ablations | D1 | A1, A5-A9 | 3 each except flat can be 1 if deterministic | about `6 * 3 * 1,986 = 35,748` |

The 2x2 table is more important than every extra ablation. If compute must be reduced, keep the 2x2 and drop additional ablations to appendix.

### 6.6 Model Scope

| Experiment | Dataset | Models | Builds | Records |
|---|---|---|---:|---:|
| Writer/retriever mismatch | D4 or a separate LoCoMo-100 balanced split | writers M1/M4/M5 x retrievers M1/M4/M5 | 3 writer builds each | `3 writers * 3 builds * 3 retrievers * 100 = 2,700` |
| Open-weight full method | D1 and D5 if budget allows | M5 for writer/retriever/answer, M1 judge | 3 builds | `3 * 1,986 + 3 * 500 = 7,458` |
| Judge independence | Main outputs only | M1/M2/M3 judges | no rerun of retrieval | `number_of_final_answers * 3 judges` |

## 7. Output Directory Contract

Use this artifact layout:

```text
artifacts/
  configs/{run_id}.json
  splits/{split_id}.jsonl
  memory_folders/{dataset_id}/{method_id}/{build_id}/
  retrieval_traces/{dataset_id}/{method_id}/{build_id}/{question_id}.json
  per_question_outputs/{dataset_id}/{method_id}/{build_id}.jsonl
  judge_outputs/{judge_model}/{dataset_id}/{method_id}/{build_id}.jsonl
  tables/{table_id}.csv
  figures/{figure_id}.pdf
  manifests/{run_id}.sha256
```

Every table and figure should have a script path in `artifacts/configs/{run_id}.json`.

## 8. Stop / Go Gates

| Gate | After runs | Continue if | Change plan if |
|---|---|---|---|
| G1 | R001-R002 | Trace schema captures all required cost fields on one sample. | Cost fields cannot distinguish retrieval observations from final evidence. Fix instrumentation first. |
| G2 | R003-R007 on LoCoMo first 500 QA | NativeMem beats LLM-Topical Files and NativeMem-NoRSP by at least a small consistent margin. | If not, reduce claim to file-memory system and do not center RSP. |
| G3 | Full LoCoMo main table | NativeMem is best or tied under primary judge and controlled rows are stable across builds. | If only judge-specific, prioritize human validation and judge independence before writing claims. |
| G4 | D3 protocol robustness | NativeMem remains above baselines at 6K or 10K total-visible cap, or reaches the same score with fewer tokens. | If not, state that the method requires interactive browsing budget and narrow the cross-system claim. |
| G5 | LongMemEval full | NativeMem is above or competitive with strong baselines and has clear update/multi-session category behavior. | If below Mnemis/Mem0, treat LME as limitation and emphasize LoCoMo/controlled mechanism. |
| G6 | Human validation | Human rankings agree with LJ enough to support LJ as primary metric. | If not, move LJ to secondary and report human subset more prominently. |

## 9. Minimum Submission Package

If schedule is limited, the minimum defensible package is:

- D1 LoCoMo-All main table for S4-S8 and B1/B2 if available.
- D5 LongMemEval-Full for S5-S8 and B1 if available.
- D3 LoCoMo-Balanced-500 total-visible-token curve.
- D1 2x2 RSP x Verification table.
- D4 human validation with 100 questions.
- Full artifact package for every row used in a main table.

BEAM and additional ablations can move to appendix or later work if the above package is not stable.
