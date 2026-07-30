# Experiment Runs

## 2026-06-29: R001 Trace Smoke

- Run ID: `R001_trace_smoke_qwen36_sample0_10`
- Purpose: validate trace schema and artifact layout before larger experiments.
- Script: `code/run_trace_smoke.py`
- Method: `S8 NativeMem`
- Memory source: existing `code/memory_test_v2/sample0_qwen3.6-flash`
- Dataset: `D1 LoCoMo-All`, sample 0 smoke split, 10 questions.
- Models: `qwen3.6-flash` retriever / answer / judge.
- Status: completed.

Outputs:

- Split: `artifacts/splits/locomo_sample0_smoke_10_v1.jsonl`
- Per-question output: `artifacts/per_question_outputs/D1/S8/build_smoke_existing_qwen36_sample0.jsonl`
- Retrieval traces: `artifacts/retrieval_traces/D1/S8/build_smoke_existing_qwen36_sample0/`
- Judge outputs: `artifacts/judge_outputs/qwen3.6-flash/D1/S8/build_smoke_existing_qwen36_sample0.jsonl`
- Summary table: `artifacts/tables/R001_trace_smoke_qwen36_sample0_10_summary.csv`
- Manifest: `artifacts/manifests/R001_trace_smoke_qwen36_sample0_10.sha256`

Result summary:

| Metric | Value |
|---|---:|
| Questions | 10 |
| Average LJ | 90.0 |
| Average retrieval calls | 3.8 |
| Average total visible tokens | 11,602 |
| Average total billable tokens | 13,334 |
| Categories covered | single-hop, temporal, multi-hop, open-domain, adversarial |
| Missing required output fields | 0 |

One failure was observed under the provisional category-5 treatment used in R001:

- `conv0_q153`, adversarial: question asks Melanie's summer adoption plans; gold answer is `researching adoption agencies`; system answered `NOT FOUND` after opening `Melanie.md`.

Interpretation:

- This run validates that the artifact structure and per-question accounting are usable.
- It should not be used as a formal performance result because it reuses an existing qwen-built memory folder and runs only 10 questions.
- The category-5 gold handling in this R001 note is superseded by R002, which follows the local LoCoMo adversarial scoring rule.

## 2026-06-29: R002 Formal Two-Stage Smoke

- Run IDs: `R002e_formal_s8_correctedgold_qwen36_sample0_10`, `R002f_formal_s6_correctedgold_topical_qwen36_sample0_10`
- Purpose: validate the `artifacts/` contract for both NativeMem and LLM-Topical Files under the formal two-stage protocol.
- Script: `code/run_trace_formal.py`
- Methods: `S8 NativeMem`, `S6 LLM-Topical Files`
- Dataset: `D1 LoCoMo-All`, sample 0 smoke split, 10 questions.
- Models: `qwen3.6-flash` retriever / answer / judge.
- Status: completed.

Protocol fixes made before the final R002 run:

- Retrieval and answering are separated: the retriever browses files and the answer model receives only bounded retrieved evidence.
- Final evidence is ordered by selector-mentioned paths before applying the 6K evidence cap.
- LoCoMo category 5 is scored with the official adversarial target: use `answer` when present; otherwise the correct reference is `Not mentioned in the conversation`, and `adversarial_answer` is treated as the distractor.
- Requests use a timeout so a single API call cannot block a run indefinitely.

Outputs:

- Split: `artifacts/splits/locomo_sample0_smoke_10_v1.jsonl`
- S8 per-question output: `artifacts/per_question_outputs/D1/S8/build_formal_correctedgold_existing_qwen36_sample0.jsonl`
- S6 per-question output: `artifacts/per_question_outputs/D1/S6/build_formal_correctedgold_topical_qwen36_sample0.jsonl`
- S8 retrieval traces: `artifacts/retrieval_traces/D1/S8/build_formal_correctedgold_existing_qwen36_sample0/`
- S6 retrieval traces: `artifacts/retrieval_traces/D1/S6/build_formal_correctedgold_topical_qwen36_sample0/`
- S8 judge outputs: `artifacts/judge_outputs/qwen3.6-flash/D1/S8/build_formal_correctedgold_existing_qwen36_sample0.jsonl`
- S6 judge outputs: `artifacts/judge_outputs/qwen3.6-flash/D1/S6/build_formal_correctedgold_topical_qwen36_sample0.jsonl`
- S8 summary table: `artifacts/tables/R002e_formal_s8_correctedgold_qwen36_sample0_10_summary.csv`
- S6 summary table: `artifacts/tables/R002f_formal_s6_correctedgold_topical_qwen36_sample0_10_summary.csv`
- S8 manifest: `artifacts/manifests/R002e_formal_s8_correctedgold_qwen36_sample0_10.sha256`
- S6 manifest: `artifacts/manifests/R002f_formal_s6_correctedgold_topical_qwen36_sample0_10.sha256`

Result summary:

| Method | Questions | Avg LJ | Avg retrieval calls | Avg total visible tokens | Avg total billable tokens | Missing required fields | Trace errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| S8 NativeMem | 10 | 100.0 | 4.3 | 12,581 | 22,579 | 0 | 0 |
| S6 LLM-Topical Files | 10 | 99.0 | 5.0 | 13,468 | 26,982 | 0 | 0 |

Interpretation:

- The formal artifact layout is now validated for two file-memory methods.
- This smoke split is too small and too easy to establish method superiority: S8 and S6 are nearly tied on LJ.
- The useful signal is protocol readiness and cost accounting. On this split, S8 uses fewer average retrieval calls and fewer average total visible / billable tokens than S6.
- The next discriminative run should use at least a balanced 100-question subset before moving to the 500-question budget sweep.
