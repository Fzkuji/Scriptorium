# Experiment Tracker

Detailed dataset/model/method definitions are in `refine-logs/RUN_MATRIX.md`.

| Run ID | Milestone | Purpose | System / Variant | Dataset / Split | Builds / Runs | Primary Outputs | Priority | Status | Notes |
|---|---|---|---|---|---:|---|---|---|---|
| R001 | M0 | Freeze trace schema and per-question accounting. | Shared runner/instrumentation | D1 sample 0 smoke split | 1 method x 10 questions completed | trace schema, cost fields, checksums | MUST | DONE | `R001_trace_smoke_qwen36_sample0_10`; NativeMem existing qwen memory; 10/10 records have required fields. |
| R002 | M0 | Validate artifact layout. | NativeMem + LLM-Topical Files | D1 sample 0 smoke split | 2 methods x 1 build completed | artifact tree and manifest | MUST | DONE | Final corrected-gold runs: `R002e_formal_s8_correctedgold_qwen36_sample0_10`, `R002f_formal_s6_correctedgold_topical_qwen36_sample0_10`; required fields missing: 0; S8 LJ 100.0, S6 LJ 99.0; smoke split is not discriminative enough for method-superiority claims. |
| R003 | M1 | Main final method on LoCoMo. | S8 NativeMem | D1 LoCoMo-All 1,986 QA | 5 builds | LJ/F1/BLEU, trace files, cost table | MUST | TODO | Main table row. |
| R004 | M1 | Remove retrieval-simulated placement. | S7 NativeMem-NoRSP | D1 LoCoMo-All 1,986 QA | 5 builds | LJ, first-path hit, cost fields | MUST | TODO | Same file tools and repair loop as NativeMem. |
| R005 | M1 | Strong file baseline without RSP or repair. | S6 LLM-Topical Files | D1 LoCoMo-All 1,986 QA | 5 builds | LJ, cost fields | MUST | TODO | Diagnostic row for file-substrate value. |
| R006 | M1 | Strong verbatim baseline. | S5 LLM-Reranked Verbatim | D1 LoCoMo-All 1,986 QA | 1 run | LJ, reranker-visible tokens, final evidence tokens | MUST | TODO | Candidate pool 50 chunks. |
| R007 | M1 | Basic copy baseline. | S4 Verbatim-BM25+LLM | D1 LoCoMo-All 1,986 QA | 1 run | LJ, final evidence tokens | MUST | TODO | Lower anchor for factor ladder. |
| R008 | M1 | Deterministic lower anchors. | S1-S3 Raw/BM25/Chrono | D1 LoCoMo-All 1,986 QA | 1 run each | LJ/F1/BLEU | SHOULD | TODO | Main or appendix depending on space. |
| R009 | M2 | Total-visible-token curve. | S5-S8 | D3 LoCoMo-Balanced-500 | S6-S8 3 builds, S5 1 run, 4 caps | T2 curve | MUST | TODO | Caps: 4K, 6K, 10K, 20K. |
| R010 | M2 | Call-limit curve. | S6-S8 | D3 LoCoMo-Balanced-500 | 3 methods x 3 builds x 3 limits | T3 support | MUST | TODO | Limits: 3, 5, 8 retrieval turns. |
| R011 | M2 | Full-LoCoMo robustness check. | S5-S8 | D1 LoCoMo-All 1,986 QA | S6-S8 3 builds, selected caps | T3 table | MUST | TODO | Caps 6K/10K/20K and call-limit 5. |
| R012 | M3 | Baseline-native operating point. | B1 Mnemis | D1 and D5 if available | 1 native/provenance run | shared-judge score, provenance | MUST | TODO | Use local outputs before rerun. |
| R013 | M3 | Baseline-native operating point. | B2 TiMem | D1 and D5 if available | 1 native/provenance run | shared-judge score, provenance | SHOULD | TODO | Keep separate if adapter cannot reproduce native behavior. |
| R014 | M3 | Platform baseline. | B3 Mem0 / memory-benchmarks | D2, D5, D6, D7 | 1 native/provenance run | native score, shared-judge score if predictions available | SHOULD | TODO | D2 has 1,540 comparable LoCoMo questions. |
| R015 | M4 | LongMemEval final method. | S8 NativeMem | D5 LongMemEval-Full 500 QA | 5 builds | LJ/category LJ/cost | MUST | TODO | Replaces sampled-only status. |
| R016 | M4 | LongMemEval controlled baselines. | S5-S7 | D5 LongMemEval-Full 500 QA | S6-S7 5 builds, S5 1 run | factor comparison and table rows | MUST | TODO | Required before claiming LME superiority. |
| R017 | M4 | LongMemEval 2x2 ablation. | RSP x Verification cells | D5 LongMemEval-Full 500 QA | 4 cells x 5 builds | T5 table | MUST | TODO | Can move to appendix if LoCoMo 2x2 is enough for main. |
| R018 | M5 | Scale stress. | S2/S4/S5/S6/S7/S8/B3 | D6 BEAM-1M 700 QA | file methods 3 builds first | scale table and cost | SHOULD | TODO | Repeat to 5 builds only if used in main table. |
| R019 | M5 | Extreme-scale stress. | S2/S4/S5/S6/S7/S8/B3 | D7 BEAM-10M 200 QA | file methods 3 builds first | appendix scale table | NICE | TODO | High cost; do not block core submission. |
| R020 | M6 | Human validation. | Final top methods only | D4 LoCoMo-Human-100 | 2 blind annotators | human/LJ correlation and agreement | MUST | TODO | 20 questions per category. |
| R021 | M6 | Writer/retriever mismatch. | M1/M4/M5 writer x M1/M4/M5 retriever | LoCoMo-100 balanced split | 3 builds per writer | 3x3 matrix, first-path hit | SHOULD | TODO | Total 2,700 question records. |
| R022 | M6 | Open-weight scope test. | M5 NativeMem | D1 and D5 | 3 builds | open-model score and path hit | SHOULD | TODO | Limitation/scope evidence. |
| R023 | M6 | Online maintenance pilot. | NativeMem with and without old-evidence verification | small interleaved update split | 3 builds | stale-path failure, update accuracy | NICE | TODO | Only needed if broad online-memory claim stays in title/intro. |
