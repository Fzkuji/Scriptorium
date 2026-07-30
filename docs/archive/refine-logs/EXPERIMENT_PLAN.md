# Experiment Plan

**Problem**: Reviewers can accept the NativeMem direction only if the paper shows that the gains do not mainly come from a more permissive file-browsing protocol, a favorable judge, or a single strong model.
**Method Thesis**: A large language model can maintain long-term memory as ordinary files by copying evidence, placing it according to future retrieval behavior, browsing it with file tools, and repairing newly written evidence after reachability failures.
**Date**: 2026-06-29

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: NativeMem improves long-term QA through an LLM-native file memory lifecycle. | This is the paper's main contribution; it separates the work from vector, graph, and trained-policy memory systems. | NativeMem beats strong reranked, topical-file, and representative published baselines on LoCoMo and LongMemEval under a transparent shared protocol, with per-question traces and independent judging. | B1, B3, B5 |
| C2: The gain is not an artifact of extra retrieval context or extra tool calls. | This is the main rejection risk from the latest independent review. | NativeMem remains competitive under total-visible-token caps, call limits, and baseline-native operating points; cost tables include retrieval-visible tokens, final evidence tokens, calls, latency, and storage cost. | B2, B4 |
| C3: Retrieval-simulated placement and verification are measurable mechanisms, not only implementation details. | This protects the paper from being read as only "LLMs write markdown files." | A factor ladder and a 2x2 RSP by verification ablation show consistent increments over LLM-Topical Files and NativeMem-NoRSP; path diagnostics correlate with answer quality. | B3, B5 |
| C4: The approach has bounded generality. | The current evidence is offline build-then-query QA; overclaiming online memory maintenance is risky. | Cross-model writer/retriever tests and a small online update benchmark are reported as scope tests or limitations, not as the primary claim unless results are strong. | B5 |

## Paper Storyline

Main paper must prove:

- NativeMem is a complete LLM-native storage, retrieval, and repair lifecycle without embedding databases, graph stores, or trained memory policies.
- The strongest evidence is the controlled file-system comparison: LLM-Reranked Verbatim, LLM-Topical Files, NativeMem-NoRSP, and NativeMem.
- Cross-system baselines should be framed as a transparent common operating point, not as fully compute-matched proof by themselves.
- Protocol robustness must be visible in the main paper because reviewer risk is concentrated there.

Appendix can support:

- Full per-category tables, lexical metrics, judge-independence tables, build-to-build variance, additional placement variants, trace examples, and artifact manifests.
- Wider baseline lists and older citation-driven comparisons.

Experiments intentionally cut or deprioritized:

- Large online agent deployments before the offline memory claim is stable.
- Many weak baselines that only pad the main table.
- Claims that NativeMem is low-latency or model-agnostic without corresponding evidence.

## Target Result Package

The concrete run matrix, target figures, model list, dataset sizes, build counts, budget settings, recorded fields, and stop/go gates are specified in `refine-logs/RUN_MATRIX.md`.

The intended final paper result package is:

| Target | Paper output | Dataset / runs | What it should show |
|---|---|---|---|
| T1 | Main benchmark table | LoCoMo-All 1,986 QA, LoCoMo-Comparable 1,540 QA, LongMemEval 500 QA, BEAM-1M 700 QA, BEAM-10M 200 QA | NativeMem is best or statistically tied for best under an auditable shared protocol. |
| T2 | Total-visible-token curve | LoCoMo-Balanced-500, methods S5-S8, caps 4K/6K/10K/20K | NativeMem is not only winning because it can inspect more intermediate context. |
| T3 | Protocol robustness table | Full LoCoMo check at total-visible 6K/10K/20K and call-limit 5 | Main conclusion survives stricter budget settings. |
| T4 | Factor ladder | LoCoMo full, five builds for LLM file methods | The improvement decomposes into copied evidence, LLM reranking, file organization, verification, and RSP. |
| T5 | RSP x Verification 2x2 | LoCoMo full and LongMemEval full, five builds | RSP and verification each add measurable value. |
| T6 | Path diagnostics | LoCoMo traces and writer/retriever mismatch subset | First-path hit explains part of the answer-quality gain. |
| T7 | Artifact audit | All main-table rows | Every aggregate result can be regenerated from per-question traces. |
| T8 | Human validation | LoCoMo-Human-100, 20 questions per category | LJ agrees with blind human scoring enough to serve as primary metric. |

## Execution Specification

Main datasets:

| Dataset ID | Size | Role |
|---|---:|---|
| D1 LoCoMo-All | 1,986 QA over 10 conversations | Main offline persistent-QA benchmark. |
| D2 LoCoMo-Comparable | 1,540 QA, categories 1-4 | Comparison with memory-benchmarks/platform rows that exclude adversarial questions. |
| D3 LoCoMo-Balanced-500 | 500 QA | Budget and call-limit curves. |
| D4 LoCoMo-Human-100 | 100 QA | Human validation. |
| D5 LongMemEval-Full | 500 QA | Generalization to longer conversations and update categories. |
| D6 BEAM-1M | 700 QA | Scale stress. |
| D7 BEAM-10M | 200 QA | Extreme-scale appendix result unless stable enough for main paper. |

Main models:

| Model ID | Role |
|---|---|
| M1 GPT-5.5 | Primary writer, retriever, answerer, and judge. |
| M2 GPT-4.1 | Secondary judge. |
| M3 Claude-4.5 | Secondary judge. |
| M4 GPT-4o-mini | Cross-model writer/retriever and legacy comparability. |
| M5 Qwen-2.5-72B-Instruct | Open-weight scope test. |
| M6 Qwen3.6-flash | Smoke-test and historical result only. |
| M7 baseline-native models | Exact baseline model strings from Mnemis, TiMem, Mem0, or memory-benchmarks configs. |

Training and repetition:

- NativeMem and all controlled file/text baselines require zero training.
- LLM-operated file methods use five independent memory builds for main tables.
- Protocol robustness sweeps use three builds for file methods, then repeat to five only if the variance changes the conclusion.
- Deterministic chunk, BM25, chrono, and reranked baselines run once at temperature 0 unless a stochastic component is introduced.
- Learned-policy baselines use released checkpoints or official predictions unless a separate reproduction plan is created.

Core parameter grid:

| Parameter | Main setting | Robustness settings |
|---|---|---|
| Final evidence cap | 6K tokens | 2K/4K/6K/10K/20K where needed |
| Intermediate observation cap | 20K tokens | Included in total-visible cap |
| Total-visible cap | Not used in main table | 4K/6K/10K/20K on LoCoMo-Balanced-500; 6K/10K/20K on full LoCoMo check |
| Tool-call limit | Safety max 15 retrieval rounds | 3/5/8 turns on LoCoMo-Balanced-500; call-limit 5 on full LoCoMo check |
| Reranker candidate pool | 50 chunks | fixed |
| Verbatim chunk size / overlap | 512 / 64 tokens | fixed |
| Retrieval, answer, judge temperature | 0 | fixed |
| Memory construction temperature | 0.2 | fixed |

Every question-level output must record at least: dataset id, question id, category/type, method id, build id, writer/retriever/answer/judge model ids, token caps, tool-call limit, retrieval calls, ordered tool calls, files opened, first content file, target evidence files, first-path hit, retrieval-visible tokens, final evidence tokens, answer tokens, judge tokens, total visible tokens, total billable tokens, latency, gold answer, system answer, LJ, F1, BLEU-1, failure tag, trace path, memory-folder checksum, and config checksum.

## Experiment Blocks

### Block 1: Unified Main Evaluation

- Claim tested: C1.
- Why this block exists: Establish the main performance result under one auditable protocol.
- Dataset / split / task: LoCoMo full 1,986 QA; LongMemEval full 500 QA; BEAM-1M and BEAM-10M if code and data are available from the local `memory-benchmarks` checkout.
- Compared systems: NativeMem, NativeMem-NoRSP, LLM-Topical Files, LLM-Reranked Verbatim, Verbatim-BM25+LLM, BM25/Chrono, and the strongest available published baselines with runnable outputs or predictions.
- Metrics: Primary LJ with GPT-5.5 judge; secondary F1, BLEU, accuracy threshold, judge agreement on a human subset.
- Setup details: Same final answer model and judge where possible; 6K final evidence cap; five independent builds for LLM-operated file-memory methods; bootstrap intervals and paired permutation tests.
- Success criterion: NativeMem is first or statistically tied for first on LoCoMo and LongMemEval, and the controlled ladder shows consistent gains over LLM-Topical Files and NativeMem-NoRSP.
- Failure interpretation: If only cross-system margins hold but controlled baselines do not, the paper should be reframed as a system/protocol result rather than a method contribution.
- Table / figure target: Main Table 1, factor ladder table, category table in appendix.
- Priority: MUST-RUN.

### Block 2: Protocol Robustness and Cost Matching

- Claim tested: C2.
- Why this block exists: Directly addresses the reviewer concern that file-browsing methods can inspect up to 20K intermediate tokens while chunk/vector methods are constrained mainly by final evidence.
- Dataset / split / task: LoCoMo full first; LongMemEval after LoCoMo is stable.
- Compared systems: NativeMem, LLM-Topical Files, NativeMem-NoRSP, LLM-Reranked Verbatim, TiMem or Mnemis where runnable, and Mem0 platform results if local reproduction is practical.
- Metrics: LJ, retrieval-visible tokens, final evidence tokens, total visible tokens, LLM retrieval calls, judge calls, latency, amortized storage tokens, LJ per 1K visible tokens.
- Setup details:
  - Final-6K: current main protocol.
  - Total-visible-6K/10K/20K: retrieval observations plus final evidence are capped together.
  - Call-limited: file-browsing methods get 3, 5, and 8 retrieval tool-call turns.
  - Native-operating-point: each strong baseline uses its recommended retrieval setting, followed by the shared judge.
- Success criterion: NativeMem remains ahead of LLM-Topical Files and LLM-Reranked Verbatim under total-token and call-limited settings, or reaches the same LJ with fewer total visible tokens.
- Failure interpretation: If NativeMem only wins with high intermediate browsing budget, main paper should narrow the claim to an interactive file-browsing operating point.
- Table / figure target: New main robustness table or compact curve; full sweep in appendix.
- Priority: MUST-RUN.

### Block 3: Mechanism Isolation

- Claim tested: C3.
- Why this block exists: Shows what part of NativeMem matters and prevents the paper from relying only on a large main table.
- Dataset / split / task: LoCoMo full; LongMemEval full if budget allows; BEAM for scale-specific ablations only after core results are stable.
- Compared systems: Verbatim-BM25+LLM, LLM-Reranked Verbatim, LLM-Topical Files, NativeMem-NoRSP, NativeMem; plus flat, random placement, chronological placement, abstract names, no cross-links, no copy, no verification.
- Metrics: LJ, first-path hit, files opened, total visible tokens, answer failure tags.
- Setup details: Same memory content where possible; change one factor per row; five builds for stochastic LLM file methods.
- Success criterion: File structure and copied evidence explain the large jump; RSP and verification add smaller but consistent increments; path-hit improvements align with LJ improvements.
- Failure interpretation: If RSP is unstable, the paper should make LLM-native file memory the primary claim and keep RSP as a useful but nonessential placement heuristic.
- Table / figure target: Main factor ladder, main 2x2 table, path-diagnostic figure, appendix ablation table.
- Priority: MUST-RUN.

### Block 4: Reproducibility and Audit Package

- Claim tested: C1 and C2.
- Why this block exists: The reported ideal paper depends on reviewers trusting that the unified protocol was implemented consistently.
- Dataset / split / task: All datasets used in the main tables.
- Compared systems: Every method included in the main table and every controlled ablation.
- Metrics: Not a performance block; audit completeness is the outcome.
- Setup details: Save generated memory folders, ordered retrieval traces, per-question outputs, judge inputs/outputs, token counts, config files, random seeds, baseline provenance, and checksums.
- Success criterion: Any row in a table can be regenerated from per-question files and traces; baseline provenance states whether it is official rerun, adapter rerun, or shared-judge rescoring.
- Failure interpretation: If a baseline cannot be audited, it should move to appendix or be marked as reported-only rather than mixed into the primary unified table.
- Table / figure target: Appendix artifact table; main paper baseline audit table.
- Priority: MUST-RUN.

### Block 5: Scope and Generalization Tests

- Claim tested: C4.
- Why this block exists: Protects the title and introduction from exceeding the evidence.
- Dataset / split / task: Small but controlled LoCoMo and LongMemEval subsets; optional synthetic online update stream.
- Compared systems: NativeMem with matched writer/retriever, mismatched writer/retriever, weaker open-weight model, and no repair.
- Metrics: LJ, first-path hit, repair success, stale-path failures, contradiction/update accuracy.
- Setup details:
  - Writer/retriever cross matrix: GPT-5.5, GPT-4o-mini, Qwen-2.5-72B or available local substitutes.
  - Online update subset: interleaved writes and delayed queries after folder growth.
  - Maintenance test: periodic verification on old evidence, not only newly written evidence.
- Success criterion: Matched writer/retriever performs best; mismatches degrade in an interpretable way; repair reduces unreachable new evidence.
- Failure interpretation: Weak cross-model portability should be written as a limitation of model-specific address spaces, not hidden.
- Table / figure target: Appendix generalization table; limitation paragraph in main paper.
- Priority: NICE-TO-HAVE for initial submission, MUST-RUN if the title keeps a broad memory claim.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Freeze trace schema and cost accounting. | Add per-question fields for retrieval-visible tokens, final evidence tokens, total visible tokens, retrieval calls, final answer calls, judge calls, files opened, and trace checksum. | One LoCoMo sample produces complete traces for NativeMem and LLM-Topical Files. | Low; engineering only. | Existing logs do not separate prompt overhead from retrieved observations. |
| M1 | Reproduce the strongest controlled file baselines on LoCoMo. | LLM-Reranked Verbatim, LLM-Topical Files, NativeMem-NoRSP, NativeMem on all LoCoMo samples, at least three builds first. | NativeMem is ahead of LLM-Topical Files and NativeMem-NoRSP with confidence intervals. | Medium API cost. | If the margin is small, the paper must rely more on efficiency and mechanism diagnostics. |
| M2 | Run protocol robustness on LoCoMo. | Final-6K, total-visible-6K/10K/20K, call-limited 3/5/8. | NativeMem's advantage persists under at least one strict total-budget setting. | Medium. | This is the highest-risk stage; negative results change the claim boundary. |
| M3 | Add baseline-native operating points. | TiMem/Mnemis/Mem0 or available strong baseline outputs under their own retrieval settings, then shared judge. | Main conclusion does not reverse when baselines use their recommended settings. | Medium to high, depends on baseline setup. | Official implementations may not expose identical outputs or costs. |
| M4 | Extend the stable protocol to LongMemEval. | NativeMem and controlled baselines on all 500 questions, with the same trace schema. | NativeMem is ahead or competitive; category gains match the memory-update story. | Medium. | LongMemEval may be easier for graph/hierarchy systems; a tie is still useful if cost is lower. |
| M5 | Run scale stress only after LoCoMo/LongMemEval are stable. | BEAM-1M and BEAM-10M for NativeMem, LLM-Reranked Verbatim, LLM-Topical Files, and one strong baseline. | NativeMem degrades more slowly or uses fewer visible tokens at comparable score. | High. | BEAM setup may dominate schedule; do not block core paper on it. |
| M6 | Scope tests and human validation. | 100-question human subset, writer/retriever mismatch, old-evidence repair subset. | Human ranking agrees with LJ; limitations are concrete. | Medium. | Human eval is slow and should be stratified. |

## Compute and Data Budget

- Total estimated API cost depends on GPT-5.5 access and baseline reruns; the current qwen3.6-flash runs were low cost but not sufficient for the ideal paper.
- Data preparation needs: normalize LoCoMo, LongMemEval, and BEAM examples into a shared per-question schema with gold answer, category, source conversation id, and trace path.
- Human evaluation needs: 100 stratified LoCoMo questions first; expand only if LJ-human agreement is weak.
- Biggest bottleneck: not raw model calls, but consistent trace/cost accounting across methods with different retrieval mechanisms.

## Risks and Mitigations

- Risk: NativeMem wins because it sees more intermediate context.
- Mitigation: Main paper includes total-visible-token and call-limited robustness, not only final 6K evidence matching.

- Risk: The ideal prewritten results exceed what current code has validated.
- Mitigation: Keep ideal-paper tables separate from current actual-result logs until the corresponding artifacts exist.

- Risk: Recent baseline numbers are incomparable because of different judges and model versions.
- Mitigation: Use recent papers for positioning only when necessary; primary evidence comes from self-run or shared-judge audited rows.

- Risk: RSP gain is modest relative to file-structure and copy gains.
- Mitigation: Claim the complete LLM-native file memory lifecycle as primary; present RSP as the placement mechanism that adds consistent incremental gain over a strong topical file baseline.

- Risk: The title suggests broad online memory, but experiments are offline build-then-query QA.
- Mitigation: Either add a small online maintenance block or explicitly limit the claim to persistent QA over conversation-derived memory.

## Current Paper Optimization

- Rename planning docs and webpage language from `Wiki Memory` / `Model-Aligned Wiki` toward `NativeMem` where the text is meant to support the paper.
- In the main paper, state that the primary protocol matches final evidence budget and reports total retrieval cost; avoid calling it fully compute-matched.
- Add a protocol-robustness table in the main experiments section if LoCoMo results are available; otherwise move the broad cross-system claim into a more cautious paragraph.
- Keep the most defensible main comparison centered on controlled file-system baselines.
- Move broad online maintenance, cross-generation portability, and low-latency claims to limitations unless Block 5 results are run.

## Final Checklist

- [ ] Main paper tables are backed by per-question artifacts.
- [ ] Novelty is isolated by controlled file-system baselines.
- [ ] Protocol robustness includes total-visible-token and call-limited settings.
- [ ] Baseline provenance is explicit for every row.
- [ ] Multiple memory builds are reported for LLM-operated methods.
- [ ] Human validation checks LJ reliability.
- [ ] Scope is limited to evidence actually supported by the experiments.
