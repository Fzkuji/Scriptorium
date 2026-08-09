# Temporal retrieval microbenchmark — 2026-08-02

## Scope

- Machine: macOS 15.7.3 arm64, Python 3.12.13, NumPy 2.5.1.
- Workload: 10,000 synthetic `MemoryEvent` records; one exact date per event.
- Time filter: calendar year 2022, matching 1,000 records.
- Embedding dimension: 384; document vectors were precomputed.
- Measurements use `time.perf_counter`; reported values are steady-state medians and p95.

This isolates the retrieval kernel. It excludes Topic parsing, file hashing, first-time embedding-model loading/document encoding, and LLM/API latency.

## Results

| Path | No time filter, median | 2022 filter, median | Filter p95 |
|---|---:|---:|---:|
| Temporal candidate scan only | — | 8.575 ms | 8.722 ms |
| BM25 query | 166.767 ms | 25.213 ms | 28.499 ms |
| Embedding query, precomputed vectors | 11.495 ms | 10.577 ms | 11.935 ms |

The filter reduced BM25 median latency by 84.9% because only 10% of records entered BM25 corpus construction and scoring. For precomputed embeddings, it reduced median latency by 8.0%; the full candidate scan remains linear in the number of memory events.

## Instrumentation changelog

- Added no runtime instrumentation and no profiling dependency.
- Collected timings with an external in-memory benchmark using the existing retrieval classes.
- During profiling, changed `MemoryEmbeddingIndex.search` to parse `date_from` and `date_to` once per query rather than once per memory event. Focused retrieval tests passed after the change.
- No instrumentation was left in production code.

## Limits

These measurements do not establish LoCoMo or LongMemEval answer accuracy, end-to-end latency, or API cost. Those require benchmark runs with identical model and retrieval settings.
