# Code provenance

`current/` 是相关代码的当前快照，用于从统一实验目录查看和复查实现。它不假定所有历史批次使用完全相同的 runner 文件。

`manifest.json` 对每个文件记录当前 SHA。构建时直接记录过的代码还包含 `recorded_build_hash_counts`，可看到每个 SHA 对应多少个目标构建。每个 run 的最终执行来源仍以 `../artifacts/completed_runs.jsonl` 指向的 `build.json` 为准。

72 个目标构建中，`src/nativemem.py`、`src/v10_memory.py`、`src/v8_memory.py` 和 `src/adapters/run_nativemem.py` 各自只有一个执行 SHA。`scripts/run_gpt56_chunk_curve.py` 有两个 SHA，`scripts/run_v88_gpt55_longmemeval.py` 有三个 SHA；所有 run 的冻结输入与 `unit.json` 已重新验证。
