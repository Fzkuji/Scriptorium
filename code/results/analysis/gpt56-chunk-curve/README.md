# GPT-5.6 Writer-Window Experiment

本目录保存 GPT-5.6 writer-window 历史实验的完成清单和分析结果。目标实验为 3 个模型档位、4 个 writer window（4、8、16、32）和 6 个 history，共 72 个构建。原计划中的 W=6、12、20、24 和 session 不属于本次完成矩阵。

目录内容：

- `../../../scripts/configs/gpt56-chunk-curve/experiment_manifest.json`：最初的完整实验计划。
- `../../../scripts/configs/gpt56-chunk-curve/run_matrix.jsonl`：原始 162 行候选配置。
- `artifacts/code_manifest.json`：执行时源码 SHA 和原冻结路径；其中的历史路径按原值保留。
- `artifacts/completed_runs.jsonl`：72 个目标 run 的唯一 artifact、指标和校验值。
- `artifacts/raw_roots.json`：原始结果目录的文件数与字节数。
- `tables/`：逐 run 数据、聚合表、配对差异、W=32 QA screening、论文表格和结论说明。

旧分析入口和源码副本依赖已经删除的 NativeMem 版本，因此不作为当前可执行代码保留。需要逐字复现时使用清理前提交 `31cc34c`；正式原始结果继续位于 `code/results/formal/`。核心 memory 方法代码在 72 个构建中保持同一 SHA；runner 和 LongMemEval converter 的执行期 SHA 变化记录在 `artifacts/code_manifest.json`。

论文结论与限制见 `tables/RESULTS.md`。机器可读总表见 `tables/summary.json`。
