# GPT-5.6 Writer-Window Experiment

本目录统一保存本次实验的配置、代码快照、原始结果入口、完成清单和分析结果。目标实验为 3 个模型档位、4 个 writer window（4、8、16、32）和 6 个 history，共 72 个构建。原计划中的 W=6、12、20、24 和 session 不属于本次完成矩阵。

目录内容：

- `experiment_manifest.json`：最初的完整实验计划，保留为设计记录。
- `run_matrix.jsonl`：原始 162 行候选配置。
- `analyze.py`：离线完整性检查与统计分析入口，不请求模型。
- `raw/`：20 个原始 construction、QA、score、analysis、smoke 和失败记录目录的相对链接。原始文件没有移动或复制。
- `code/current/`：37 个相关实现与测试文件的当前快照。
- `code/manifest.json`：当前代码 SHA 与 72 个构建记录的执行时 SHA 对照。
- `artifacts/completed_runs.jsonl`：72 个目标 run 的唯一 artifact、指标和校验值。
- `artifacts/raw_roots.json`：原始结果目录的文件数与字节数。
- `analysis/`：逐 run 数据、聚合表、配对差异、W=32 QA screening、论文表格和结论说明。

重新生成分析：

```bash
python3 experiments/gpt56-chunk-curve/analyze.py
```

分析会核对 72 个 `build.json`、`unit.json`、`memory/_SUCCESS.json`、最终 memory tree、冻结数据绑定和 LoCoMo evaluator SHA。核心 memory 方法代码在 72 个构建中保持同一 SHA；runner 和 LongMemEval converter 存在执行期间的 SHA 变化，具体计数保存在 `code/manifest.json`，但所有冻结 `unit.json` 均能由当前数据重建并通过一致性检查。

论文结论与限制见 `analysis/RESULTS.md`。机器可读总表见 `analysis/summary.json`。
