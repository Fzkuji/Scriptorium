# 实验配置目录

`experiments/` 保存实验设计、冻结 manifest 和待执行矩阵，不保存模型输出。

每个 study 使用独立子目录：

```text
experiments/<study-id>/
  experiment_manifest.json
  run_matrix.jsonl
  README.md
```

`experiment_manifest.json` 定义研究问题、自变量、固定配置、数据与 evaluator hash、模型和指标。`run_matrix.jsonl` 一行对应一个独立输出目录。模型输出写入 `results/formal/<study-id>/`。

`gpt56-chunk-curve/` 由 `scripts/prepare_gpt56_chunk_curve.py` 从本地冻结数据生成。该生成器没有模型执行模式。
