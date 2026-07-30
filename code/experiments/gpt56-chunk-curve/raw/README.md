# Raw artifact links

本目录使用相对链接集中提供原始实验入口，不复制约 650 MB 的 build、memory、QA、score 和 audit 文件。链接目标都位于仓库的 `results/formal/` 下，原始 artifact 保持不变。

72 个目标构建的唯一文件位置不应通过目录名推断，应使用 `../artifacts/completed_runs.jsonl`。该清单已经排除 W=24、session、smoke、失败和未完成记录。

`../artifacts/raw_roots.json` 将 20 个入口标记为 `construction_target_source`、`qa_or_scoring` 或 `excluded_archive`。`excluded_archive` 仍保留用于错误追踪，但不参与统计。
