# NativeMem 文档信息架构设计

## 目标

将 `Model-Aligned-Wiki.html` 保留原名并移动到 `docs/`，作为项目文档的唯一总入口。其余文档按职责归档，消除含义不明确的 `index.html`，并修复仓库内全部路径引用。

## 内容职责

- `docs/Model-Aligned-Wiki.html`：研究问题、核心设计、架构图、贡献、结果摘要和主要文档入口。
- `docs/related-work/related-work.html`：论文正文式 Related Work，只讨论已有工作。
- `docs/related-work/survey.html`：完整 taxonomy、系统比较、评测与资源。
- `docs/method/nativemem-method.html`：当前方法的唯一权威说明。
- `docs/experiments/experiment.html`：实验设置、结果和分析的主页面；研究计划由 `docs/research/research-plan.html` 单独保存。

## 目录职责

- `assets/`：共享页面样式。
- `related-work/evidence/`：benchmark、实验设置和可比性调查。
- `method/designs/`：当前具体设计；`method/versions/`：历史版本；`method/reports/`：方法演进报告。
- `experiments/protocols/`：正式评测协议；`experiments/results/`：结果与结果选择；`experiments/studies/`：编号研究；`experiments/runs/`：执行记录；`experiments/baselines/`：适配器；`experiments/infrastructure/`：运行网关。
- `prompts/`：提示词材料；`archive/`：失效历史材料；`internal/`：研究分析和开发过程文档。

## 命名规则

原文件名默认不变，只改变目录。三个入口例外：

- `Model-Aligned-Wiki.html` 保留名称，不改为 `index.html`。
- `method/index.html` 改为 `method/nativemem-method.html`。
- `method/report/index.html` 改为 `method/reports/method-evolution-results.html`。

迁移后不得存在 `index.html`。不保留旧路径副本或兼容链接。

## 内容整理规则

- 总入口保留项目概览，不重复完整 Method、Related Work 表格或实验明细。
- Related Work 与 Survey 分工明确，其他调研材料作为 evidence，不进入主要导航。
- Method 页面保留当前方法；历史版本和演进材料只由支撑文档承载。
- Experiments 页面保留综合叙述；详细协议、运行记录和结果矩阵通过链接引用，不在多个主页面重复维护。
- 所有主 HTML 页面使用相同样式、顶部项目路径和回到总入口的链接。

## 验证

- 搜索并消除旧路径引用和 `index.html`。
- 验证 HTML 目录锚点、CSS、图片和相对链接。
- 验证 Markdown 本地链接。
- 运行文档页面测试和 `git diff --check`。
