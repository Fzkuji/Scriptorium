# Topic 编辑接口决策记录

> 状态：已被当前 Topic block contract 取代，2026-08-01。

本文件最初讨论“LLM 任意改写无标注正文，Runtime 再调用第二个 LLM Reconciler 恢复记忆边界、日期和来源”的方案。该方案没有作为当前主方法采用，原因是它增加一次语义调用，并且让 Writer 的编辑结果在提交前仍需要另一个模型重新解释。

当前实现采用单一 Markdown 编辑契约：

- Writer、局部 Manager、全局 Manager 和 repair Agent 只使用共享 shell 编辑 `topics/` 与 `core.md`；
- 一个 Topic memory unit 是一个自然语言段落，末尾有一个 Obsidian-compatible `^block-id`；
- 事实后的 evidence footnote 同时记录 `YYYY`、`YYYY-MM`、`YYYY-MM-DD` 或 `undated` 以及 Source handles；前三种按原始精度进入 Timeline，`undated` 不进入；
- 新 block/evidence 使用 transaction-local `new-*` labels，Runtime 物化稳定 ID；
- LLM 自己决定补充、改写、拆分、合并和删除后的完整 Markdown；Runtime 不再次判断这些语义；
- Runtime 只负责格式与来源校验、相对链接改写、Timeline/Recent/Relations 重建和原子安装；
- 旧版 Reconciler 与 `save_memory(events[])` 仅保留为旧 workspace 的兼容代码，不暴露给当前 Agent。

当前规范与完整示例见 [`../scriptorium-method.html`](../scriptorium-method.html)，实现计划与验收范围见 [`../../internal/superpowers/plans/2026-08-01-topic-block-memory-runtime.md`](../../internal/superpowers/plans/2026-08-01-topic-block-memory-runtime.md)。
