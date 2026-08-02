# 近期 Agent Memory 框架对比

> 范围：2026 年 4 月至 7 月的近期预印本，以及 Mem0、Letta、LangGraph、Zep/Graphiti 等实际框架。重点记录它们已经具备而 NativeMem 当前没有或尚未设计完整的组件。

## 1. 完全没有的组件

### 1.1 独立的后台整理 Agent

- Letta 的 sleep-time agent 在主 Agent 之外处理共享 memory blocks。
- LangGraph/Deep Agents 支持在对话之间运行 consolidation agent，读取近期历史并合并长期记忆。
- NativeMem 当前采用定时启动同一类 Organizer 的设计，没有独立常驻后台 Agent；这不是当前必需组件。

### 1.2 多用户和权限隔离

- LightMem 使用 user ID 隔离 MTM，并将 LTM 设计为去标识化知识。
- LangGraph 使用 namespace 区分用户、组织和应用范围。
- Letta 支持 read-only blocks、共享 blocks 及动态 attach/detach。
- NativeMem 当前没有 namespace、用户权限或敏感路径访问控制。

### 1.3 查询缓存和独立 OOD 检测

- ByteRover 使用查询缓存和 OOD detection。
- NativeMem 每次重新检索，也没有独立的“记忆中不存在答案”检测器。

## 2. 已有但尚未设计完整的组件

### 2.1 Recent Memory 与 Core Memory

- LightMem 明确定义 STM、MTM、LTM，设置固定检索预算、MTM 容量和低效用淘汰。
- Letta 使用始终在上下文中的 memory blocks 表示重要工作状态，并允许 attach/detach。
- LangGraph 使用 thread-scoped checkpoints 保存短期状态。
- NativeMem 不采用额外的工作记忆操作集合。Recent Memory 固定保留最近写入的 50 条记录，超过容量时按 FIFO 移除最早记录；Core Memory 是由 LLM 管理的独立文本文件，用于保存每次交互都需要提供的信息。

### 2.2 记忆整理机制

- Infini Memory 使用 CURRENT、token/时间 flush、局部 rewrite 和 topic split/merge。
- LightMem 在线写 MTM，后台增量整理到 graph-structured LTM。
- Zep/Graphiti 增量更新 episodes，并建议周期性重建 communities。
- NativeMem 不需要额外 CURRENT buffer；Source Memory 加 session cursor 已覆盖待整理输入。当前采用三级管理：按 token、上下文规模或 session 空闲一小时增量写入，按固定 session 周期或累计新增量进行局部整理，以及每日全局管理。

### 2.3 生命周期和遗忘

- ByteRover 使用 importance score、maturity tier 和 recency decay。
- LightMem 使用容量上限、低效用淘汰和 confidence decay。
- GEM 定义 ingestion、revision、forgetting、retrieval 四类状态操作及正确性条件。
- DeMem 从决策损失定义安全遗忘边界。
- NativeMem 的 Recent Memory FIFO 移除不删除 Source Memory 或长期派生记忆。当前没有长期失效、归档和遗忘规则。

### 2.4 多视图与事实变化

- Zep/Graphiti 使用 temporal edges、有效区间和 edge invalidation 管理新旧关系。
- NativeMem 不维护单一 current state，也不增加 `supersedes` 或图边有效期。旧事件和新事件都以带时间、带 source references 的文本记录保存在 Topical 与 Temporal View 中，由 LLM 在查询时根据时间范围和事件顺序判断。

### 2.5 规模与运行成本

- ByteRover 使用多级检索和缓存。
- Infini Memory 使用 summary catalog、行范围读取和 BM25 fallback。
- Filesystem-Based Memory 系统评估 store health、增长轨迹、工具集影响和检索成本。
- Agent Memory: Characterization 分阶段统计 construction、retrieval 和 generation 成本。
- NativeMem 当前设计包含目录、grep、BM25、Embedding 和多步访问。标准检索预算暂定为 8 个 retrieval rounds、5 次工具调用和 10K memory-visible tokens，并在 development set 上比较 `3/5/8` calls 与 `6K/10K/20K` tokens。规模增长、维护成本和尾延迟属于实验评测，不是新的方法组件。

## 3. 成熟框架的增量与周期处理

### Mem0

每次 `add` 时立即抽取事实并执行新增、更新、删除或 no-op。它主要做逐条增量更新，不默认周期性重组整个 memory store。

### Letta

主 Agent 可在 hot path 更新 memory blocks。启用 sleep-time agent 后，独立 Agent 按配置频率处理共享记忆。Archival memory 使用外部向量存储；重要状态通过始终在上下文中的 blocks 保存。

### LangGraph

短期状态通过 thread-scoped checkpointer 在图执行步骤之间持久化；长期记忆写入 Store。应用可以在 hot path 更新，也可以调度后台 consolidation agent。框架负责 checkpoint、pending writes 和故障恢复，但不规定具体整理算法。

### Zep/Graphiti

新 episode 到达时增量抽取和更新实体、关系与有效时间。批量导入适合初始化空图，但不执行 edge invalidation。高层 communities 可以单独构建，并建议周期性重建。

## 4. 当前取舍

NativeMem 当前只吸收必要的执行逻辑：

1. 按 token、上下文规模或 session 空闲一小时触发增量写入；
2. benchmark 中按固定 session 周期、实际部署中按累计写入批次或新增 token 触发局部 Topic 整理；
3. 每天最多执行一次全局记忆管理。

Source Memory、session cursor 和 Git commit 是这三级管理共同使用的运行基础。系统不增加独立 buffer、batch ID、`memory_dirty` 或为了后台整理而引入常驻多线程服务。

Procedural memory 或 skill memory 不属于当前范围。NativeMem 当前研究对话中的外部事实记忆，不把代码、执行轨迹或任务技能统一定义为 memory。

## 5. 参考资料

- [ByteRover](https://arxiv.org/html/2604.01599)
- [LightMem](https://arxiv.org/html/2604.07798)
- [Infini Memory](https://arxiv.org/html/2606.10677)
- [Agent Memory: Characterization and System Implications](https://arxiv.org/html/2606.06448)
- [Are We Ready for an Agent-Native Memory System?](https://arxiv.org/html/2606.24775)
- [Filesystem-Based Memory for LLM Agents](https://arxiv.org/html/2607.26637)
- [Is Agent Memory a Database?](https://arxiv.org/html/2605.26252)
- [DeMem](https://arxiv.org/html/2605.10870)
- [LangGraph Memory](https://docs.langchain.com/oss/python/concepts/memory)
- [LangGraph Background Consolidation](https://docs.langchain.com/oss/python/deepagents/memory)
- [Letta Memory Blocks](https://docs.letta.com/guides/core-concepts/memory/memory-blocks)
- [Graphiti Episodes](https://help.getzep.com/graphiti/core-concepts/adding-episodes)
- [Graphiti Communities](https://help.getzep.com/graphiti/core-concepts/communities/)
