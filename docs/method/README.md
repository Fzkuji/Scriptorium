# NativeMem 方法文档

> 更新：2026-07-31。本文件说明方法目录的内容边界和证据状态；当前方法正文以 `nativemem-method.html` 为准。

当前实现模块：

- `code/src/nativemem_versions/v11/memory.py`：Writer、Manager、工作区事务和 benchmark 入口；
- `topic_markdown.py`：自由 Markdown memory unit 的 footnote 解析与物化；
- `reconciliation.py`：Topic diff 分类和 Reconciler 输出校验；
- `derived_views.py`：Timeline、Recent 和 structure map 的确定性重建；
- `runtime_state.py`：provider source record、cursor、Git 提交和触发条件；
- `online_runtime.py`：增量写入、局部整理和每日全局管理的在线编排。

## 文档职责

| 文档 | 状态 | 内容 |
|---|---|---|
| 本文件 | 目录说明 | 方法文档关系和证据边界 |
| [`nativemem-method.html`](nativemem-method.html) | 当前方法正文 | 两个主要贡献、记忆结构、Topic 规范、构建、管理、检索和实现边界 |
| [`designs/file_native_multiview_design.md`](designs/file_native_multiview_design.md) | 当前设计 | 记忆状态、写入触发、查询、维护和恢复机制 |
| [`../internal/analysis/recent_memory_framework_comparison.md`](../internal/analysis/recent_memory_framework_comparison.md) | 内部设计分析 | 近期系统已有而 NativeMem 缺少或尚未定型的组件 |
| [`designs/memory_management_design.md`](designs/memory_management_design.md) | 历史草案 | 2026-07-04 的运行逻辑，不再作为当前规范 |
| [`versions/nativemem_v3_baseline.md`](versions/nativemem_v3_baseline.md) | 历史版本 | v3 baseline 固化记录 |
| [`versions/`](versions/) | 历史版本 | v1–v10 的设计和结果记录 |
| [`../experiments/experiment.html`](../experiments/experiment.html) | 实验页面 | 实验矩阵、结果与执行状态 |

出现冲突时，以 `nativemem-method.html` 和 `designs/file_native_multiview_design.md` 为当前设计依据；历史文档只用于追踪方案演变和既有实验。

## 当前方法概览

NativeMem 是一个 file-native、multi-view 的外部长期记忆系统。原始交互和派生记忆均保存在可读、可编辑、可版本化的文本文件中；LLM 负责内容抽取、组织和检索决策，Runtime 负责路径、日期、引用、链接、预算和事务边界等确定性约束。

当前设计包含六类状态：

1. `Source Memory`：追加保存完整原始交互，是事实来源；
2. `Topical View`：按主题组织派生记忆；
3. `Temporal View`：按事件时间组织派生记忆；
4. `Hyperlink Relations`：使用 Markdown links 和 backlinks 表示跨文件关系；
5. `Recent Memory`：始终保留最近写入的 50 条记录，超过容量时按 FIFO 移除最早记录；
6. `Core Memory`：由 LLM 管理的独立文本文件，保存每次交互都需要提供的信息。

文本文件是权威状态。BM25 和 Embedding 索引都可以从文件重建，只提供检索能力，不保存唯一事实。

事实变化采用完整时间记录：旧事件和新事件都保存在 Topical View 与 Temporal View 中，并绑定各自的 source references；查询时由 LLM 根据时间范围和事件顺序判断。系统不通过覆盖旧事实维护单一 current state，也不增加 `supersedes` 或图边有效期。

## 三级记忆管理

记忆管理只包含三级：

1. **增量写入**：cursor 后的内容达到规定 token 或上下文规模时写入记忆；session 空闲一小时后处理剩余内容。写入时由 Agent 自行决定 topic path、文件和章节。
2. **周期性局部整理**：benchmark 中每累计固定数量的 session，整理本周期发生新增写入的 Topic；实际部署中按累计写入批次或新增 token 阈值触发。基于查询成功率和检索成本的动态触发暂不作为标准方法。
3. **每日全局管理**：每天最多一次，对跨 Topic 的目录、文件、章节和链接进行整体整理；没有新增 incremental-writing commit 时跳过。

Git commit 是三类修改共同使用的有效状态边界。增量写入必须在 commit 成功后推进 session cursor；不增加 batch ID、`memory_dirty` 或独立写入日志。

Git/cursor、token/空闲触发和每日管理条件已经由 `online_runtime.py` 实现；静态 benchmark harness 仍使用固定 session 写入、每 5 个 session 的局部整理和有新增记忆时的最终全局整理。Writer 外层最多 12 轮，Manager 最多 8 轮。长时间 scheduler 进程和具体 Agent provider plugin 不属于静态 benchmark 入口。

## Query-Time Access

查询时，LLM 获得问题、Recent Memory、Core Memory 和 compact structure map。标准配置提供三种检索方法，由 LLM 自主决定调用顺序、检索词和停止条件：

| 检索方法 | 作用 |
|---|---|
| `grep` | 查找名称、数字、编号和精确短语 |
| `bm25_search` | 在路径不明确或需要跨文件召回时提供词法排序候选 |
| `embedding_search` | 提供语义相似候选 |

目录、文件、timeline、links 和 source references 作为结构化访问方式继续保留。三种检索都只返回候选，不替代 LLM 的相关性判断和后续检索决策。

标准检索预算暂定为最多 8 次 LLM retrieval rounds、5 次工具调用和 10K memory-visible tokens；development set 扫描 `3/5/8` calls 与 `6K/10K/20K` tokens，test set 固定参数。主要策略对标是 ByteRover 的 5-Tier Progressive Retrieval。其他对标包括 Infini Memory-H/A、LightMem，以及作为后续增强参考的 Semble。详细映射见 [`designs/file_native_multiview_design.md`](designs/file_native_multiview_design.md#34-reference-systems)。

## 当前证据边界

| 状态 | 内容 | 证据边界 |
|---|---|---|
| 已完成结果 | v8.8 风格的 Topical + Temporal 视图、source resolution 和连续文件导航 | LoCoMo 1,540 题与 LongMemEval-S 500 题 |
| 当前实现线 | V11 agentic writer/manager、Recent 50 FIFO、Core 2K、周期局部整理、事务回滚、BM25、Embedding 与检索预算 | 已有实现、单元测试和小规模真实 API smoke test；正式实验矩阵尚未完成 |
| 已实现、待部署验证 | Footnote Topic、自动 Reconciler、provider source IDs、Git/cursor、token/空闲触发和每日管理条件 | 通过单元与集成测试；尚未完成多 provider 长时间在线运行和成本评估 |

现有 90.8% LoCoMo 与 87.0% LongMemEval-S 结果不能自动归因于 V11、Recent Memory、Core Memory、Git/cursor 事务、BM25 或 Embedding。新组件需要在相同 memory、answerer、judge 和预算下分别消融。

## 待确定项

当前只保留会影响实现或实验定义的未决问题：

1. Topical 与 Temporal 采用同步文本 occurrence，还是规范记录加视图链接；
2. Writer/Manager 轮数与 Core Memory 2K token 上限是否需要根据 development set 调整；
3. 实际部署中是否需要加入基于查询成功率和检索成本的动态整理触发；
4. 是否采用 Semble 风格的 BM25 + Embedding 融合与结构重排。

## 历史演进

- v1–v3：从自定义工具转向标准文件工具；
- v4–v6：探索原文保留、事件抽取和代码路径约束；
- v7：模型自组织与周期维护；
- v8/v8.8：形成 topic + timeline + source refs 的主结构；
- v9/v10：研究 writer 粒度、前序信息和整理频率；
- v11：引入 agentic writer/manager、事务同步和检索验证。
