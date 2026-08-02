# NativeMem 方法文档

> 更新：2026-08-02。本文件说明方法目录的内容边界和证据状态；当前方法正文以 `nativemem-method.html` 为准。

当前实现模块：

- `code/src/build.py`：Source session 转换、Writer 批量和构建调度；
- `code/src/management/`：Writer、Manager 和工作区事务；
- `code/src/markdown/`：段落 block、claim-adjacent evidence 与关系链接解析；
- `code/src/runtime/`：派生视图、在线状态、tokenizer 和 Writer capacity artifact；
- `code/src/retrieval/`：文件、grep、BM25、Embedding 和 Agent-controlled retrieval；
- `code/scripts/model_capacity/calibrate_writer.py`：按模型校准完整 session 写入容量；
- `code/scripts/nativemem/`：LoCoMo、LongMemEval 和消融入口。

## 文档职责

| 文档 | 状态 | 内容 |
|---|---|---|
| 本文件 | 目录说明 | 方法文档关系和证据边界 |
| [`nativemem-method.html`](nativemem-method.html) | 当前方法正文 | 两个主要贡献、记忆结构、Topic 规范、构建、管理、检索和实现边界 |
| [`designs/file_native_multiview_design.md`](designs/file_native_multiview_design.md) | 当前设计 | 记忆状态、写入触发、查询、维护和恢复机制 |
| [`../internal/analysis/recent_memory_framework_comparison.md`](../internal/analysis/recent_memory_framework_comparison.md) | 内部设计分析 | 近期系统已有而 NativeMem 缺少或尚未定型的组件 |
| [`../experiments/experiment.html`](../experiments/experiment.html) | 实验页面 | 实验矩阵、结果与执行状态 |

出现冲突时，以 `nativemem-method.html` 和 `designs/file_native_multiview_design.md` 为当前设计依据。

## 当前方法概览

NativeMem 是一个 file-native、multi-view 的外部记忆系统。Source、Topic 和 Core 是需要保存的权威文本；Timeline、Recent、Relations 与检索索引从这些状态重建。LLM 负责 Topic/Core 的语义内容、组织、证据日期和检索决策，Runtime 负责 ID、来源解析、路径、派生视图、预算和事务边界。

当前设计包含六类状态：

1. `Source Memory`：追加保存完整原始交互，是事实来源；
2. `Topical View`：LLM 直接编辑的主要语义状态；
3. `Temporal View`：Runtime 从 Topic 中 dated evidence 生成的时间视图；
4. `Hyperlink Relations`：Runtime 从正文 Markdown links 生成的 outbound/backlinks；
5. `Recent Memory`：始终保留最近写入的 50 条记录，超过容量时按 FIFO 移除最早记录；
6. `Core Memory`：由 LLM 管理的独立文本文件，保存每次交互都需要提供的信息。

一个 Topic memory unit 是一个自然语言段落：事实后跟 evidence footnote，段落末尾只有一个 Obsidian-compatible block ID。Runtime 将新 block 物化为 8 位十六进制 ID，并在候选冲突时重新计算直到唯一。temporal evidence 的 `Time` 字段使用 `YYYY`、`YYYY-MM` 或 `YYYY-MM-DD`；无法确定任何年份时使用 `undated`。前三种按原始精度生成 Timeline，`undated` 不生成。时间只作为 footnote 元数据保存，正文仅在事实本身自然包含时间时保留日期，不为复制 `Time` 字段而追加日期。Topic 间的 Markdown link 必须指向 `#^block-id`，文件级或 heading 级 Topic link 会被拒绝。Writer、局部 Manager、全局 Manager 和 repair Agent 都只使用 shell 形成完整 Markdown；Runtime 物化临时 ID、解析 Source、规范化 marker 间距、校验 Topic links、改写移动后的相对链接，并重建派生视图。BM25 和 Embedding 只提供检索能力，不保存唯一事实。

事实变化采用完整时间记录：旧状态和新状态都保存在 Topic 中，并绑定各自的 dated evidence；Runtime 将它们分别生成到 Timeline。`undated` evidence 不生成 Timeline 条目。系统不增加 `supersedes` 或图边有效期。

## 三级记忆管理

记忆管理只包含三级：

1. **增量写入**：cursor 后的内容达到规定 token 或上下文规模时写入记忆；session 空闲一小时后处理剩余内容。写入时由 Agent 自行决定 topic path、文件和章节。
2. **周期性局部整理**：benchmark 中每累计固定数量的 session，整理本周期发生新增写入的 Topic；实际部署中按累计写入批次或新增 token 阈值触发。基于查询成功率和检索成本的动态触发暂不作为标准方法。
3. **每日全局管理**：每天最多一次，对跨 Topic 的目录、文件、章节和链接进行整体整理；没有新增 incremental-writing commit 时跳过。

Git commit 是三类修改共同使用的有效状态边界。增量写入必须在 commit 成功后推进 session cursor；不增加 batch ID、`memory_dirty` 或独立写入日志。

Git/cursor、token/空闲触发和每日管理条件已经由 `code/src/runtime/online.py` 实现；静态 benchmark harness 使用冻结的 Writer capacity artifact 或显式 `session_batch`，并按固定 session 周期执行局部整理。Writer 外层最多 12 轮，Manager 最多 8 轮。长时间 scheduler 进程和具体 Agent provider plugin 不属于静态 benchmark 入口。

Writer 批量可以由 `BuildConfig.calibration_path` 指向的校准产物控制。校准使用两组通用合成 session 扫描递增 token 上限，只有在事务完成且全部 source references 被写入权威记忆时才通过；首个失败容量之前的最大通过值成为安全上限。Runtime 验证 model、Writer protocol hash 和 tokenizer identity，随后只组合完整 session。未提供校准产物时使用显式 `session_batch`。

## Query-Time Access

查询时，LLM 获得问题、Recent Memory、Core Memory 和 compact structure map。标准配置提供三种检索方法，由 LLM 自主决定调用顺序、检索词和停止条件：

| 检索方法 | 作用 |
|---|---|
| `grep` | 查找名称、数字、编号和精确短语 |
| `bm25_search` | 在路径不明确或需要跨文件召回时提供词法排序候选 |
| `embedding_search` | 提供语义相似候选 |

目录、文件、timeline、links 和 source references 作为结构化访问方式继续保留。Topic 和 Source 都是三种检索方法的可访问文本；Source 不需要通过专用工具间接读取。三种检索都只返回候选，不替代 LLM 的相关性判断和后续检索决策。

Source 始终可以通过目录浏览、文件读取、`grep`、BM25 和 Embedding 直接访问。函数参数 `retrieval.QueryConfig(verify_sources=True|False)` 只控制 prompt 是否要求查询 Agent 在回答前核验相关原始对话；设为 `False` 时核验变为可选，不改变 Source 可见性或工具集合。实际参数值记录在每题 `tool_trace` 的 termination 条目中。NativeMem 的运行参数均由入口显式构造后逐层传入，不从进程环境读取。

标准检索预算暂定为最多 8 次 LLM retrieval rounds、5 次工具调用和 10K memory-visible tokens；development set 扫描 `3/5/8` calls 与 `6K/10K/20K` tokens，test set 固定参数。主要策略对标是 ByteRover 的 5-Tier Progressive Retrieval。其他对标包括 Infini Memory-H/A、LightMem，以及作为后续增强参考的 Semble。详细映射见 [`designs/file_native_multiview_design.md`](designs/file_native_multiview_design.md#34-reference-systems)。

## 当前证据边界

| 状态 | 内容 | 证据边界 |
|---|---|---|
| 已完成结果 | v8.8 风格的 Topical + Temporal 视图、source resolution 和连续文件导航 | LoCoMo 1,540 题与 LongMemEval-S 500 题 |
| 当前实现线 | shell-only Writer/Manager、Topic paragraph blocks、evidence footnotes、Timeline/Recent/Relations 派生、Recent 50 FIFO、Core 2K、事务回滚、Writer 容量校准、BM25、Embedding 与检索预算 | 已有实现与单元/集成测试；正式实验矩阵尚未完成 |
| 已实现、待部署验证 | provider source IDs、Git/cursor、token/空闲触发和每日管理条件 | 通过单元与集成测试；尚未完成多 provider 长时间在线运行和成本评估 |

现有 90.8% LoCoMo 与 87.0% LongMemEval-S 结果来自此前实现，不能自动归因于当前 Writer/Manager、Recent Memory、Core Memory、Git/cursor 事务、容量校准、BM25 或 Embedding。新组件需要在相同 memory、answerer、judge 和预算下分别消融。

## 待确定项

当前只保留会影响实现或实验定义的未决问题：

1. Topic memory block 的理想语义粒度是否需要进一步约束；
2. Writer/Manager 轮数与 Core Memory 2K token 上限是否需要根据 development set 调整；
3. 实际部署中是否需要加入基于查询成功率和检索成本的动态整理触发；
4. 是否采用 Semble 风格的 BM25 + Embedding 融合与结构重排。
