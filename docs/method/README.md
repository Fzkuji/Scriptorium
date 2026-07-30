# NativeMem 方法设计与版本演进

> 更新：2026-07-30。本文件是当前方法总入口；历史版本文档保留原貌，不删除。

## 相关文档

| 文档 | 内容 |
|---|---|
| 本文件 | 当前方法、检索架构、版本边界与实验状态 |
| [`memory_management_design.md`](memory_management_design.md) | 写入、整理、检索三个环节的运行逻辑 |
| [`nativemem_v3_baseline.md`](nativemem_v3_baseline.md) | v3 baseline 的历史固化 |
| [`versions/`](versions/) | v1–v10 的独立设计与结果记录 |
| [`../experiment-plan.html`](../experiment-plan.html) | 实验矩阵与统一评测协议 |

---

## 1. 当前方法

**NativeMem 让 LLM 自主管理可读的 Markdown 长期记忆，并把 LLM 本身作为主检索器。** 文件目录、`grep`、BM25、时间线和原文解析都是确定性工具；它们只返回候选或证据，LLM 在连续上下文中决定调用哪个工具、是否改写查询、是否继续搜索以及何时回答。

方法坚持：

- 不使用 Embedding 模型、向量索引或向量数据库；
- 不把 BM25 设为固定的“先检索再回答”流水线；
- 写入、维护和查询共享同一套文件结构；
- 每条抽象事件绑定稳定 source reference，可按需回到原始轮次核验；
- 语义决策由 LLM 完成，日期、引用、索引同步和结构不变量由代码保证。

核心思想仍是 **RAM（Retrieval-Aligned Memorization）**：写入时根据“未来会到哪里寻找”选择 topic path，查询时由 LLM 在同一结构上导航。新增的 BM25 不改变这一立论，它只是 LLM 在路径不明确或需要跨文件召回时可调用的词法工具。

## 2. 记忆状态

NativeMem 的逻辑状态包含：

1. **Source Memory**：按稳定轮次 ID 保存完整原始对话；
2. **Topical View**：LLM 选择 topic path 和 heading 的抽象事件；
3. **Temporal View**：代码按日期生成 `timeline/YYYY/MM/DD.md`；
4. **Cross-links**：topic、timeline 和 source 共享 event ID 与 source refs。

抽象视图负责缩小证据范围，`read_original` 负责恢复原词和邻近上下文。通用 shell 不应直接读取 `sources/`，避免查询退化为对原始对话的全文搜索。

## 3. LLM-controlled retrieval

NativeMem 不预设固定检索顺序。LLM 根据问题与每轮 observation 自主组合以下工具：

| 工具 | 适用情况 | 返回内容 |
|---|---|---|
| 目录导航与 `cat` | 主题或人物路径明确 | 文件结构与完整 topic 上下文 |
| `grep` | 专名、原句、编号或精确短语 | 命中行与附近文本 |
| `bm25_search` | 路径不明确、多概念问题、grep 未命中 | 按词法相关性排序的事件候选 |
| timeline | 日期、先后、持续时间、状态变化 | 按日期组织的事件 |
| `read_original` | 已有 refs，需要核验原词、指代或冲突 | 受限的原始轮次与邻近上下文 |

### BM25 的定位

BM25 是增强版关键词排序，不是语义模型。当前扩展以**记忆事件**为索引单元，只索引 topic 中的规范事件，避免 timeline 重复；正文、topic path、heading 和日期共同参与词法匹配，再用确定性规则对实体、路径、日期和 source grounding 做可解释重排。

该工具默认关闭，必须通过实验确认增益后才进入正式主配置。Semble 只提供“BM25 + 结构规则重排 + 增量索引”的工程启发；NativeMem 不采用其 Embedding 部分。

## 4. 写入与维护

### 写入

Writer 查看现有结构后提出：

- 自包含事件内容；
- 日期；
- source refs；
- topic path 与 headings。

代码校验日期、路径和引用，生成 event ID，并同步 topic、timeline 和 source links。没有有效 source reference 的候选不能进入正式记忆。

### 维护

LLM 判断近义事件、topic 合并和章节组织；代码负责：

- 精确去重；
- source refs 并集守恒；
- topic/timeline 事件集合一致；
- backlink 重建；
- staging、原子替换和失败回滚。

当前 V11 还包含写后检索验证与修复，但正式完整实验尚未完成，不能用 v8.8 的结果证明该机制有效。

## 5. 版本与证据边界

| 状态 | 内容 | 证据边界 |
|---|---|---|
| 已完成主结果 | v8.8 风格的双视图、source resolution、连续文件导航 | LoCoMo 1,540 题与 LongMemEval-S 500 题 |
| 当前实现 | V11 agentic writer/manager、事务同步、写后验证 | 实现与单元测试已有，正式矩阵待完成 |
| 检索扩展 | 可选事件级 BM25 与结构重排 | 代码原型与测试已有，受控 QA 尚未完成 |

现有 90.8% LoCoMo 与 87.0% LongMemEval-S 结果不能自动归因于 V11 或 BM25。新机制必须在同一 memory、同一 answerer、同一 judge 和同一预算下独立消融。

## 6. 下一步实验

固定同一批已构建 memory，比较：

1. 当前 LLM 文件导航；
2. 导航 + 原始 BM25；
3. 导航 + BM25 + 结构重排；
4. 固定 BM25 top-k + answerer 诊断基线；
5. 分别移除路径、实体、时间和 source-grounding 重排。

除最终准确率外，还要报告 gold evidence recall、首次有效证据轮数、各工具调用次数、可见 token、延迟和索引大小。只有受控实验确认增益后，才更新正式论文主方法和主结果。

## 7. 历史演进

历史版本不删除，完整记录保留在 [`versions/`](versions/)：

- v1–v3：从自定义工具转向标准文件工具；
- v4–v6：探索原文保留、事件抽取和代码路径约束；
- v7：模型自组织与周期维护；
- v8/v8.8：形成 topic + timeline + source refs 的主结构；
- v9/v10：研究 writer 粒度、前序信息和整理频率；
- v11：引入 agentic writer/manager、事务同步和检索验证。

当前统一表述不是“完全没有检索算法”，而是：**没有独立语义向量检索器；LLM 控制一组无学习、可审计的检索工具。**
