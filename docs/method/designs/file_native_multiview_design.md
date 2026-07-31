# NativeMem File-Native Multi-View 设计

> 状态：当前设计草案，2026-07-31。本文档是方法运行机制的当前规范。

NativeMem 使用文本文件保存外部记忆。LLM 负责记忆内容、组织路径和检索动作；Runtime 负责引用、路径、日期、链接和事务一致性。本文只定义方法结构，不固定 benchmark、模型、prompt 文本或实验参数。

## 1. Memory State

NativeMem 的记忆状态包括完整原始记忆和多种派生视图：

$$M_t=(S_t,A_t,R_t,C_t), \qquad A_t=(T_t,D_t,G_t).$$

- `Source Memory` $S_t$：完整、追加式原始交互，使用稳定 source IDs，不进行摘要、覆盖或语义合并。
- `Topical View` $T_t$：按主题组织的 Markdown 目录、文件和章节。
- `Temporal View` $D_t$：按事件日期组织的时间视图。
- `Hyperlink Relations` $G_t$：由 Markdown links 和 backlinks 表示的跨文件关系。
- `Recent Memory` $R_t$：始终保留最近写入的 50 条记忆记录；写入第 51 条时按 FIFO 移除最早一条。
- `Core Memory` $C_t$：由 LLM 管理的独立文本文件，保存需要在每次交互中持续提供的信息。

文本文件是权威记忆状态。BM25 和 Embedding 索引都从文本文件构建，只用于检索，不保存唯一事实；索引损坏或删除后可以重新生成。

建议的逻辑目录为：

```text
memory/
  sources/       # 完整原始交互
  topics/        # Topical View
  timeline/      # Temporal View
  recent_events.jsonl  # 最近 50 条已写入记录
  core.md        # 持续提供给 Agent 的核心信息
```

当前 V11 查询在内存中从 `topics/` 重建 BM25 与 Embedding 索引，不要求向量数据库，也不在查询期间向 memory tree 写入索引文件。

Runtime 可以从目录、文件和章节生成 compact structure map，供写入和查询使用。它是可重建的结构索引，不是新的权威记忆状态。

Topical View、Temporal View 和 Hyperlink Relations 使用相同 memory ID 和 source references，使派生内容能够回到对应原始交互。无法确定日期的事件保存在 Temporal View 的 `undated` 区域。

Recent Memory 是长期派生记忆的有限窗口，不是独立事实来源。FIFO 移除只修改 `recent_events.jsonl`，不删除 Source Memory、Topical View 或 Temporal View 中的记录。Core Memory 用于稳定偏好、长期目标、持续任务和必须遵守的约束；LLM 可以修改该文件，但其内容必须能够回到长期记忆或 source references。当前 Core Memory 上限为 2K local-tokenizer tokens。

### 1.1 Fact Evolution

NativeMem 不通过覆盖旧事实来维护单一 current state。事实发生变化时，Writer 在 Topical View 和 Temporal View 中继续记录新的带时间事件，并保留原有事件及其 source references。例如，用户上个月住在北京、这个月搬到上海，两条记录都保留；查询时由 LLM 根据问题中的时间范围和事件顺序作答。

纠错也作为完整过程记录：先前表达及其 source 保留，后续记录明确说明用户进行了纠正。Source Memory 始终保存全部原始交互，Git 保存派生文件的修改版本。因此，当前设计不增加 `supersedes`、有效区间或独立 current-state 数据库。

| 对标对象 | 新旧事实处理 | NativeMem 的区别 |
|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 对 memory item 执行 add、update、delete 或 no-op | NativeMem 不通过删除旧 item 只保留当前值 |
| [Infini Memory](https://arxiv.org/html/2606.10677) | 重写相关 Topic Document，并记录 update relation | NativeMem 保留带时间的完整变化记录 |
| [Zep / Graphiti](https://help.getzep.com/graphiti/core-concepts/adding-episodes) | 使用 temporal edges、有效区间和 edge invalidation | NativeMem 使用文本事件、时间视图和 LLM时间判断，不维护图边有效期 |
| [LightMem](https://arxiv.org/html/2604.07798) | 根据时间与证据执行 merge、update 或 drop | NativeMem 不删除 Source Memory，也不要求丢弃旧事件 |
| [ByteRover](https://arxiv.org/html/2604.01599) | 使用 UPDATE、UPSERT、MERGE 和 DELETE 修改 Markdown entries | NativeMem 默认追加事实变化，结构整理不删除原始证据 |

## 2. Three-Level Memory Management

NativeMem 只采用三级记忆管理。

这里的三级表示三种管理时机和作用域，不是三种记忆存储层级。

### 2.1 Incremental Writing

触发条件是 cursor 后未处理内容达到规定 token 或上下文规模，或者 session 一小时没有新消息。作用域仅限本批新增消息及其直接涉及的局部记忆。

每个 session 使用一个 `memory_cursor` 表示已处理到的最后一条消息。Writer 读取本批原始消息、compact structure map 和相关局部文件，自行决定写入内容、topic path、章节、日期和 links，并在需要时更新 Core Memory。Runtime 将新写入的记忆记录追加到 Recent Memory；超过 50 条时按 FIFO 移除最早记录。执行期间新到达的消息留到下一批处理，不建立额外 buffer。

写入 Topical View 时，LLM 根据未来检索这条记忆时最可能访问的位置复用或创建路径、文件、章节和链接。LLM 判断事件时间，Runtime 将通过校验的日期写入 Temporal View。

### 2.2 Periodic Local Reorganization

局部整理采用预先确定的周期触发，不依赖当前查询结果。作用域只限周期内发生新增写入的 Topic，不处理无关 Topic 或整体目录。Agent 可以重构这些 Topic 内的目录、文件划分、章节、摘要和链接，同时保留完整事件历史和 source references。

在 benchmark 中，每累计固定数量的完整 session 后执行一次局部整理；整理周期只允许在 development set 上确定，test set 使用固定参数且只执行一次评测。这样可以测量整理机制本身的作用，同时避免根据测试问题反复修改记忆结构。

在实际部署中，Runtime 可以在某个 Topic 累计达到固定数量的 incremental-writing batches 或新增 token 后触发局部整理。阈值只决定何时整理，不根据文件数量、路径或章节长度判断结构质量。按查询成功率、耗时、工具调用次数或 token 成本动态触发整理保留为后续扩展，不属于当前标准方法。

### 2.3 Daily Global Management

触发条件是每日管理时间到达，且最近一次全局管理后存在新的 incremental-writing commit。作用域是完整派生记忆，负责跨 Topic 的主题合并、文件拆分、路径调整、重复内容处理、章节整理和链接一致性。

如果没有新的 incremental-writing commit，本次管理直接结束。全局管理不重新处理 Source Memory，也不改变 session cursor。

当前 benchmark 实现将 Writer/verification agent 的外层上限设为 12 个模型轮次，将局部与全局 Manager 设为 8 个模型轮次；达到上限时在 audit 中记录 `round_limit`。这些上限用于限制异常循环，不替代后续的构建成本消融。

## 3. Query-Time Access

查询开始时，Agent 获得用户问题、Recent Memory、Core Memory 和 compact structure map，而不是全部记忆正文。当前规范只允许查询读取 committed tree。

### 3.1 Retrieval Methods

标准配置提供三种并列的检索方法，由 Agent 自主决定是否调用、使用什么 query、返回多少候选以及是否继续检索：

| 方法 | 工具 | 作用 |
|---|---|---|
| Exact lexical retrieval | `grep` | 查找名称、数字、编号、原句和精确短语 |
| Sparse lexical retrieval | `bm25_search` | 使用倒排索引返回按词法相关性排序的候选 |
| Dense semantic retrieval | `embedding_search` | 使用 Embedding 索引返回按语义相似度排序的候选 |

三种方法都只返回候选。Agent 负责判断相关性、改写 query、组合多次结果以及决定何时停止。系统不规定先执行 BM25 或 Embedding，也不直接拿用户原始问题固定运行某一种检索。

### 3.2 Structural Access

除三种检索方法外，Agent 可以使用以下结构化访问工具读取和核验证据：

| 工具 | 作用 |
|---|---|
| `list` / `read` | 浏览目录、文件、章节和局部上下文 |
| `timeline` | 按日期范围读取事件 |
| `follow_link` | 读取正向链接和 backlinks |
| `read_source` | 根据 source references 核验原始交互 |

工具没有固定调用顺序。Agent 根据每轮返回结果决定继续检索、切换视图、核验原文或回答。

查询不能无约束扫描全部 Source Memory。Agent 先在派生视图中定位 source references，再通过 `read_source` 读取选定原文及有限邻近上下文。

### 3.3 Retrieval Budget and Stopping

标准配置为每个 query 设置三个同时生效的上限：

| 约束 | 默认值 | 定义 |
|---|---:|---|
| LLM retrieval rounds | 8 | Agent 判断是否调用工具或结束检索的次数 |
| Tool calls | 5 | `grep`、BM25、Embedding、文件读取、timeline、links 和 source resolution 的调用总数 |
| Memory-visible tokens | 10K | Recent Memory、Core Memory、structure map 及全部工具返回给 Agent 的记忆内容总量 |

单次 BM25 或 Embedding 默认最多返回 10 个候选。文件读取优先返回一个完整 heading block 或有限行范围，不默认读取整个文件。工具结果超过剩余 token 预算时由 Runtime 截断，并保留路径、日期和 source references。

Agent 满足以下任一条件时停止检索：

1. 已获得足以支持答案的证据，并能给出相应 source references；
2. 连续两次工具调用没有增加新的 source references 或新的规范化证据行；
3. 任一硬上限耗尽。

正常停止时，Agent 根据已有证据回答。达到硬上限但证据不足时，Agent明确返回信息不足，不继续调用工具。wall-clock time 只记录为评测指标，不作为 benchmark 的停止条件，避免不同 API 延迟改变检索行为。

默认值不是方法结论。development set 比较 `3/5/8` 次 tool-call limits 和 `6K/10K/20K` memory-visible-token budgets，在不降低任务成功率的配置中选择成本最低者；test set 固定所选参数，只运行一次。8 个 retrieval rounds 作为防止异常循环的外层上限，不单独调参。

### 3.4 Reference Systems

| 对标对象 | 对应技术 | 与 NativeMem 的比较 |
|---|---|---|
| [ByteRover](https://arxiv.org/html/2604.01599) | 5-Tier Progressive Retrieval：exact cache、fuzzy cache、BM25 direct response、single LLM、full agentic search | 主要策略对标。ByteRover 使用固定自动路由；NativeMem 由 Agent 自主选择 grep、BM25、Embedding 和文件工具 |
| [Infini Memory-H](https://arxiv.org/html/2606.10677) | LLM summary selection + BM25 partition retrieval | 固定混合检索对标 |
| [Infini Memory-A](https://arxiv.org/html/2606.10677) | Agent 调用 `grep`、`search`、`list_docs`、`read_lines` | Agent-controlled retrieval 对标 |
| [LightMem](https://arxiv.org/html/2604.07798) | query rewriting/routing + Embedding top-k + semantic reranking | Embedding 检索与模型控制对标 |
| [Semble](https://github.com/MinishLab/semble) | BM25 + static Embedding + RRF + code-aware reranking | 后续混合检索增强参考，不作为当前标准方法 |

ByteRover 的完整 agentic fallback 允许最多 50 次迭代；LightMem 的粗检索固定返回 10 个候选；Infini Memory 同时使用最大迭代数、evidence budget 和无新增证据停止。NativeMem 采用更小的默认上限，并通过 development-set budget sweep 确定正式配置。

后续可以参考 Semble，将 BM25 与 Embedding 排名通过 RRF 融合，再使用 topic path、日期、source references 和链接等记忆结构信号重排。该增强需要与三种独立工具配置分别比较，验证后才能加入标准方法。

## 4. Runtime Guarantees

当前 V11 benchmark Runtime 已执行以下确定性约束，不替代 LLM 做语义组织决策：

- 校验 source references、路径、日期和链接，执行 Recent Memory 的 50 条 FIFO 容量限制与 Core Memory token 上限；
- 根据 LLM 给出的事件日期更新 Temporal View，无法确定日期时写入 `undated`；
- 保证 Topical View、Temporal View 和 links 中的 memory IDs 与 source references 一致；
- 维护 backlinks，并在文件移动后更新相关路径；
- 在查询时从文本文件只读重建 BM25 与 Embedding 索引；
- 原子替换 Topic、Timeline、Core 与 Recent 状态，并在失败时恢复上一版本；
- 对局部整理、全局整理和查询循环执行独立轮数或 token 预算。

Git commit、session cursor、按 token 增量触发、一小时空闲触发和每日管理条件已在在线 Runtime 中实现。当前静态 benchmark harness 不启动长时间 scheduler 进程，仍使用固定 session 边界和固定局部/最终管理映射，因此在线触发保证不能归属于既有 benchmark 结果。

系统不增加 batch ID、`memory_dirty`、独立写入日志或额外 CURRENT buffer。

## 5. Evaluation

记忆系统首先按任务是否成功评估。任务未解决即为失败；任务成功后，使用以下指标衡量效率：

- 完成时间；
- 工具调用次数；
- LLM 输入与输出 token；
- 工具返回的可见 token；
- 可直接计费的模型和工具成本。

周期性局部整理通过固定构建策略的消融实验评估。整理频率只能使用 development set 确定，不能先观察 test query 的结果再调整结构或重新回答。至少比较以下三种构建策略：

| 构建策略 | 整理方式 |
|---|---|
| No local reorganization | 增量写入后不执行局部整理 |
| Final-only reorganization | 所有 session 写入完成后只整理一次 |
| Periodic local reorganization | 每累计固定数量的 session，整理本周期发生新增写入的 Topic |

三种策略使用相同输入顺序、记忆内容、Writer、Retriever、Answerer 和测试预算。首先比较任务成功率；任务成功时，再比较完成时间、工具调用次数和 token 成本。周期整理只有在不降低任务成功率的前提下改善这些指标，才能认为有效。不额外评估路径是否正确，也不诊断失败属于证据缺失还是检索选择错误。

写后 verification repair 可能改变记忆结构，因此实验通过 `NATIVEMEM_V11_VERIFY` 独立开关控制，并在三种整理策略之间保持相同设置。

检索实验固定同一份 memory、answerer、judge 和预算，至少比较以下配置：

| 配置 | 检索策略 |
|---|---|
| File-only | Agent 使用目录、文件和 grep |
| BM25-only | 每个 query 固定执行 BM25，再由 LLM回答 |
| Embedding-only | 每个 query 固定执行 Embedding search，再由 LLM回答 |
| NativeMem Retrieval | Agent 自主选择 grep、BM25、Embedding 和结构化访问工具 |
| ByteRover-style | exact/fuzzy cache、BM25、single LLM、full agentic search 的固定五级路由 |
| Semble-style | BM25 + Embedding + RRF + 记忆结构重排 |

主要比较 NativeMem Retrieval 与 ByteRover-style；其余配置用于判断增益来自具体检索器、Agent控制还是混合重排。

所有配置必须记录逐题 retrieval rounds、tool calls、memory-visible tokens、最终证据 tokens、端到端时间和模型调用成本。除方法本身具有固定预算外，主比较统一使用 development set 选定的 NativeMem 预算；另报告 `3/5/8` tool calls 与 `6K/10K/20K` visible tokens 的预算曲线。

现有 LoCoMo 和 LongMemEval-S 结果只能证明已经完成实验的版本。Recent Memory、Core Memory、周期性局部整理、BM25、Embedding、Git/cursor 事务与在线触发均已进入代码，但仍需要分别验证；新增实现不能由既有结果直接证明。

## 6. Open Questions

当前只保留会影响实现或实验定义的问题：

1. Topical View 与 Temporal View 使用同步文本 occurrence，还是规范记录加视图链接；
2. Writer 12 轮、Manager 8 轮和 Core Memory 2K token 上限是否需要根据 development set 调整；
3. 实际部署中是否需要加入基于查询成功率与检索成本的动态整理触发；
4. 是否采用 Semble 风格的 BM25 + Embedding 融合与结构重排。

这些事项在确定前不引入额外状态字段或后台服务。
