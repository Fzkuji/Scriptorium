# NativeMem File-Native Multi-View 设计

> 状态：当前设计规范，updated 2026-08-02。Topic Markdown 的逐字段规则以 [`../nativemem-method.html`](../nativemem-method.html) 为准。

NativeMem 使用文本文件保存外部记忆。LLM 负责 Topic/Core 的语义内容、组织路径、证据日期和检索动作；Runtime 负责稳定 ID、来源解析、相对路径、派生视图和事务一致性。本文只定义方法结构，不固定 benchmark、模型或实验参数。

## 1. Memory State

NativeMem 的记忆状态包括完整原始记忆和多种派生视图：

$$M_t=(S_t,T_t,C_t;D_t,R_t,G_t).$$

- `Source Memory` $S_t$：完整、追加式原始交互，使用稳定 source IDs，不进行摘要、覆盖或语义合并。
- `Topical View` $T_t$：LLM 直接编辑的主要语义状态，由 Markdown 目录、文件、章节和段落 blocks 组成。
- `Temporal View` $D_t$：Runtime 从 Topic 中具有年份、月份或完整日期的 evidence footnotes 按原始精度生成的时间视图。
- `Hyperlink Relations` $G_t$：Runtime 从正文 Markdown links 生成的 outbound/backlinks。
- `Recent Memory` $R_t$：始终保留最近写入的 50 条记忆记录；写入第 51 条时按 FIFO 移除最早一条。
- `Core Memory` $C_t$：由 LLM 管理的独立文本文件，保存需要在每次交互中持续提供的信息。

Source、Topic 和 Core 文本是权威记忆状态。Timeline、Recent、Relations、BM25 和 Embedding 均由权威状态生成，不保存唯一事实。

建议的逻辑目录为：

```text
memory/
  sources/       # 完整原始交互
  topics/        # Topical View
  timeline/      # Temporal View
  recent_events.jsonl  # 最近 50 条已写入记录
  relations.json # 从 Topic links 生成的 outbound/backlinks
  core.md        # 持续提供给 Agent 的核心信息
  .nativemem/runtime.json # cursor、触发状态、完整创建顺序
```

当前 V11 查询在内存中从 `topics/` 重建 BM25 与 Embedding 索引，不要求向量数据库，也不在查询期间向 memory tree 写入索引文件。

Runtime 可以从目录、文件和章节生成 compact structure map，供写入和查询使用。它是可重建的结构索引，不是新的权威记忆状态。

Topical View、Temporal View 和 Hyperlink Relations 使用相同 block ID 和 source references，使派生内容能够回到对应原始交互。evidence 可以保存年份、月份、完整日期或 `undated`；前三种按原始精度进入 Temporal View，`undated` 不进入。

Recent Memory 是 Topic blocks 的有限窗口，不是独立事实来源。FIFO 移除只修改 `recent_events.jsonl`，不删除 Source 或 Topic。完整创建顺序保存在 Runtime 状态中，因此窗口外 block 的编辑不会改变其顺序。Core Memory 用于稳定偏好、长期目标、持续任务和必须遵守的约束；当前上限为 2K local-tokenizer tokens。

每个 Topic 记忆单元是一个自然语言段落，段落末尾有且只有一个 `^block-id`。事实后紧跟 evidence footnote，定义形式为 ``[^evidence-id]: Time: `YYYY|YYYY-MM|YYYY-MM-DD|undated`; Sources: <source handles>``，其中时间字段每次只能取一个值，多个 Source handle 用逗号分隔。相对年份结合 Source observation date 解析，例如 2023 年上下文中的“去年”在 footnote 中保存为 `2022`。正文只在事实本身自然包含时间时保留日期，不为复制 `Time` 字段而追加日期。只有无法确定任何年份时才使用 `undated`。Topic 间的相对 Markdown link 必须指向 `#^block-id`，不能只指向文件或 heading。新内容使用 `^new-block-label` 和 `[^new-evidence-label]`，Runtime 在提交前物化 8 位十六进制稳定 ID；若候选 ID 冲突，则递增计数器重新计算直到唯一。Runtime 同时把 marker 间距统一为 `正文[^evidence-id] ^block-id`。任何非标题正文缺少 block ID 或有效 evidence、任何 Source 不存在、Topic link 缺少 block 目标或关系目标不存在，都会使事务失败。

### 1.1 Fact Evolution

NativeMem 不通过覆盖旧事实来维护单一 current state。事实发生变化时，Writer 在 Topic 中保留旧状态并写入具有新时间值和来源的状态；Runtime 随后从具有年份、月份或完整日期的 evidence footnotes 重建 Timeline。例如，用户在两个明确日期分别居住于北京和上海时，两项完整日期 evidence 都保留。

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

每个 session 使用一个 `memory_cursor` 表示已处理到的最后一条消息。Writer 读取本批原始消息、compact structure map 和相关局部文件，通过 shell 直接形成完整 Topic Markdown，并在需要时更新 Core。Runtime 解析本次新增 block IDs，按首次创建顺序刷新 Recent；超过 50 条时只从 Recent 窗口移除最早记录。执行期间新到达的消息留到下一批处理，不建立额外 buffer。

写入 Topical View 时，LLM 根据未来检索位置复用或创建路径、文件、章节和链接，并把语义时间按实际精度写在对应 evidence footnote 中。无法确定任何年份时写 `undated`。Runtime 按年份、月份或完整日期的原始精度生成 Temporal View，不从正文、文件时间或 commit 时间重新推断。

### 2.2 Periodic Local Reorganization

局部整理采用预先确定的周期触发，不依赖当前查询结果。作用域只限周期内发生新增写入的 Topic，不处理无关 Topic 或整体目录。Agent 可以重构这些 Topic 内的目录、文件划分、章节、摘要和链接，同时保留完整事件历史和 source references。

在 benchmark 中，每累计固定数量的完整 session 后执行一次局部整理；整理周期只允许在 development set 上确定，test set 使用固定参数且只执行一次评测。这样可以测量整理机制本身的作用，同时避免根据测试问题反复修改记忆结构。

在实际部署中，Runtime 可以在某个 Topic 累计达到固定数量的 incremental-writing batches 或新增 token 后触发局部整理。阈值只决定何时整理，不根据文件数量、路径或章节长度判断结构质量。按查询成功率、耗时、工具调用次数或 token 成本动态触发整理保留为后续扩展，不属于当前标准方法。

### 2.3 Daily Global Management

触发条件是每日管理时间到达，且最近一次全局管理后存在新的 incremental-writing commit。作用域是完整 Topic/Core 状态，负责跨 Topic 的主题合并、文件拆分、路径调整、重复内容处理、章节整理和链接一致性；派生视图随后由 Runtime 重建。

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

BM25 与 Embedding 接受可选的 `date_from` 和 `date_to`。这两个参数属于同一次检索工具调用，不触发额外 LLM 调用；Agent 只在问题给出明确日历时间约束时填写，格式可以是 `YYYY`、`YYYY-MM` 或 `YYYY-MM-DD`。没有明确时间约束时省略两个参数。对于“搬到上海之前”一类尚未解析成日历时间的语义关系，Agent 先定位锚点事件；在锚点时间确定前不得使用硬时间过滤。无法可靠解析的时间表达也不使用硬过滤。

Runtime 对查询时间执行以下确定性处理：

1. 校验 `date_from` 和 `date_to` 的日历合法性；只有一个参数时形成开放区间，两个参数逆序时拒绝本次调用；
2. 按输入精度展开为半开区间：`2022` 对应 `[2022-01-01, 2023-01-01)`，`2022-07` 对应 `[2022-07-01, 2022-08-01)`，`2022-07-20` 对应 `[2022-07-20, 2022-07-21)`；`date_to` 包含其给定年份、月份或日期的完整范围；
3. 将 Topic 中每项 evidence 的 `YYYY`、`YYYY-MM` 或 `YYYY-MM-DD` 用相同规则展开；候选只要有一项 evidence 与查询区间重叠即可保留；带硬时间约束时不保留只有 `undated` evidence 的候选；
4. 时间过滤完成后，BM25 执行词法排序，Embedding 执行向量相似度排序。没有时间参数时，两种检索器保持原始候选集合。

BM25 与 Embedding 的索引单元是 evidence 支持的事实片段，不是给整个 Topic 段落强行指定一个日期。同一段落在 2019 年和 2023 年分别包含事实时，Runtime 保留两个离散时间区间；不能把它们合成连续的 2019–2023 范围，否则查询 2021 年会产生错误候选。这个设计不增加 Writer 的常规时间字段：Source observation time 仍由 Runtime 保存，Writer 仍只为每项 evidence 写现有的单个语义时间或 `undated`。

### 3.2 Structural Access

除三种检索方法外，Agent 可以使用以下结构化访问工具读取和核验证据：

| 工具 | 作用 |
|---|---|
| `list` / `read` | 浏览目录、文件、章节和局部上下文 |
| `timeline` | 按日期范围读取事件 |
| `follow_link` | 读取正向链接和 backlinks |
| `grep` / `BM25` / `Embedding` | 直接在 Source 文件中检索原始交互 |

工具没有固定调用顺序。Agent 根据每轮返回结果决定继续检索、切换视图、核验原文或回答。

Source Memory 是普通、可检索的 Markdown 文件。`grep`、BM25、Embedding、目录浏览和文件读取始终可以访问 Source，不要求 Agent 先从 Topic 获得 source ID，也不使用单独的 `resolve_sources` 工具。

查询函数接收 `retrieval.QueryConfig.verify_sources` 布尔参数，默认值为 `True`。它只控制 prompt 是否要求 Agent 在回答前核验相关 Source；设为 `False` 时核验变为可选，但 Source 的可见性和所有检索工具保持不变。每题 termination trace 保存实际参数值。命令行使用 `--verify-sources` / `--no-verify-sources`，入口将其转换为同一个配置对象后逐层传入。

### 3.3 Retrieval Budget and Stopping

标准配置为每个 query 设置三个同时生效的上限：

| 约束 | 默认值 | 定义 |
|---|---:|---|
| LLM retrieval rounds | 8 | Agent 判断是否调用工具或结束检索的次数 |
| Tool calls | 5 | `grep`、BM25、Embedding、文件读取、timeline 和 links 的调用总数 |
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

- 校验 block/evidence IDs、source references、Topic-to-Topic block links 和关系目标，执行 Recent 50 条 FIFO 与 Core token 上限；
- 每个具有 `YYYY`、`YYYY-MM` 或 `YYYY-MM-DD` 的 evidence 按原始精度生成一个 Temporal 条目；`undated` evidence 不生成条目；
- 保证 Topic、Timeline、Recent 和 Relations 使用同一 block IDs 与 source references；
- 维护 outbound/backlinks，并在 Topic 文件移动后按目标 ID 更新相对路径；
- 在查询时从文本文件只读重建 BM25 与 Embedding 索引；
- 将 Topic 段落解析为 evidence 支持的事实片段，并由每项 evidence 的原始时间精度生成检索区间；
- 校验检索 Agent 提供的可选 `date_from` / `date_to`，使用区间重叠完成 BM25 与 Embedding 的候选过滤；
- 原子替换 Topic、Timeline、Recent、Relations、Core 与创建顺序，并在失败时恢复上一版本；
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

写后 verification repair 可能改变记忆结构，因此实验通过 `adapter.BuildConfig.verify_writes` 独立控制，并在三种整理策略之间保持相同设置。

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

1. memory block 的理想语义粒度是否需要在 development set 上进一步约束；
2. Writer 12 轮、Manager 8 轮和 Core Memory 2K token 上限是否需要根据 development set 调整；
3. 实际部署中是否需要加入基于查询成功率与检索成本的动态整理触发；
4. 是否采用 Semble 风格的 BM25 + Embedding 融合与结构重排。

这些事项在确定前不引入额外状态字段或后台服务。
