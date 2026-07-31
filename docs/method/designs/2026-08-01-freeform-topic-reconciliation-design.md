# NativeMem 自由 Topic 编辑与自动记忆对齐设计

> 状态：待实现的目标规范，2026-08-01。本文只定义 Topic Markdown 的直接编辑、记忆身份恢复和派生视图同步；不修改检索方法、调度周期或 benchmark 参数。

## 1. 目标

NativeMem 允许 LLM 像编辑普通 Markdown 一样修改 `topics/`：重写段落、重命名标题、调整顺序、移动内容并补充新事实。编辑者不需要为普通更新调用 `update_memory`，也不需要提交逐字段 patch。

系统同时保留以下性质：

- 每条可追踪记忆具有稳定 `memory_id`；
- 新事实必须能够回到真实 Source；
- Topic 修改后，Timeline、Recent 和可重建索引自动同步；
- 普通改写不能意外删除已有记忆；
- 校验失败时不提交部分状态。

## 2. 状态与职责

### 2.1 LLM 的职责

编辑 Agent 直接修改 Topic Markdown，负责正文表达和组织决策：

- 创建、重命名和移动标题；
- 改写、合并和拆分段落；
- 调整记忆在 Topic 中的出现顺序；
- 将内容移动到其他文件或章节；
- 根据当前交互和已读取证据补充新内容。

编辑 Agent 不负责生成稳定 ID、计算相对链接、同步其他视图或提交 Git 事务。

### 2.2 Runtime 的职责

Runtime 负责：

- 在修改前保存受影响文件和记忆目录快照；
- 计算修改前后的局部 diff；
- 直接处理不改变事实正文的结构性修改；
- 在事实正文新增、删除或变化时调用 Reconciler；
- 校验 Reconciler 返回的正文范围、日期和来源；
- 分配新 `memory_id` 并生成 Markdown 引用；
- 重建 Timeline、Recent 和索引；
- 原子替换文件并创建 Git commit；
- 任一步失败时恢复修改前状态。

### 2.3 Reconciler 的职责

Reconciler 只解决代码不能确定的语义对应关系：

- 新文本应拆成几条记忆；
- 新记忆由哪些候选来源支持；
- 消失的旧引用对应改写、合并、移动还是无效修改；
- 新事件的发生时间是什么。

Reconciler 不直接写文件，不生成路径，不创建 Source，也不能删除 Source。

## 3. Topic Markdown 表示

Topic 正文采用自由 Markdown。`memory_id` 使用行内 footnote 引用，来源定义集中放在文件底部。

```md
## 居住经历

用户于 2026 年 7 月搬到上海。[^mem_7f31c2]
同年 8 月搬到浦东，并开始在新公司工作。[^mem_21ab90][^mem_a8d910]

[^mem_7f31c2]: 2026-07-20 · Sources: [项目迁移讨论 · “我准备下个月搬到上海……”](../../sources/thread_123.md#msg_a8f2)
[^mem_21ab90]: 2026-08-03 · Sources: [居住地点更新 · “现在已经搬到浦东……”](../../sources/thread_456.md#msg_b31d)
[^mem_a8d910]: 2026-08-05 · Sources: [入职讨论 · “今天正式入职……”](../../sources/thread_456.md#msg_c902)
```

正文与引用采用以下确定性绑定规则：一个 Markdown 正文段落被解析为交替出现的正文片段和引用组；引用组是一个或多个相邻的 `[^mem_*]`。每个引用组对应从段落开头或上一引用组结束位置开始、到当前引用组之前结束的正文片段。相邻的多个 ID 共同对应同一片段。引用组之后如果还有正文，则该正文必须由后续引用组结束；修改后新增且没有引用的正文交给 Reconciler 分类和补全。标题、代码块和文件底部的 footnote 定义不参与正文片段解析；列表项和引用块分别作为独立正文段落处理。

其他规则如下：

- 一个 `memory_id` 在全部 Topic 主视图中只能有一个主要正文位置；
- 标题和纯组织说明可以没有引用；包含外部事实的正文必须有引用；
- footnote 的可读标题由 Runtime 根据 session title、消息开头、文档标题、文件名和定位信息生成；
- 链接目标使用稳定的原始 `thread_id + message_id` 或文件路径与定位信息；
- Source 可以有多个，Source 文件始终保持追加且不可修改。

footnote 是目标存储语法，不因渲染器不同而切换成另一种权威格式。不支持 footnote 的纯 Markdown 查看器可以显示原始标记，但不影响 Runtime 解析；文档预览集成应单独验证 GitHub 与 Obsidian 的显示结果。

## 4. 修改后的处理流程

### 4.1 修改捕获

Agent 的写文件工具只操作 staged Topic tree。每次写操作结束后，Runtime 获得：

- 修改前和修改后的文件；
- 受影响的标题范围；
- 当前文件中的全部 `memory_id`；
- 修改前目录中对应 ID 的日期、来源和派生位置；
- 本轮 Agent 实际读取过的 Source 与外部文件；
- 当前 session cursor 后尚未处理的原始消息。

Runtime 不监听无关仓库文件，也不在每次键盘输入后执行同步。同步边界是一次 Agent 文件写操作完成时。

### 4.2 确定性处理

同时满足以下条件时不调用 Reconciler：

- 旧 ID 全部保留；
- 每个 ID 对应的规范化正文均未改变；
- 没有新增、删除或改变未引用的普通正文；
- 每个 ID 仍有唯一正文位置；
- Source 定义仍然有效。

Runtime 直接从新位置更新 Topic path 和 heading path。标题重命名、整段移动、顺序调整以及不改变正文语义内容的空白或 Markdown 格式修改属于此类。仅保留 ID 但改变其对应正文并不足以跳过 Reconciler，因为新正文可能增加了第二条事实。

### 4.3 自动语义对齐

出现下列任一情况时，Runtime 对所有相关 diff 一次性调用 Reconciler：

- 新增了没有稳定 ID 的事实文本；
- 旧 ID 从原位置消失；
- 已有 ID 对应的正文发生变化；
- 新增、删除或改变了未引用的普通正文；
- 一个旧文本被拆成多个事实；
- 多个旧文本被合并；
- 新内容需要选择来源或确定日期。

Reconciler 只接收修改过的章节、受影响的旧记忆和有限候选来源，不接收完整记忆库。

候选来源仅包括：

1. 受影响旧记忆已经引用的 Source；
2. 当前 session cursor 后的新增消息；
3. 编辑 Agent 本轮通过受控工具读取过的 Source、文件和文件定位结果。

Reconciler 不能引用候选集合之外的 ID。找不到有效来源的新事实不能提交。

### 4.4 Reconciler 内部接口

Reconciler 返回结构化结果，但该格式只存在于 Runtime 内部，编辑 Agent 不需要生成。

```json
{
  "matches": [
    {
      "memory_id": "mem_a",
      "quote": "用户于 7 月搬到上海",
      "when": "2026-07-20",
      "source_refs": ["src_old_a"]
    }
  ],
  "creates": [
    {
      "quote": "8 月搬到浦东",
      "when": "2026-08-03",
      "source_refs": ["src_17"]
    },
    {
      "quote": "开始在新公司工作",
      "when": "2026-08-05",
      "source_refs": ["src_23"]
    }
  ],
  "organizational_quotes": []
}
```

`quote` 必须逐字存在于修改后的章节。Runtime 根据章节与 diff 确定 occurrence；相同文本出现多次且无法唯一定位时拒绝结果。`matches` 对受影响旧 ID 返回完整日期与来源；内容未改动的旧记忆不进入请求。`organizational_quotes` 只允许列出标题说明、过渡文字或不陈述外部事实的文本，这些文字不生成记忆 ID。`src_*` 是本次 reconciliation 的临时候选 handle，持久化前转换为稳定来源定位。

## 5. 编辑语义

### 5.1 改写与移动

仅移动内容时，Runtime 直接更新位置，不改变 ID、创建顺序和 Recent 顺序。正文改变时由 Reconciler 判断它是同一事实的表达更新，还是包含需要独立 ID 的新事实；确认是同一事实后才保留原 ID并同步正文与派生视图。

### 5.2 新事实

Reconciler 为每条新事实返回正文范围、时间和一个或多个来源。Runtime 分配 `mem_*`，插入引用和 footnote 定义，再更新 Timeline 与 Recent。

### 5.3 合并

多个旧事实被写成一段连贯文本时，原 ID 均保留并连续附在该文本后。Topic 只显示一次合并文本，索引仍记录全部 ID 与各自来源。Timeline 只在这些 ID 具有相同事件日期时合并显示；日期不同则分别出现在对应日期下。系统不自动把多个旧 ID 替换成一个新 ID。

### 5.4 拆分

旧事实被拆成多个独立事实时，原 ID 保留给与旧语义最接近且来源不变的部分；其余部分只有在存在有效来源时创建新 ID。无法可靠对应时拒绝提交并保留旧版本。

### 5.5 状态变化与纠错

现实状态后来发生变化时，新增带时间记忆，不覆盖旧事件。例如先住上海、后来住杭州，应保留两条记录。只有先前派生内容确实提取错误时才允许纠错修改原 ID。

### 5.6 删除

普通文件改写不能隐式删除旧 ID。旧 ID 消失且没有在其他 Topic 位置找到时，本次提交失败。删除错误派生记忆必须通过显式纠错删除操作；该操作只删除 Topic、Timeline、Recent 和索引中的派生记录，不删除 Source。

## 6. 顺序定义

NativeMem 分别维护四种顺序：

- **Source 顺序**：原始消息数组、JSONL 文件或原始时间戳的顺序，不可由 Topic 编辑改变。
- **Topic 顺序**：引用在 Markdown 中的出现顺序，由 LLM 的组织结果决定。
- **Timeline 顺序**：先按事件 `when` 排序；相同事件时间按主要 Source 时间排序；仍相同时按 `memory_id` 排序，保证确定性。
- **Recent 顺序**：按记忆首次创建顺序维护。普通改写、移动和标题调整不改变顺序；新 ID 追加到尾部并执行固定容量 FIFO。

footnote 定义在文件底部按对应引用第一次出现的顺序重排，只影响可读性，不改变以上任何语义顺序。

## 7. 校验、事务与失败处理

Runtime 在提交前执行以下检查：

- 所有旧 ID 均被保留、移动或通过显式纠错删除；
- 每个 ID 只有一个主要正文位置；
- Reconciler 返回的正文是修改后文本中的精确片段；
- 新 ID 的来源均属于候选来源集合且实际存在；
- message、文件、页码、章节或行号定位可以解析；
- 日期合法，缺少事件日期时明确进入 `undated`；
- footnote 引用与定义一一对应；
- Source tree 没有修改；
- Timeline、Recent 和索引可以由新 Topic 状态完整生成。

无效 Reconciler 输出最多使用校验错误重试一次。第二次仍无效、没有来源、正文定位有歧义或跨视图生成失败时，整次修改回滚。系统不得提交只有 Topic 更新而派生视图未更新的中间状态。

## 8. 调用与成本控制

Reconciler 不是每次写文件都调用。只改变标题、路径、顺序、空白或 Markdown 格式的修改由代码处理；任何事实正文变化，以及新增、引用缺失、拆分、合并或来源变化，会产生一次额外 LLM 调用。这一调用是允许完全自由编辑正文同时维持记忆边界所需的语义处理成本。

同一次文件写操作涉及多个相关章节时，Runtime 将这些 diff 合并成一次 reconciliation 请求。请求只包含受影响内容和有限候选来源，输出只包含引用对齐结果。构建成本报告必须分别记录 Writer、Manager 和 Reconciler 的调用次数、输入输出 token、延迟与价格。

如果部署环境更重视成本，可以要求编辑 Agent 自行保留或生成引用，从而减少 Reconciler 调用；这属于接口配置，不改变 Topic、ID 和同步语义。标准方法采用自由编辑与自动 reconciliation。

## 9. 当前 V11 与目标规范的差异

当前 V11 已有以下可复用机制：

- Topic staged tree；
- shell 修改后解析 Topic 位置和正文；
- Timeline 与 Recent 自动同步；
- 重复 ID、缺失 ID 和 Source 引用校验；
- 文件级原子替换与失败恢复。

当前 V11 已使用 footnote memory ID 解析自由 Markdown，并从 Topic 重建 catalog；Timeline JSON comment 不再是恢复权威。Runtime 已加入 diff-based reconciliation、候选来源审计、一次无效输出重试和显式纠错删除。为兼容既有 benchmark 解析器，Writer 仍输出 `memory-event` HTML comment，但该 marker 不参与新版 Topic 权威解析，后续可以在旧检索器完成迁移后删除。

## 10. 最小验证

实现至少覆盖以下测试：

1. 重命名标题或移动带 ID 的段落不调用 Reconciler，Timeline 路径自动更新；
2. 保留 ID 的正文改写触发 Reconciler；确认是同一事实后同步到 Timeline 与 Recent；
3. 新事实触发一次 Reconciler，生成新 ID、来源定义和 Recent 记录；
4. 同一文本后的多个 ID 在 Topic 中只渲染一次正文；Timeline 只合并相同日期的记录；
5. 拆分后保留一个旧 ID，并只为有来源的新事实创建新 ID；
6. 相同日期事件按 Source 时间和 ID 稳定排序；
7. 普通编辑删除旧 ID 时拒绝提交并恢复文件；
8. 虚构 source handle、歧义 quote、重复 ID 或无来源事实导致回滚；
9. Reconciler 第二次返回无效结果后停止，不提交部分状态；
10. Source tree 在所有更新、合并、拆分和纠错删除中保持不变。

验收标准是：编辑 Agent 只执行普通 Topic 文件编辑，最终 committed state 中的 Topic、Timeline、Recent、索引和 Source references 一致；需要语义恢复时由 Runtime 自动完成，不要求编辑 Agent 额外调用记忆更新工具。
