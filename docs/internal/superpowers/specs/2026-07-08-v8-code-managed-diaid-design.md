# v8 优化：dia_id 全程代码托管（提炼带号 + 检索工具必填 id）

日期: 2026-07-08
迭代对象: v8（双视图 + 回原文），修两个弱模型短板：提炼漏标 dia_id（6/9 空）、检索抄不对 dia_id。

## 一句话

dia_id 从"模型手记/手抄"变成"代码托管，模型只做确认"：
- **提炼**：喂 chunk 时每句带行号 → 模型事件里写行号(refs，至少一个) → 代码把行号转成真实 dia_id 存进事件行。模型碰不到 dia_id 格式。
- **检索**：给模型一个 `read_original(dia_id)` 工具，**dia_id 是必填参数** → 模型 grep 到事件行（行尾 `· [D1:3]`）后必须把 id 填进工具才能拿到原文。填不出就调不了，逼它输出 id。

## 为什么

现状根因：`chunk_text` 是纯 `speaker: text`，**不含任何 id**。模型看不到句子↔id 的对应，只能凭空写 dia_id → 弱模型要么漏（空 []）、要么写错（裸数字，已修但治标）。检索侧靠模型把 id 抄进自由文本、代码正则抠，弱模型抄漏。

改法把 id 的"生成/传递"交给代码，模型只负责它擅长的判断（哪句相关、哪条事件相关），不负责它不擅长的（记 id 格式、手抄 id）。

## 改动 1：提炼带行号 + refs 必填

### chunk 文本带行号
`split_into_chunks` 或提炼前，给 chunk 每句加一个**局部行号**（1-based，chunk 内），并保留"行号 → 真实 dia_id"的映射：
```
[1] Caroline: hi
[2] Melanie: hello
[3] Caroline: I went to a LGBTQ support group yesterday
[4] Melanie: wow that's brave
[5] Caroline: it gave me courage to embrace myself
```
映射：`{1:"D1:1", 2:"D1:2", 3:"D1:3", ...}`（来自 split 已有的 dia_ids 列表，按顺序对应）。

### 模型只写行号
`_V8_DISTILL_PROMPT` 改：事件不再要 `dia_ids`，改要 **`refs`（这条事件对应的行号，至少写一个）**。
```json
{"events":[{"when":"2023-05-07","summary":"...","refs":[3,5],"topic":"..."}]}
```
prompt 明确：refs 必须至少一个行号；就是上面每句开头 `[n]` 的那个数字。

### 代码转 dia_id
`distill_events` 拿到事件的 `refs`（行号列表）→ 用行号→dia_id 映射转成真实 dia_ids → 存进事件（下游 write_events 不变）。
- refs 为空的事件：丢弃（无法回原文的事件没意义），或按 max_retry 重试一次。
- refs 里越界/无效行号：丢弃该号，保留有效的；全无效则丢弃该事件。

### 精度
行号是**句级**的，chunk 多大都不影响 dia_id 精度——模型指第 3、5 句，就只挂 D1:3、D1:5，回原文只取这两句，不取整个 chunk。

## 改动 2：检索改用 read_original 工具（必填 dia_id）

### 现状
`_collect_v8` 让模型 grep 后把 dia_id 抄进最终回答文本，代码 `_DIA_ID_RE` 正则抠 → 抄漏就取不到。

### 改成工具
除了现有 bash（grep/cat 查视图），加一个工具：
```
read_original(dia_ids: list[str])   # 必填
  说明：读事件对应的原始对话。dia_ids 从事件行末尾的 [D1:3, D1:5] 里取。
  返回：这些 dia_id 对应的原始对话原文（含前后一句上下文）。
```
检索流程：模型 grep/cat 视图 → 找到相关事件行 → **调 read_original 并把行尾的 dia_id 填进去** → 工具用 `read_turns(turn_index, dia_ids)` 回原文返回给模型 → 模型据此收尾。

`_collect_v8` 收集 read_original 返回过的原文片段作为 memories（不再靠正则从最终文本抠 id）。

### 为什么更稳
- dia_id 是工具**必填参数**，模型不填就调不了工具、拿不到原文 → 逼它输出 id。
- 工具调用（结构化 JSON 参数）比自由文本抄写可靠。
- 代码在工具里做 dia_id → 原文，模型不需要理解格式。

## 边界

- 提炼 refs 全空的 chunk：该 chunk 无有效事件，跳过（记数便于观察弱模型漏标率）。
- read_original 收到无效/不存在 dia_id：跳过无效的，返回能取到的（read_turns 已有此行为）。
- 行号映射：严格按 split 的 dia_ids 顺序 1:1，chunk 内第 i 句 = dia_ids[i-1]。
- 向后兼容：仅改 v8；v4/v6/v7 不动。

## 验收

- 冒烟重跑：timeline/topics 里**事件行 dia_id 空 [] 的比例大幅下降**（当前 6/9 → 目标 <2/9）。
- 检索非空题数从 1/3 提升（同样小样本 + max-sessions 限制下尽量多命中）。
- q2 那类回原文链路仍通（不回退）。
- 全套单测通过（含新的行号转换、refs 必填、read_original 工具测试）。
