# 消融实验：拆分子目录时是否留同名摘要页（SUMMARY_PAGE）

日期: 2026-07-07
分支: `feature/v7-agent-memory`
脚本: `scripts/exp_summary_page.sh`
原始产物: `results/exp-sumpage-off/`、`results/exp-sumpage-on/`

## 一句话结论

弱模型（qwen3.6-flash）自组织记忆库时，大文件拆成子目录后**不留同名摘要页更好**（`SUMMARY_PAGE=off`）。开摘要页会让弱模型误解指令、把结构做乱、增加检索难度，无净收益。已定为默认 `off`，`on` 仅作强模型实验的可选项保留。

## 背景与研究问题

v7 让模型用 bash 自组织 markdown 记忆库（RAM：检索对齐的记忆化）。当单个文件条目过多（超 `BIG_FILE`）时，rhythm2/rhythm3 的重排 agent 会把它按 `##` 主题拆成子目录。

**RQ**：拆完之后，要不要在父层留一个同名摘要页（wiki summary-style：主文件放每个子主题一行摘要 + 链接）？
- `on`（wiki 式）：检索时先看摘要页，少翻一层；但有同步成本，摘要可能过时误导。
- `off`（纯目录）：只留子目录，子文件名即话题，靠 `ls` + `grep -r` 检索；零同步成本。

哪个净收益高，实测定。

## 实验配置

| 项 | 值 |
|---|---|
| 数据 | LoCoMo sample0，前 2 个 session，3 个问题 |
| 模型 | qwen3.6-flash（阿里云），oneshot 存储模式 |
| 唯一变量 | `NATIVEMEM_SUMMARY_PAGE = off` vs `on` |
| 触发拆分 | `BIG_FILE=4`（人为压低阈值强制拆分；单人文件超 4 条即标 `big` ⚠）|
| 其余 | `REBALANCE=on`（开 rhythm2）、`REORG_GROWTH=off`（不叠加 rhythm3 增长触发）|

> 注：首轮用 `BIG_FILE=12` 未触发任何拆分（最大文件正好 12 条不超标），结论无效；降到 4 后两模式均触发拆分，本记录为有效轮。

## Build 成本（两模式基本持平，非区分因素）

| 模式 | build 时间 | LLM 调用 | tokens_in | tokens_out |
|---|---|---|---|---|
| off | 217s | 51 | 190,486 | 44,691 |
| on  | 213s | 52 | 181,590 | 43,195 |

## 产物结构对比（核心区别）

### off —— 干净两级树
```
memory_sample0/
  caroline/  adoption-plans.md  lgbtq-support.md  personal-life.md
  melanie/   activities.md      family.md         hobbies.md
```
每个主体一目录、目录下按话题分文件，命名规整无冗余。

### on —— 结构混乱
```
memory_sample0/
  Caroline.md            Melanie.md
  adoption.md            education-career.md   lgbtq-support.md
  work-family-balance.md   work-family-balance/index.md   ← 唯一算「同名 .md + /」的
  future-plans/index.md
  hobbies-activities/index.md
  lgbtq-support-group/index.md
```
散 .md 与子目录混杂；弱模型自创 `index.md` 命名（指令要求「父层留同名摘要页」，它却在子目录里建 `index.md`）。

## 三个关键观察

**1. 弱模型没能实现「摘要页」语义。**
on 模式的 `work-family-balance.md`、`future-plans/index.md` 里装的是**完整三行块条目**（含 verbatim 原文 + `[source]` 锚），不是要求的「每个子主题一行摘要 + 链接」。摘要页机制在弱模型上没落地，反而多出一层无意义的目录嵌套。

**2. source 覆盖差异在噪声内，非开关导致。**
按单个 dia_id（拆开多源锚逗号）统计 turn 覆盖：
- off 覆盖 24 个 turn，独有 `D1:6 / D2:15 / D2:16 / D2:17`
- on 覆盖 23 个 turn，独有 `D1:12 / D1:15 / D1:16`
- 两边各有漏的 turn，是每次 distill/组织的随机性，差 1 个在噪声范围内，不能归因于开关。

**3. 检索：off 更稳。**

| 问题 | off steps / mem | on steps / mem |
|---|---|---|
| q0 | 6 / 10 | 6 / 20 |
| q1 | 5 / 2  | 8 / 10 |
| q2 | 6 / 7  | 6 / **1** |

on 的 q2 只捞到 1 条——散文件 + `index.md` 嵌套的乱结构让检索 agent 没翻到该翻的文件。off 三问 mem 都 ≥2，检索路径更短更稳。

## 结论与建议

在弱模型 + 本数据规模下：
- **`SUMMARY_PAGE=on` 有害**：弱模型无法正确生成摘要页，把它当普通内容文件填，制造散文件与目录混杂的乱结构，`index.md` 嵌套增加检索难度。
- **`SUMMARY_PAGE=off` 更好**：拆分后子文件名即话题，两级树干净，检索稳定。
- **默认设 `off`**。摘要页是给强模型/大库设计的优化，弱模型上收益为负。代码保留 `on` 开关不删，作为将来强模型实验的可选项。

## 若写入论文的补强建议

本结论在小样本 + 人为压低 `BIG_FILE=4` 下得出。结构质量差异是确定性、可解释的（弱模型误解指令），可直接采信 default=off。若入论文，建议：
- 全 10 样本、真实 `BIG_FILE`（让文件自然长到超标）复跑，确认结构质量结论稳定；
- 用 32B LLM judge 打 LoCoMo 分（非 teacher-forcing），确认 off 的检索优势转化为答题分优势。

可作为论文中「弱模型自组织记忆库的设计选择」一节的一个消融点：**面向弱模型的记忆结构应偏简单（纯目录 > wiki 式摘要页）**，因为摘要页要求的「维护一致性 + 语义压缩」超出弱模型稳定执行的能力，引入的结构熵抵消了理论上的检索捷径。
