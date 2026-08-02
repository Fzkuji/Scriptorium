# NativeMem v3 — Baseline 设计固化

> 本文档把当前 v3 实现（`src/nativemem.py`）的**每个流程步骤、每个 prompt** 逐字固化下来，作为后续所有优化的**对照基准（baseline）**。任何优化都是在此之上的 diff。
> 固化日期：2026-07-04。对应代码：`src/nativemem.py`（+ `src/adapters/run_nativemem.py`）。

---

## 一句话定义

NativeMem 让 LLM 通过 **bash 工具**（ls/grep/cat/sed/mkdir…）把对话整理进一个**人可读的 markdown 文件夹**，并在同一套文件结构上做检索。核心思想 **RAM（Retrieval-Aligned Memorization）**：写入前先按"以后会怎么检索它"去导航到存放位置，让存和取共用同一套路径。

区别于主流方法（都用向量库/图库 + 固定代码管线，检索靠 embedding 相似度）：NativeMem 无任何检索基建，存储组织权和检索导航权都交给模型。

---

## 完整流程（5 个阶段）

```
对话
 │  切成 10-turn 的 chunk，每个 chunk 原样存入 raw/ 存档
 ▼
[1] 构建 M（process_chunk）  ── SYSTEM_PROMPT
 │    模型用 bash 把 chunk 里的用户事实"照抄"进 markdown 文件，按实体/主题归档
 ▼
[2] 重组（check_and_reorganize）── REORG_PROMPT
 │    构建全部结束后，检测超长文件(>100行)/拥挤目录(>15文件)，让模型拆分
 ▼
[3] 验证（validate_memory）
 │    从 raw/ 随机抽段落 → 生成问题 → 用检索去找 → 判断找没找到/慢不慢
 ▼
[4] 修复（repair_memory）── REPAIR_PROMPT
 │    对验证失败的项（STRUCTURAL/SLOW/MISSING）加交叉引用链接或补存
 ▼
[5] 检索（retrieve）── RETRIEVAL_PROMPT
      答题时模型用 bash 导航文件夹找答案（评测中改为 collect 模式：只收集不作答）
```

**评测适配**（`run_nativemem.py`）：阶段 [1][2] 用于构建；阶段 [5] 改成 **collect 模式**（COLLECT_PROMPT，只收集相关记忆条目、不作答），交给统一 answerer 生成答案。阶段 [3][4] 的验证修复在全量评测里默认不开（`--validate` 才触发）。

---

## 四个核心 Prompt（逐字）

### Prompt 1 — 构建 `SYSTEM_PROMPT`（阶段 1，最核心）

关键设计点：
- **RAM 三步**：THINK（以后会怎么问）→ BROWSE（会去哪找）→ STORE（存在导航到的位置）。
- **内容规则**：`COPY the original conversation text. Do not rephrase, summarize, or generalize.`（← **照抄原文、禁止总结**，这是 baseline 的关键约束，也是优化要动的第一处）
- 每条带 `[YYYY-MM-DD]` 时间戳 + `[source](路径)` 回链。
- 矛盾信息：追加新条目、不覆盖不删除、两版都留。
- **结构规则**：文件/目录名用具体名词；按 people/ groups/ events/ 分类；每个标题唯一（禁止重复标题）；**文件超 100 行就拆**；提到已有实体加 `See also:` 交叉引用；同一主题出现在 ≥3 文件就抽成独立文件。

### Prompt 2 — 重组 `REORG_PROMPT`（阶段 2）

- 触发条件（写死在 `check_and_reorganize`）：文件 > 100 行 → 拆分；目录 > 15 文件 → 分子目录。
- 让模型用 bash（mkdir/mv/sed）拆文件、建索引、更新交叉引用。**只重组不删内容。**

### Prompt 3 — 检索 `RETRIEVAL_PROMPT`（阶段 5，答题）

- 导航策略：`ls -lhR` 看结构 → 挑文件名最相关的 → `grep '^#'` 看标题 → `cat`/`sed` 读段落 → 顺 `See also:` 跨文件 → 第一个没找到换一个。
- 规则：只用文件里的信息（禁用训练知识）；找不到答 "No information available."；答得尽量短。

### Prompt 4 — 修复 `REPAIR_PROMPT`（阶段 4）

三种失败类型对应三种修法：
- **STRUCTURAL**（信息存了但检索走错地方）：在检索去过的位置加一条指向真实位置的交叉引用。
- **SLOW**（找到了但绕路）：在首个访问位置加指向真实位置的链接。
- **MISSING**（根本没存）：按 RAM 找位置补存。
- 原则：不改写已有内容，只加链接/补条目。

### Prompt 5 — 评测检索 `COLLECT_PROMPT`（`run_nativemem.py`，评测专用）

同 RETRIEVAL 的导航策略，但**只收集相关条目、不作答**——逐字抄下带 `[YYYY-MM-DD]` 的条目，包进 `<memories></memories>`，交给统一 answerer。最多 20 条。

---

## Baseline 已知问题（来自 s0 实测归因，2026-07-04）

s0 上 J=70.4（5.4-mini 建+答，4o-mini judge）。152 题错 45 题，归因：

| 错因 | 数量 | 根源 |
|---|---|---|
| 信息翻出来了但错在别处（时间没换算/答错/判严） | 21 | 其中 9 道是**时间没换算**（"上周五"没算成绝对日期） |
| 信息存了但没翻出来 | 17 | **检索**：翻到"差不多但不准"的条目；跨人物文件不查 |
| 信息根本没存进去 | 7 | **构建**：具体细节被概括掉（书名/狗脸杯子/棕榈树） |
| gold 歧义 | 2 | 无解 |

**深层根源**：baseline 明令"照抄原文、禁止总结"，导致——
1. 关键信息淹没在长原话里 → 检索命中率低（17 道）。
2. 没有提炼层 → 时间没在写入时归一化（9 道）。
3. 太像 RAG（存原文、搜原文），缺少"主动加工"这个记忆的本质动作。

**换模型脆弱性**：baseline 把"拆文件/整理"完全交给模型自觉。deepseek 听话（v3 拆成 25 文件），5.4-mini 不拆（堆成 2 个 450 行大文件）→ 构建 token 滚雪球（O(n²)）、检索大海捞针。主流方法用固定代码管线，换模型不崩。

---

## ⚠️ 实测发现：收尾拆文件是负优化（2026-07-04）

原以为"适配器漏了 check_and_reorganize"是 bug，实测证伪：

| 版本 | J | F1 | 构建 |
|---|---|---|---|
| **不拆（2 大文件）← 真 baseline** | **70.4** | 44.2 | 1316k/252/2文件 |
| 收尾拆文件（reorg on） | 53.9 | 35.5 | 1162k/260/3文件 |

收尾拆文件省 12% token 却掉 16 分（每类全跌，搞坏 34 道题）。原因：5.4-mini 拆文件拆得烂——打散条目、丢日期锚点，检索命中更多但更碎更杂（平均 3.0→5.3 条/题，但 q2 塞满 20 条→模型答 Undetermined）。**5.4-mini 不适合自己拆文件。**

结论：**真 baseline = 不拆版 J=70.4**（`results/nativemem-54mini-locomo-s0/`）。适配器 reorg 默认关闭（`NATIVEMEM_REORG=1` 才开）。若要拆文件（O3），必须先解决拆分质量（保住日期锚点、不打散上下文），且应在**构建过程中周期拆**而非收尾拆（收尾拆不救 O(n²) token）。

## 优化方向（在此 baseline 上做 diff，逐项验证）

目标：**双重** —— (1) LoCoMo/LongMemEval 性能好；(2) **跨模型稳**（GPT-5.4-mini 和千问系都要好，证明方法本身强而非蹭模型）。

候选优化（按预期收益排序，每项单独消融）：
- **O1 双层记忆**：把"照抄原文"改成"提炼精炼事实句 + 原文垫底"。提炼层负责被检索命中、原文层负责核对。借鉴 Mem0（禁止概括+专名优先）、Memobase（时间精度自适应）、LightMem（source_id 回指）。预期救 9 道时间 + 7 道细节 + 部分 17 道检索。
- **O2 检索增强**：collect 强制跨两个人物文件都查；用问题关键词直接 grep 正文而非只看标题；聚合题多凑几条再停。预期救部分 17 道检索。
- **O3 代码强制整理**：把"拆文件"从"靠模型自觉"改成"代码强制触发"（借鉴各家用固定管线）。预期解决换模型脆弱性 + 构建效率。

每项优化的验证口径：s0 上重跑 → 对比 J/F1/B1 + 三类错题变化 + 构建效率 + **跨模型**（5.4-mini vs 千问）。

---

## 方法定型（2026-07-05）：自适应加工强度（方向 A）

**核心结论**：最优加工强度取决于 backbone 能力——这是文献空白，也是我们的贡献点。经三个环节反复印证：

| 环节 | 弱模型（千问）| 强模型（5.4-mini）|
|---|---|---|
| 提炼加工 | 分层脚手架 v6（骨架+细节原词+代码补漏）救细节 | 简单双层 v4 就够，v6 的额外约束反伤 |
| 检索 | grep 确定性 | 导航靠理解（弱模型选不准）|
| 存储/整理 | 代码强制结构 + 周期整理（防乱建）| 同 |

**两档最优配置（s0, 4o-mini judge, cat1-4）**：
- **强档 = v4**（NATIVEMEM_PROMPT=v4）：5.4-mini **75.0**，超 LightMem(73)
- **弱档 = v6**（NATIVEMEM_PROMPT=v6，分层加工+代码存储+边整理+few-shot）：千问 **68.4**，超 Mem0(~62)

同口径排位：Full-ctx~80 > 我们强档75 > LightMem73 > 我们弱档68.4 > Mem0~62 > Zep55。**两档都在第一梯队**。

**v6 关键组件**（`src/nativemem.py` + `run_nativemem.py`）：
- 分层加工 `_LAYERED_DISTILL_PROMPT`：骨架总结 + keep_verbatim 专名清单 + verbatim；含 few-shot worked example（规则太软，例子才管用）
- 代码补漏 `verify_details`：只补真专名/数字（不补普通短语，否则噪声伤强模型）
- 代码存储 `store_facts_code`：确定性写 people/<Person>.md（防弱模型乱建深目录树）
- 周期整理 `consolidate_topics`：代码找近义标题候选 → 模型判断合并（83→46）
- 检索：grep 默认；导航 `NATIVEMEM_RETRIEVAL=nav`（强模型待验证）

**基础设施修复**：所有 OpenAI client `trust_env=False`（绕 macOS 系统代理 502）；distill 容错非 JSON（本地代理不支持 json_object）。

**待修评测口径**：judge 对日期格式严（"2023-05-07" 判 ≠ "7 May 2023"），我们方法因记忆存 ISO 日期吃亏更多（12/37 temporal 题输出 ISO vs LightMem 4/37）。**全量跑完后**统一对所有方法做日期归一化重判（现在改会造成新旧口径不一致）。

## 实验结果记录（2026-07-04，s0，4o-mini judge）

| 版本 | 5.4-mini | 千问30B | 备注 |
|---|---|---|---|
| baseline（照抄原文，不拆） | 70.4 | — | 真 baseline |
| 收尾拆文件 | 53.9 ❌ | — | **负优化**，弃（5.4-mini 拆得烂，打散条目丢日期锚点）|
| O1 v4 双层记忆 | **75.0** | 24.3 ❌ | 5.4-mini +4.6（temporal 70→92 大胜）；千问崩溃（62% 空检索）|
| v4 + 精准grep检索 | 73.0 | 48.7 | grep 救千问空检索（94→9题）；5.4-mini 换2分稳健 |
| v4 + grep + TOP_K=8 | 待测 | 54.6 | 收敛检索救千问被淹（+5.9，single 51→63）|

**关键诊断（决定 v5 方向）**：
- 千问 Full-ctx = **80.9**（不弱！和 5.4-mini 的 80.3 一样）→ 千问 54.6 是**方法问题不是模型弱**
- 千问剩余短板 temporal=35：**建记忆时日期算错**（gold "7 May" 存成 "2023-05-06"）
- 根因（agent 代码级分析）：baseline/v4 让模型**一次 bash 会话干 7 件事**（浏览+提炼+算日期+写文件+交叉引用+拆文件…），弱模型认知过载，先丢的就是日期心算和细节保留。Mem0/LightMem 稳是因为提炼是**单步无状态 text→JSON**，日期用结构化字段注入、存储/去重交给代码

## O4（v5）：提炼与存储解耦（NATIVEMEM_PROMPT=v5，待验证）

针对上述根因，把 process_chunk 拆成两个独立调用：
- **Step A `distill_chunk`（无工具）**：喂一段对话 + 结构化 Observation Date 字段（代码 `normalize_date` 转 ISO），只输出 `{facts:[{person,topic,distilled,verbatim}]}` JSON。**日期允许写相对形式**（"the week before 2023-05-08"），不强制心算——最易错的算术从提炼移除，检索时再算
- **Step B `store_facts`（有工具）**：拿提炼好的 JSON，模型只 `ls`+append 到对应人物文件，**不再提炼**、不拆文件、不加交叉引用（留给 check_and_reorganize）

设计取舍：调用次数翻倍（提炼+存储）、割裂了"边提炼边组织"的灵活性；换来弱模型不再一心多用。冒烟（千问）确认：ISO 日期干净、双层格式标准、提炼保细节。**分数待验证**。
