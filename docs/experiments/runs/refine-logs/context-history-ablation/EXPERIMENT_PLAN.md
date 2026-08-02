# v10 前序上下文形式与预算实验计划

**Problem**: memory writer 每 6 条消息写入一次时，是否需要看到同一 session 的前序信息；如果需要，应使用原始消息、已提炼事件还是滚动摘要，以及保留多少。
**Method Thesis**: 前序上下文只应在提高事实覆盖或最终问答质量时使用；选择策略必须同时考虑质量、输入 token、输出 token、调用数和时间。
**Date**: 2026-07-17

## Claim Map

| Claim | Minimum evidence | Blocks |
|---|---|---|
| C1：同一 writer 下，前序上下文的形式和预算会改变记忆质量与建库成本。 | 固定除 context 外的所有参数，在至少两个 LoCoMo conversation 上比较 build evidence recall、QA 和成本。 | B1-B3 |
| C2：context 策略的排序可能随 writer 模型变化。 | GPT-5.5 与 MiniMax 使用完全相同的矩阵，固定下游 retriever/answerer，检验 `model × context` 交互。 | B2-B4 |
| Anti-claim：更长历史天然更好，或不同形式的 `20/50` 是相同预算。 | 同时报 none；分别标明 items、turns、words，并报告实际 build input tokens。 | B1-B4 |

## 冻结变量

- Dataset：LoCoMo `locomo10.json`，SHA256 记录在运行 manifest。
- Writer interval：每 6 条消息一次，`NATIVEMEM_V10_WRITE_TURNS=6`。
- Calendar：开启，使用同一 v10 writer prompt。
- Verify：开启；每个 session 1 次整理，全部 session 后 1 次 final 整理。
- 整理并发：1，避免不同服务的速率限制改变失败率。
- Memory representation、topic placement、retrieval、answerer 均不改变。
- Judge 只允许 `scripts/eval_full.py`，要求 SHA256 `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`，固定 GPT-4o-mini。
- Writer models：GPT-5.5 subscription proxy，`thinking=off`；MiniMax `MiniMax-M2.7` 通过 OpenRouter 路由。该 endpoint 强制 reasoning，显式 `reasoning.enabled=false` 返回 HTTP 400，因此保留其强制行为并把 reasoning 计入 completion tokens；DeepSeek Flash 使用 OpenRouter 的准确模型 ID `deepseek/deepseek-v4-flash`，preflight 实测 reasoning tokens 为 0。
- 每个 run 使用空目录；adapter SDK 重试为 0，失败 run 不作为 0-event 成功结果。GPT subscription proxy 的内部上游尝试次数由 health manifest 单独记录。

## Context 矩阵

| ID | writer 可见的同一 session 前序信息 | 预算单位 |
|---|---|---|
| `none` | **无前序历史（仅看当前 6 条消息）**。不读取前序原始消息、Event 或摘要；仍保留所有条件共有的 session-start topic 名称快照 | 无 |
| `raw20` | **前序原始消息（最近 20 条）**，再加当前 6 条消息 | turns |
| `raw50` | **前序原始消息（最近 50 条）**，再加当前 6 条消息；LoCoMo 单 session 最长 47，因此在最长 session 中等于全部前序原文 | turns |
| `events20` | **前序已提炼 Event（最近 20 条）**，再加当前 6 条消息 | events |
| `events50` | **前序已提炼 Event（最近 50 条）**，再加当前 6 条消息 | events |
| `summary100` | **前序滚动聊天摘要（最多 100 words）**，再加当前 6 条消息 | words |
| `summary250` | **前序滚动聊天摘要（最多 250 words）**，再加当前 6 条消息 | words |

内部 ID 只用于目录和 manifest。论文表格与分析输出必须使用上面的完整显示名，不能只写 `none/raw/events/summary`。

`raw/events` 的 20/50 与 `summary` 的 100/250 不是 token-matched 条件。主表按实际 API input tokens 报告成本；只有成本接近的配置才做同预算解释。

## Experiment Blocks

### B0：配置与真实接口 smoke

- Scope：LoCoMo sample 1、session 1，共 28 条消息，5 个 writer chunks。
- Runs：7 context × 2 writers，共 14 个 build-only run。
- Purpose：验证两种服务、context 差异、usage 记录、独立目录和失败处理。
- Gate：14 个 run 均生成合法 `_build_stats`；manifest 中除 model/context 外无配置差异。
- Limitation：只用于系统验证与成本估计，不进入主结果表。

DeepSeek 扩展使用相同 scope 和 7 个 context 条件，单独保存为 7-run smoke，完成后与上述 14 个 GPT-5.5/MiniMax 结果按同一分析脚本合并比较。阿里云 token-plan 端点在 2026-07-17 返回 `429 insufficient_quota`，因此 DeepSeek 使用 OpenRouter 的同名准确模型，不混入其他 DeepSeek 版本。

### B1：Build-only 筛选

- Scope：LoCoMo sample 0 和 1 的全部 session；每个配置至少 2 个 conversation。
- Runs：7 context × 2 writers × 2 conversations，共 28 个 build。
- Primary metrics：question-level evidence-any/all recall、unique gold evidence-id recall。
- Cost metrics：build input/output tokens、calls、LLM time、wall time、各 phase usage。
- Other metrics：event 数、topic 文件数、memory bytes、exact normalized duplicate fraction。
- Gate：每个 writer 保留 `none`、默认 `events20`，以及所有 evidence-all 距该 writer 最优不超过 2pp 且不被质量和 input-token 同时支配的配置；每种表示最多保留一个预算。

### B2：固定下游 QA

- Scope：先对 B1 memory 使用冻结的 s0+s1 分层问题集，再扩到 s0+s1 全部 cat1-4。
- Downstream：所有 memory 使用同一个 retriever/answerer；writer 模型不参与回答。
- Judge：仅使用锁定的 `scripts/eval_full.py`。
- Primary metric：locked-judge accuracy；同时报告类别分数与 conversation-cluster bootstrap 95% CI。
- Gate：相对该 writer 的 `events20`，候选 accuracy 下界不低于 -1.5pp；若质量相当，选择 build input tokens 更少者。

### B3：模型交互与失败归因

- Model：`correct ~ writer_model * context_policy + category`，conversation 聚类；build evidence recall 做对应 bootstrap。
- Failure taxonomy：write omission、wrong fact、reference loss、topic placement、retrieval miss、answer error。
- Claim boundary：只有方向在两个 conversation 和 QA 中一致时才报告模型依赖；否则选择两个 writer 共同的默认策略。

### B4：全量确认

- Scope：LoCoMo 10 conversations、cat1-4 共 1540 题。
- Runs：每个 writer 只运行 `events20`、B1/B2 选出的最佳配置和必要的 `none`；不把 7 个配置全部扩到全量。
- Reuse：复用 build memory，不重复建库；judge 仍为锁定 GPT-4o-mini。

## Run Order

1. M0：本地 dry-run、hash 校验、14-run smoke。
2. M1：读取 smoke 成本；接口稳定后执行 s0+s1 build-only。
3. M2：分析 B1，按预注册 gate 选择候选。
4. M3：固定下游 QA，不重新建库。
5. M4：仅对预选配置做 10-conversation 确认。

## Paper Artifacts

- Table：writer × context 的 evidence recall、QA、input/output tokens、calls。
- Figure：accuracy/evidence recall 对 build input tokens 的 Pareto 图。
- Figure：GPT-5.5 与 MiniMax 的 context-policy interaction plot。
- Appendix：配置 manifest、source hashes、phase usage、失败分类和逐 conversation 结果。

## 最终统一比较口径

最终表不能只报告 evidence recall 或 QA accuracy，必须同时包含以下四组结果：

1. **质量与性能**
   - evidence-any、evidence-all、unique evidence-id recall；
   - 固定 retriever/answerer 后的 LoCoMo overall accuracy、cat1-4 accuracy；
   - conversation-cluster bootstrap 95% CI、逐 conversation 方差和失败类型；
   - event 数、重复率、topic 数与实际 memory bytes，避免用事件数量代替质量。
2. **实际成本**
   - build input、output、reasoning tokens 和调用数；
   - writer、verify、merge、sections 分阶段成本及各阶段占比；
   - 每条消息、每个 writer chunk、每个成功覆盖 evidence-id 的 token/call；
   - MiniMax 按冻结的 provider 价格快照估算 variable USD；GPT subscription 不伪造 per-token USD，报告 token、call、订阅口径和配额占用。
3. **时间与吞吐效率**
   - build wall time、LLM time、每条消息耗时、messages/minute；
   - writer/verify 在线路径与 session/final maintenance 路径分别报告；
   - QA 阶段 retrieval/answer calls、tokens、latency 和 questions/minute；
   - 同时给出 quality-per-1k-token、quality-per-call 和 Pareto frontier，不使用单一复合分数替代原始指标。
4. **长期部署与扩展效率**
   - 在不同 session 长度下测量 context token 增长、memory bytes 增长和 maintenance 增长；
   - 基于实测分层回归估计 100、500、1000 条消息的 build tokens、calls、wall time 和 storage，全部标为 projection；
   - 报告 build 成本在 10、100、1000 次后续查询下的 amortized cost；
   - 检查 20/50 budget 是否实际达到上限，以及更长 context 是否只增加成本而没有稳定质量增益。

所有效率指标同时给绝对值和相对 `none`、`events20` 的变化；最终推荐只从质量-成本-时间三者的非支配配置中选择。

## 当前检查表

- [x] context 单位分别定义
- [x] writer interval、verify、tidy、calendar 固定
- [x] 两个 writer 与固定下游模型分离
- [x] evaluator 路径与 hash 固定
- [x] 最终质量、成本、时间、吞吐和长期扩展指标固定
- [ ] B0 dry-run 与真实 smoke 完成
- [ ] B1 build-only 完成
- [ ] B2 QA 完成
- [ ] B3 交互与失败分析完成
- [ ] B4 全量确认完成
