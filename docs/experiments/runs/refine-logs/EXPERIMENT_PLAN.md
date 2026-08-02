# GPT-5.6 三模型、三 benchmark 单次读取长度实验计划

状态：subscription smoke 已通过；误启动的 Luna 筛选阶段已停止
日期：2026-07-18

## 执行状态

2026-07-18 曾启动两个 Luna + BEAM-100K 单配置 pilot：`gpt56-beam-100k-100K-conv-1-luna-wsession` 与 `gpt56-beam-100k-100K-conv-1-luna-w32`。两者均在完成前被停止并记录为 `interrupted`，合计尝试 2 个配置、完成 0 个配置，因此不构成实验结果，也不参与参数选择、曲线或论文表格。现有代理日志记录 45 次成功请求、314,229 input tokens 和 49,580 output tokens；这些数值只用于记录已发生的调用消耗。

随后确认 frontier 直接提供 `gpt-5.6-luna`、`gpt-5.6-terra` 和 `gpt-5.6-sol`。最小文本与 tool-call 预检均返回精确模型 ID、完整 usage 和 0 reasoning tokens，但这一预检不能代表长构建中的 reasoning 行为。独立执行的 `gpt56-locomo-conv-48-luna-w32` 完成了 33 次 Writer calls 与后续维护后，被 runner 判定为失败：provider 累计报告 125,687 reasoning tokens，与冻结的 `reasoning=none` 冲突。该运行完成 0 个配置，没有生成 `build.json` 或 `memory/_SUCCESS.json`，只保留失败状态、trace 与 `memory.failed.*` 供审计。frontier 因而不能用于本协议的正式结果。

frontier smoke 失败后，frontier tmux 编排器与正式队列均未启动。正式筛选阶段先等待一个能够在完整构建中显式证明 reasoning tokens 为 0 的 provider smoke，避免将短预检误当作正式合同验证。

随后通过冻结在 `reasoning=none` 的 ChatGPT subscription proxy 执行 `gpt56-beam-100k-100K-conv-1-luna-w32`。该 smoke 同时生成 `build.json`、`call_log.json` 和 `memory/_SUCCESS.json`，133 次 provider 请求全部成功，实际模型均为 `gpt-5.6-luna`，实际 reasoning effort 均为 `none`，累计 reasoning tokens 为 0。该配置处理 188 条消息与 130,800 source tokens，生成 926 个 entries；usage 为 516,452 input、129,792 cached input 和 63,060 output tokens，wall time 为 1,344.166 秒。source hashes、response IDs、完成标记与 credential 隔离均已核对通过。

subscription smoke 通过后曾因错误理解用户授权而启动 Luna 的 54 配置筛选阶段。该队列在第 1 个配置 `gpt56-beam-100k-100K-conv-1-luna-wsession` 完成前被停止；本次误启动新增 20 次成功请求、111,036 input tokens、15,000 output tokens和 0 reasoning tokens，完成 0 个新配置。中间 memory 已保留为 `memory.interrupted.*`，不构成结果。tmux、runner 与 8204 proxy 均已停止；Terra 与 Sol 从未启动。当前唯一有效的新结果仍是已完成的 subscription `luna + BEAM-100K conv-1 + W=32` smoke。

## 研究问题

在 NativeMem v10 中，Writer 每次读取的新消息数 `W` 如何影响最终问答性能、记忆构建成本和可支持的对话长度？这种关系是否随 GPT-5.6 的能力档位变化？

一条消息指一条 user 或 assistant 消息，不是一组 user-assistant 往返。

主曲线固定为：

```text
W = {4, 6, 8, 12, 16, 20, 24, 32, session}
```

`session` 表示一个自然 session 一次处理。每个自然 session 单独分块，不把不同日期的 session 展平后再切分。

## 模型

| 档位 | 模型 ID | reasoning | API 参考输入价 | API 参考输出价 |
|---|---|---|---:|---:|
| 高 | `gpt-5.6-sol` | `none` | $5.00/M | $30.00/M |
| 中 | `gpt-5.6-terra` | `none` | $2.50/M | $15.00/M |
| 低 | `gpt-5.6-luna` | `none` | $1.00/M | $6.00/M |

价格只用于 API 参考。若实际通过订阅额度执行，结果中同时报告 token、调用次数和额度消耗，不把 API 价格当成实际账单。

## 六个实验单元

| Benchmark | 单元 | 自然 sessions | 消息数 | 最长 session | QA |
|---|---|---:|---:|---:|---:|
| LoCoMo | `conv-44`，索引 5 | 28 | 675 | 47 | 123 道 cat1-4 |
| LoCoMo | `conv-48`，索引 7 | 30 | 681 | 44 | 191 道 cat1-4 |
| LongMemEval-S | `2318644b`，索引 97 | 41 | 523 | 82 | 1 |
| LongMemEval-S | `gpt4_6dc9b45b`，索引 248 | 48 | 565 | 132 | 1 |
| BEAM 100K | conversation `1`，索引 0 | 3 | 188 | 72 | 20 |
| BEAM 100K | conversation `2`，索引 1 | 3 | 200 | 72 | 20 |

这六个单元用于读取长度曲线筛选。LoCoMo 是 2/10 conversations，BEAM 是 2/20 conversations，LongMemEval-S 只有两道题，均不能作为完整 benchmark 分数。

BEAM 首轮只用 100K。1M 单个 conversation 约有 1500–1800 条消息，在 8 个 `W`、3 个模型的组合下成本明显高于本轮筛选需要。

## 控制变量

主曲线只改变 `NATIVEMEM_V10_WRITE_TURNS`。固定条件包括：

- `NATIVEMEM_PROMPT=v10`；
- `NATIVEMEM_V10_CONTEXT_MODE=none`，Writer 只看当前块；
- v10 的 session 整理和最终整理次数固定为当前默认值；
- Verify 开启；
- 三个模型均使用 `reasoning_effort=none`；
- 同一 benchmark 内的数据转换、检索、答题和 evaluator 不变；
- 每个配置使用独立空输出目录；
- 不在本实验中同时改变 Event 历史、raw 历史或 rolling summary。

主曲线完成后，才能在选定的 `W` 上单独研究分层历史输入。这样能够区分“单次读取长度”与“前序上下文形式”各自的影响。

## 评测协议

LoCoMo 只允许使用 `scripts/eval_full.py`，SHA-256 必须为：

```text
f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd
```

其现有 prompt、GPT-4o-mini judge、cat1-4 范围和输出 schema 均不得修改。LongMemEval-S 使用其仓库中的 `evaluate_qa.py`，BEAM 使用现有 `evaluate_v88_gpt55_beam.py --profile primary`。三个评测通道的 judge 均固定为 GPT-4o-mini，但各 benchmark 保留自己的任务定义和评分 prompt。

## 记录指标

性能：

- benchmark QA 分数；
- First-pass evidence recall；
- Post-verify evidence recall；
- Verify 前后的变化；
- 按问题类型和失败类型分组的结果。

成本：

- Writer、Verify、Tidy 分别记录 calls、input tokens、cached input tokens、output tokens、reasoning tokens 和时间；
- 总构建成本；
- 每 100 条消息的构建成本；
- 每 1000 个原始对话 token 的构建成本；
- 若 provider 返回实际费用，保存实际费用；否则只按公开 API 价格计算参考值。

效率：

- QA 分数/美元；
- QA 分数/100 次 Writer 调用；
- 在相对 `W=6` 性能下降不超过 1.5 个百分点时，可支持的最大 `W`；
- 在该约束下相对 `W=6` 的调用数、token 和时间下降比例。

First-pass 和 Post-verify 必须从同一次执行的中间状态记录，不增加额外模型请求。否则 Verify 可能掩盖较大 `W` 的 Writer 遗漏。

## 运行规模

```text
3 models × 9 W × 6 units = 162 memory builds
```

这 162 个配置是尚未启动的筛选矩阵，不一次性扩展到三个 benchmark 全量。`W=6` 是当前系统默认参照。筛选后每个模型保留 `W=6`、一个满足质量约束的最大 `W`、一个失败边界；只有在两个筛选单元上方向一致的参数才进入完整 benchmark。

## 结果判定

- 不根据 evidence ID 是否存在单独选择参数；最终 QA 是性能主指标。
- 在每个模型、每个 benchmark 内以当前默认 `W=6` 为参照，不跨模型直接比较绝对分数来选择 `W`。
- 候选 `W` 的平均性能下降不得超过 1.5 个百分点，任何主要问题类别下降不得超过 3 个百分点。
- 候选 `W` 的构建调用数或总 token 至少下降 20%。
- 如果三个模型的曲线差异不稳定，不声称模型能力决定最大读取长度。
- LongMemEval-S 两题只能显示执行失败或明显退化，不能估计准确率差异。

## 准备产物

`scripts/prepare_gpt56_chunk_curve.py` 只读取本地数据并生成：

- `experiments/gpt56-chunk-curve/experiment_manifest.json`；
- `experiments/gpt56-chunk-curve/run_matrix.jsonl`。

生成器会检查三个数据文件与 LoCoMo evaluator 的冻结 hash，核对六个单元的 ID 和统计量，并写出 162 行配置。Manifest 同时冻结 generator、runner、NativeMem runtime、adapter、proxy、benchmark converters 和 evaluators 的 source hashes。生成器没有模型执行模式，也不会访问模型 API。

正式输出统一放在 `results/formal/gpt56-chunk-curve/` 下。现有历史结果不移动，以免破坏 manifest、审计脚本和路径引用。
