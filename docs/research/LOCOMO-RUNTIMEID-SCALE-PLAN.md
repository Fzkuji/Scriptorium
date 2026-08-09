# LoCoMo Runtime-ID 扩大实验计划

更新日期：2026-08-08

## 1. 目标与当前状态

本计划验证 Runtime-owned IDs、内置文件工具与事务边界修复后，Scriptorium 能否在
LoCoMo 完整 conversation 上稳定、可恢复地扩大构建规模。当前运行中的首个完整
smoke 是 `conv-30`：19 sessions、369 messages、15k writer token cap、120 turns，
仅构建 memory，不查询、不评分。

当前 smoke 输出：

`/home/qi2/scriptorium-runs/locomo-conv30-token15k-t120-wsl-ext4-runtimeids-r1`

截至计划创建时它仍为 `building`。进程存活，memory、verification 和 writer stage
持续有文件活动，`launch-002.stderr` 为空，未发现 failure snapshot；因此尚不能
宣告 smoke 完成，但也没有需要人工干预的错误。

## 2. 不可混用的实验 cohort

所有结果必须记录 writer protocol，不按“文件还能读取”来判断是否属于同一版本。

| Cohort | 内容 | 用途 |
|---|---|---|
| `pre-runtimeid-transaction-fix` | 2026-08-08 修复前完成的 LongMemEval 构建 | 保留既有结果、成本和并发证据；不得与修复后样本直接合并成同一均值 |
| `runtimeid-r1` | Runtime-ID 合并及 shell 事务边界修复后的新构建 | 新的扩大实验主 cohort |
| `hybrid-old-build-new-query` | 旧 memory 使用新 retrieval/runtime 查询 | 只做兼容性分析，必须单独标记，不进入任一纯版本主结果 |

昨天的 LongMemEval 结果不需要删除或全部重跑。它们仍可用于已完成实验的结论；若要
量化本次逻辑修改的影响，应在相同输入上做匹配复建，而不是把新样本追加到旧均值。

## 3. Gate A：完成当前 `conv-30` smoke

只有同时满足下列条件才进入扩大实验：

- `status.json` 与 `build.json` 均显示完整构建完成；
- Source、Topic、Timeline、Core、Recent、Relations 均存在且非空；
- Runtime-ID 唯一，Topic/Core/Timeline 的内部链接和 source links 均可解析；
- Recent 无重复或悬空 ID，Relations 无悬空 ID；
- verification 没有 unsupported/failed；
- 没有 writer failure snapshot；
- 没有重复 TopicFormatError、PermissionError、429、rate limit 或 timeout；
- `final_management=complete`，或 build 记录能证明本配置的 final management 已完成。

若进程退出但没有 `build.json`，本次 smoke 判为失败，不在原目录盲目重启。先保留
现场、定位最后一次活动和 stderr，再决定是否用补齐 checkpoint 后的新目录重跑。

## 4. Gate B：补齐 LoCoMo checkpoint 与恢复能力

当前 conversation runner 调用 `backend.build_memory()` 时没有传 `checkpoint_path`，
因此虽然底层已经支持 token-balanced batch、live audit、failure snapshot、批后
安全停靠和 resume，LoCoMo 运行没有使用这些能力。扩大到多 sample 前先完成：

1. 将每个 LoCoMo 输出目录的 `build-checkpoint.json` 传给构建层。
2. 新构建使用 `resume=false`；只有 checkpoint schema、batch-plan hash、sample ID、
   model 和 writer protocol 全部相符时才允许 `resume=true`。
3. 目录中存在部分 memory、没有完整 build、也没有有效 checkpoint 时继续拒绝运行。
4. 保留 `.stop-after-current-batch`：只在一个完整 batch 提交 checkpoint 后停靠。
5. 保存 writer live audit、progress、failure snapshot 和每批 input tokens/agent turns。
6. 增加无 API 回归测试：中断第二批、恢复后跳过第一批、最终 memory 与无中断运行
   一致；batch plan 或 writer protocol 改变时必须拒绝恢复。
7. 用一个新目录做两批 micro-smoke：第一批完成后安全停靠，再恢复到 complete。

当前 `conv-30` 不应中途迁移为 checkpoint run；让它按原协议自然完成，用于 Gate A。

## 5. Phase 1：三路代表性 LoCoMo pilot

Gate A、B 均通过后，先同时构建三个不同规模的 conversation：

| Lane | Sample | Sessions | Messages | 选择理由 |
|---|---:|---:|---:|---|
| 1 | `conv-26` | 19 | 419 | sessions 少、消息较多的小型样本 |
| 2 | `conv-49` | 25 | 509 | 中等规模 |
| 3 | `conv-50` | 30 | 568 | 较大的长程样本 |

统一配置：`deepseek-v4-flash`、15k token cap、120 turns、verify every 4 sessions、
final management 开启、WSL ext4 独立目录、build-only。每个 sample 单独保存配置、
checkpoint、日志和 memory，禁止多个 sample 共用 memory 目录。

首个 batch 提交后比较每路秒/agent-turn、batch 墙钟、tool error 和 provider 错误：

- 三路均无 429/timeout/PermissionError，且无重复 TopicFormatError：继续运行；
- 多数 lane 高于 24 秒/agent-turn，或出现 provider 排队：只让最慢的一路在当前
  batch 后安全停靠，观察两路是否恢复；
- 单路出现可重复工具错误：让该路在当前 batch 后停靠，其他路继续，不直接杀进程；
- checkpoint 30 分钟无推进且 progress 也无活动：报告现场，不自动重启。

Phase 1 通过标准：三个 sample 全部 complete，组件与引用完整，unsupported 为 0，
failure snapshot 为 0；并发吞吐优于顺序估算，且没有固定 lane 持续滞后。

## 6. Phase 2：完成 LoCoMo 10-sample 构建

已有 `conv-30` 加上 Phase 1 三个样本后，剩余六个为：

`conv-41`、`conv-42`、`conv-43`、`conv-44`、`conv-47`、`conv-48`。

若三路 pilot 稳定，使用最多 5 workers 的固定队列：

| Worker | 顺序 |
|---|---|
| 1 | `conv-41` → `conv-48` |
| 2 | `conv-42` |
| 3 | `conv-43` |
| 4 | `conv-44` |
| 5 | `conv-47` |

先让五个 worker 各提交一个 batch，再决定是否保持五路。沿用 Phase 1 的
24 秒/agent-turn 和错误门槛；并发退化时只停靠最慢的一路，不能一次停多路。

每 15 分钟监控一次即可。报告总完成 sample 数、每个 worker 当前 sample/batch、
相较上次新增 batch/session/source turns、最快/最慢、最新活动和新异常；状态完全
不变时不通知。全部完成后输出十个 sample 的总墙钟、顺序耗时估算、并行效率、
agent turns、provider 用量和错误统计。

## 7. Phase 3：LongMemEval 匹配复建

旧 LongMemEval 结果继续保留。为判断事务修复是否改变记忆结构或成本，使用新代码
复建固定五个已有样本：index `75`、`85`、`92`、`99`、`104`。这些样本覆盖最早
smoke、20-sample 批次的首尾和中间位置。

比较项目只包括同输入可配对指标：

- batch 数、agent turns、墙钟、provider token/请求量；
- tool error、failure snapshot、verification unsupported；
- Topic canonical units、Core/Timeline/Recent/Relations 规模；
- source link 与 Runtime-ID 完整性。

在没有重新执行相同 query/evaluator 前，不比较最终准确率，也不把五个新结果加入
旧 20-sample 的平均分。

## 8. Phase 4：查询与评分门槛

构建稳定不等于 LoCoMo 评分已获准。当前：

- 仓库锁要求 `scripts/evaluation/eval_full.py` SHA-256 为
  `17ef2179cd2781880649eed4a7d62988c069123b5044f007c194cdab8e63f88b`；
- 当前文件实际 SHA-256 为
  `c68a0e1f18a65e60c6708ed748a6064104ed1d12f4606bca060fa8654573dcbd`；
- 因此所有 LoCoMo 评分必须保持阻断，不得自动修改期望 hash、替换 evaluator、
  使用其他 judge 或绕过校验。

可以先完成 build-only 扩大实验。后续若需要查询但暂不评分，应单独冻结代码版本、
记录 memory tree hash，并使用 `--no-evaluate`；查询产物仍不能报告为 LoCoMo 分数。
只有用户在当前对话明确撤销或替换 evaluator lock，或恢复到被锁定的准确文件且
hash 验证通过后，才制定评分运行计划。

## 9. 版本与产物记录

每次运行至少保存：

- Git HEAD、origin/main、dirty diff hash；
- model、base URL 域名、writer protocol hash、tokenizer identity；
- 完整 JSON 配置和 dataset SHA-256；
- checkpoint、live audit/progress、failure snapshots、stderr；
- build/memory tree hash、开始/结束时间和 provider usage；
- cohort 标签：`runtimeid-r1`，不得省略。

本地修改可以不上传，但扩大实验开始前必须冻结一份可复现快照；实验运行期间不
pull、不切换代码、不修改 prompts/tools/runtime。代码更新统一等当前 worker 完成
或安全停靠后再审计合并。

## 10. 推荐执行顺序

1. 等待并验收当前 `conv-30`。
2. 实现并测试 LoCoMo checkpoint/resume。
3. 完成一次“停靠后恢复”的两批 micro-smoke。
4. 启动 `conv-26/49/50` 三路 pilot。
5. 通过门槛后启动剩余六个 sample 的 5-worker 队列。
6. 汇总十个 build-only 结果。
7. 复建 LongMemEval 五个匹配样本。
8. evaluator lock 未解决前停止在构建/兼容性分析阶段，不进入 LoCoMo 评分。
