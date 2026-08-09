# LoCoMo Runtime-ID WSL 交接文档

更新时间：2026-08-08（Asia/Shanghai）

用途：在新的 WSL/Codex 环境中继续 LoCoMo `runtimeid-r1` build-only 实验。本文是当前状态与用户要求的权威交接入口；启动前还应阅读根目录 `AGENTS.md`、`docs/research/LOCOMO-RUNTIMEID-RECOVERY-QUEUE.md` 和 `docs/research/LOCOMO-RUNTIMEID-SCALE-PLAN.md`。

## 1. 一句话状态

LoCoMo 10 个 conversation 中已有 4 个完整构建，2 个保留可靠 checkpoint 等待恢复，4 个尚未运行。下一步计划是六路并发：恢复 `conv-26`、恢复 `conv-43`，同时新建 `conv-47/48/49/50`。实验尚未启动。

两项代码门禁已在当前 dirty tree 中处理：远端 `05ad7e3` 的相关 Runtime 修复已手工并入本地重复标签修复；conversation runner 已暴露 `shell_backend`，所有恢复/新建配置均显式使用 `posix-bash`。Git 引用仍显示本地落后远端 2 个提交，因为没有 pull 或 commit；不得因此再次覆盖式合并。

## 2. 用户要求与不可违背的边界

### 2.1 实验与数据

- 构建模型使用 PackyAPI 的 `deepseek-v4-flash`，base URL 为 `https://www.packyapi.ai`。
- 当前价格按用户提供的官方计费界面记录：输入 `$0.25 / 1M tokens`、输出 `$0.50 / 1M tokens`、缓存读取 `$0.005 / 1M tokens`。
- Writer 按约 15k 输入 token 动态装箱，不再用固定 session 数决定批次；`session_batch=5` 只作后备。
- `max_turns=120`、`verify_writes=true`、`verify_every_sessions=4`、`local_reorg_every_sessions=8`、`final_manage=true`。
- 当前阶段严格 `build_only=true`、`evaluate=false`，不启动 LoCoMo query、judge 或评分。
- 用户此前已授权将已运行的 `conv-26/41/42/43/44` 发送到 PackyAPI。`conv-47/48/49/50` 尚未真正启动；切换环境后启动前应再次向用户复述六路样本和第三方 API 目标并取得最终启动确认。
- 不得读取、显示、复制、记录或提交密钥。密钥只从仓库根目录 `provider-api-key.txt` 读取；文件已被 gitignore。

### 2.2 WSL 与文件系统

- Python、Claude Agent SDK/CLI、shell 和 runner 均从 WSL Ubuntu 24.04 执行。
- 当前共享代码工作树是 Windows `E:\Scriptorium`，WSL 路径为 `/mnt/e/Scriptorium`。
- 结果、Memory、stage、checkpoint、audit 和 failure snapshot 必须写到 WSL ext4：`/home/qi2/scriptorium-runs/...`。
- 代码位于 `/mnt/e` 可以继续使用；关键是 Agent 的 memory workspace 与 stage 位于 `/home/qi2/...`，这样文件工具实际操作 Linux ext4，而不是 DrvFs。
- 不要误用 Windows Claude CLI、Windows `rg` 或 PowerShell/cmd 语法。WSL 中 `command -v rg` 应指向 `/usr/bin/rg` 等 Linux 路径。
- 若要把代码仓库也迁到 ext4，不能只从远端重新 clone 后直接开跑：当前关键实现仍在未提交工作树中。必须先完整迁移和核对 dirty diff，或继续使用 `/mnt/e/Scriptorium`。

### 2.3 checkpoint、停止与恢复

- 每个完整 token batch 提交后必须原子写入 `build-checkpoint.json`。
- 中断后只从最后一个已提交 batch 恢复；不得删除 checkpoint、覆盖旧目录或默认整项重跑。
- conversation runner 当前根据“输出目录已有 checkpoint”自动选择恢复，没有 `--resume` CLI 参数。使用原配置和原输出目录重新执行即可。
- `conv-26` 已完成全部 batch，只剩 final management；恢复后不会增加 sessions/source turns，监控不能据此误报停滞。
- `conv-43` 的第 3 批未提交，恢复会重做第 3 批；前两批不得重做。
- 降并发时只能在用户授权后，对指定 lane 创建 `.stop-after-current-batch`；不得宽泛 kill、`pkill python` 或一次停多个 lane。
- 任何停止、重启、恢复、创建 sentinel 或修改运行中实验，都需要用户明确授权。只读诊断不需要扩大权限。

### 2.4 Git 与用户文件

- 用户要求本地修改暂不上传：不得自动 stage、commit、push 或创建 PR。
- 不得盲目 pull、stash、reset、checkout 或覆盖 dirty worktree。
- 用户自己的 `.obsidian/workspace.json` 与 `.obsidian/graph.json` 不得修改或删除。
- 旧 LongMemEval 结果仍然有效，属于旧 cohort；逻辑更新不要求把已完成结果全部重跑，但新旧 cohort 不得混成同一均值。

### 2.5 汇报偏好

- 用中文汇报，必须给出具体数字，不能只说“仍在运行”。
- 每路同时报告 batch、session、source turns、当前 batch/token、event count、writer rounds、progress、最近活动、进程状态和相较上次增量。
- 明确指出最快、最慢、是否为固定 lane 滞后，以及是否需要人工处理。
- 只报告新出现或状态变化的异常；但启动初期按用户要求每 5 分钟给出五/六路的完整状态。
- 全部完成或全部进程退出后停止监控并汇报，不能留下无意义的自动巡检。

## 3. LoCoMo evaluator 硬锁

根目录 `AGENTS.md` 的用户锁高于所有实验计划：

- 唯一允许的 evaluator 是 `code/scripts/evaluation/eval_full.py`。
- 要求 SHA-256：`17ef2179cd2781880649eed4a7d62988c069123b5044f007c194cdab8e63f88b`。
- 当前实际 SHA-256：`c68a0e1f18a65e60c6708ed748a6064104ed1d12f4606bca060fa8654573dcbd`。
- 因 hash 不符，所有 LoCoMo 评分都必须阻断。不得自动更新期望 hash、替换 evaluator、加 fallback、改 judge/prompt/category 或绕过检查。
- `scripts/score_v88_gpt55_benchmarks.py` 永远不能用于 `locomo` 或 `locomo-cat5`。
- 当前六路只能 build-only。除非用户在当前对话明确撤销或替换锁，否则不要讨论“先临时评分”。

## 4. 当前 Git 与代码状态

截至交接时：

```text
branch: main
local HEAD: 8ad0354f624309d452150902ff75797dd47ceed8
origin/main: c6e2ac1cdd0bc89f6c6950b01d917d415d6f5f27
HEAD...origin/main: 0 ahead / 2 behind
```

远端新增：

1. `05ad7e3 Bind a rewritten citation back to the definition it kept`
2. `c6e2ac1 Drop editor state and Windows host scripts from the repository`

`05ad7e3` 与本地改动高度重叠：它修改 `build.py`、`topic_normalization.py` 和 transaction 测试，并删除大量 Obsidian/Windows 文件。当前本地对 `topic_normalization.py` 还有针对 `conv-43` 的“重复本地脚注标签按出现次序分配”修复。这两个修复解决相邻但不相同的问题，不能二选一覆盖：

- 远端修复：孤立的新 `[^eN]` citation 重新绑定到保留的 definition，并保留 on-disk 原始文本用于写回判断。
- 本地修复：同一 trajectory 多次从 `e1` 重新编号时，不再把多个 definition 映射成同一个稳定脚注 ID。

WSL 接手后的正确顺序：

1. `git fetch origin` 只更新远端引用，不修改工作树。
2. 阅读 `git diff HEAD..origin/main -- code/src/management/topic_normalization.py code/src/build.py code/tests/management/test_transaction.py`。
3. 保存当前 dirty diff 清单；不要动 Obsidian 用户状态。
4. 当前 dirty tree 已手工整合远端 orphan-binding/on-disk 修复与本地 occurrence-scoping 修复；先审计现有 diff，避免重复应用。
5. 当前合并后测试为 106 passed；任何后续调整都必须重跑第 8 节测试。

当前工作树包含大量用户和前序任务修改及未跟踪文件。使用 `git status --short` 重新确认；不要假设所有 diff 都属于本次 LoCoMo 修复。

## 5. 已实现的关键运行能力

- token-15k 动态 batch plan。
- batch checkpoint、resume、writer live audit、progress 和 failure snapshot。
- checkpoint 记录并校验 sample/model/batch plan/writer protocol。
- final management 状态记录。
- provider token 与估算成本记录；`provider_spend` 是共享账户累计差值，可能包含其他并发请求，不能当成单样本美元成本。
- Runtime-owned block/evidence ID、Source/Topic/Timeline/Core/Recent/Relations 派生与引用维护。
- WSL ext4 运行已证明可避免先前 Windows/DrvFs 的 ACL、目录替换和临时目录竞态。

已补齐：

- `code/scripts/runners/conversation/config.py` 已增加 `--shell-backend`。
- `conversation/runner.py` 已把该值传给 `MemoryConfig`。
- 当前 LoCoMo 恢复配置和四个六路新配置都锁定 `shell_backend: posix-bash`。
- 先前 lane1/lane5 的少量 `/bin/sh` 语法错误应不再由该路径复现；正式运行仍需监控新的 shell error。

## 6. 十个样本状态

| Sample | Sessions | Source messages/turns | 状态 | 备注 |
|---|---:|---:|---|---|
| `conv-26` | 19 | 419 | 待恢复 | 2/2 batches 完成，final management 因 `ECONNRESET` 失败 |
| `conv-30` | 19 | 369 | complete | 单路 smoke，约 3331 秒（55分31秒） |
| `conv-41` | 32 | 663 | complete | 3/3 batches，229 agent turns，估算 `$0.32823058` |
| `conv-42` | 29 | 629 | complete | 3/3 batches，242 agent turns，估算 `$0.43763579` |
| `conv-43` | 29 | 680 | 待恢复 | checkpoint 2/3，19/29 sessions，433/680 turns |
| `conv-44` | 28 | 675 | complete | 3/3 batches，254 agent turns，估算 `$0.37192641` |
| `conv-47` | 31 | 689 | 未运行 | 下一组六路新样本 |
| `conv-48` | 30 | 681 | 未运行 | 下一组六路新样本 |
| `conv-49` | 25 | 509 | 未运行 | 下一组六路新样本 |
| `conv-50` | 30 | 568 | 未运行 | 下一组六路新样本 |

完整完成数为 4/10；计入两个中途失败样本后尚有 6 个未完整完成。排除 `conv-43` 时正好是 5 路；修复确认后可组成六路。

## 7. 两个恢复任务

### 7.1 conv-26

```text
config: code/scripts/configs/locomo_5way_lane1_conv26_token15k_t120_runtimeids_r1.json
output: /home/qi2/scriptorium-runs/locomo-5way-lane1-conv26-token15k-t120-runtimeids-r1
checkpoint: 2/2 batches, 19/19 sessions, 419/419 source turns
final_management: pending
```

错误为 PackyAPI 连接被重置：`API Error: Unable to connect to API (ECONNRESET)`。这是临时网络/API 连接失败，不是 WSL、Shell 或文件系统错误。原目录、memory 和 checkpoint 均保留；重新执行原配置时 runner 应自动进入恢复，只重做 final management。

### 7.2 conv-43

```text
config: code/scripts/configs/locomo_5way_lane4_conv43_token15k_t120_runtimeids_r1.json
output: /home/qi2/scriptorium-runs/locomo-5way-lane4-conv43-token15k-t120-runtimeids-r1
checkpoint: 2/3 batches, 19/29 sessions, 433/680 source turns
current batch: 3, about 14241 input tokens
failure snapshot: writer-failure-batch-002.json
```

第一次提交因 `duplicate footnote definition: e-a58f5fbb2e` 回滚；自动修复又因 `e-2992cc900e` 重复而失败。根因已定位为 Runtime 对模型重复使用本地 `e1/e2` 标签的归一化作用域错误。

本地已修改：

- `code/src/management/topic_normalization.py`
- `code/tests/management/test_memory.py`

新增测试覆盖“同一文件两次使用 e1”和“不同文件各自使用 e1”。当前 WSL 验证：

- management 定向：56 passed。
- Markdown + management + current runtime：合并远端修复后 106 passed。

此修改没有改变 prompt/tools，因此当前 `writer_protocol_sha256` 不变；但在合并远端 `05ad7e3` 后必须重新确认并重跑测试。恢复 `conv-43` 会重做未提交的第 3 批。

## 8. WSL 接手与测试门禁

进入 WSL：

```bash
cd /mnt/e/Scriptorium
source .venv/bin/activate
python --version
python -c 'import claude_agent_sdk; print("claude-agent-sdk: ok")'
command -v rg
rg --version | head -1
test -s /mnt/e/Scriptorium/provider-api-key.txt
```

最后一行只能检查文件非空，禁止 `cat` 密钥。

确认无旧进程：

```bash
pgrep -af 'run_conversation.py|scripts.runners.run_conversation' || true
```

确认 Git 身份和 dirty tree：

```bash
cd /mnt/e/Scriptorium
git status --short
git rev-parse HEAD
git rev-parse origin/main
git rev-list --left-right --count HEAD...origin/main
```

完成远端/本地整合及 `posix-bash` 配置后运行：

```bash
cd /mnt/e/Scriptorium
.venv/bin/python -m pytest -q \
  code/tests/markdown \
  code/tests/management \
  code/tests/scripts/test_current_runtime.py
```

当前合并后基线是 106 passed。若数量因后续测试变化，以“全部通过且新回归仍存在”为准，不能只比较数量。

另外建议运行 conversation runner 的完整相关测试（若已拆分到其他文件，先用 `rg` 定位）：

```bash
rg -n 'run_conversation|conversation runner|shell_backend' code/tests
```

## 9. 六路计划

建议 lane 分配：

| Lane | 任务 | 类型 | 输出目录 |
|---|---|---|---|
| 1 | `conv-26` | 原目录恢复 final management | `locomo-5way-lane1-conv26-token15k-t120-runtimeids-r1` |
| 2 | `conv-43` | 原目录从 2/3 checkpoint 恢复 | `locomo-5way-lane4-conv43-token15k-t120-runtimeids-r1` |
| 3 | `conv-47` | 新构建 | 新建 `locomo-6way-lane3-conv47-token15k-t120-runtimeids-r1` |
| 4 | `conv-48` | 新构建 | 新建 `locomo-6way-lane4-conv48-token15k-t120-runtimeids-r1` |
| 5 | `conv-49` | 新构建 | 新建 `locomo-6way-lane5-conv49-token15k-t120-runtimeids-r1` |
| 6 | `conv-50` | 新构建 | 新建 `locomo-6way-lane6-conv50-token15k-t120-runtimeids-r1` |

不要为恢复任务创建新目录；否则会丢失 checkpoint 语义。四个新样本必须使用从未存在的新目录和独立配置。

四个新配置应复制当前五路配置的冻结参数，只修改：

- `sample_id`
- `output_dir`
- 文件名/lane 编号
- 保持已配置的 `shell_backend: posix-bash`

所有 lane 都是独立进程、`workers=1`。不要把 6 个 sample 塞进一个共享 memory/output 目录。

六路可行性判断：上次五路没有出现 429、持续限流、普遍 timeout、PermissionError 或固定 lane 阻塞；整体有效吞吐约为单路 4 倍。`conv-26` 只剩 final management，预计六路峰值并发持续时间较短，随后自然降为五路。六路仍属于新容量点，必须先观察再放任长跑。

## 10. 启动方式

配置完成并取得用户最终启动确认后，在 WSL 中逐路启动。示例：

```bash
cd /mnt/e/Scriptorium/code
RUN=/home/qi2/scriptorium-runs/locomo-6way-lane3-conv47-token15k-t120-runtimeids-r1
mkdir -p "$RUN"
nohup /mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_conversation.py \
  --config scripts/configs/locomo_6way_lane3_conv47_token15k_t120_runtimeids_r1.json \
  > "$RUN/launch-001.stdout" \
  2> "$RUN/launch-001.stderr" &
echo $!
```

恢复任务使用原配置、原输出目录和新的日志编号，例如 `launch-002.*`。runner 没有 `--resume` 参数；checkpoint 存在时自动恢复：

```bash
cd /mnt/e/Scriptorium/code
RUN=/home/qi2/scriptorium-runs/locomo-5way-lane1-conv26-token15k-t120-runtimeids-r1
nohup /mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_conversation.py \
  --config scripts/configs/locomo_5way_lane1_conv26_token15k_t120_runtimeids_r1.json \
  > "$RUN/launch-002.stdout" \
  2> "$RUN/launch-002.stderr" &
echo $!
```

对 `conv-43` 同理。启动后立即记录六个 PID 与配置到当次运行清单。不得把密钥放入命令行；配置只引用 `api_key_file`。

## 11. 监控要求

### 11.1 频率

- 启动后的前 30 分钟：每 5 分钟一次。
- 若六路均稳定、无 429/timeout/重复格式错误，可询问用户是否改为每 15 分钟。
- 任一路完成后继续监控其余 lane；全部 complete 或全部 wrapper 退出后停止并删除自动化。

### 11.2 每路必须报告

- `status.json` 的 phase/status。
- `build-checkpoint.json`：
  - `completed_batches/total_batches`
  - `completed_sessions/total_sessions`
  - `completed_source_turns/total_source_turns`
  - `current_batch_number/current_batch_input_tokens`
  - `event_count`
  - `final_management`
  - `writer_trajectories` 最后一项 rounds
- 最新 `writer-live-*.jsonl.progress.json`：`num_turns`、assistant messages、`is_error`、`windows_retries`、更新时间。
- wrapper/runner 是否存活、最新活动距今多久。
- 相较上次新增多少 batch/session/source turns；没有 checkpoint 增量时用 progress 轮数与活动时间说明是否仍推进。
- 六路最快、最慢、是否固定 lane 滞后。

### 11.3 新异常检查

- `status=error` 工具调用与重复错误类型。
- `TopicFormatError`、尤其 duplicate footnote definition。
- commit/repair rejected、Reached maximum turns。
- 429、rate limit、持续 timeout、`ECONNRESET`。
- PermissionError、非零 `windows_retries`、DrvFs 路径误用。
- writer failure snapshot 与 `launch-*.stderr` traceback。

单次普通 shell 探查错误不等于样本不可用；只有重复、阻断提交或导致 wrapper 退出才升级处理。`conv-43` 若再次出现重复稳定脚注，立即报告并保留 snapshot，不要无限恢复。

### 11.4 六路降载建议

- 若多数 lane 出现 429/排队/持续 timeout，或相对五路基线出现明显普遍退化，先报告，不自动停。
- 用户授权后只停靠最慢或异常的一路，并等待当前 batch checkpoint 提交；不得一次停多路。
- `conv-26` 很可能先完成，正常情况下会自动把并发从 6 降到 5，无需人为干预。

## 12. 完成验收

每个 sample 必须满足：

- `status=complete` 且 `final_management=complete`。
- checkpoint batches/sessions/source turns 全部完成。
- Source、Topic、Timeline、Core、Recent、Relations、verification 文件均存在且非空。
- Runtime-ID 唯一。
- Topic/Core/Timeline source links 可解析并能回查 Source。
- Recent 无重复或悬空 ID；Relations 无悬空 ID。
- verification 的 unsupported/failed 为 0。
- 没有未解释的 failure snapshot、stderr 或重复工具错误。
- 记录 wall time、agent turns、输入/输出/cache tokens、`provider_estimated_cost_usd`。

自动监控只能读取元数据和错误摘要，不应输出记忆正文；内容级引用核验由当前主任务在用户授权范围内执行。

全部十个 build-only 样本完成后汇总：

- 10/10 完成状态与每样本组件完整性。
- 六路总墙钟、顺序耗时估算、并行加速比和并行效率。
- 每样本与总 agent turns/tokens/估算成本。
- 所有工具、格式、网络和恢复异常。
- 明确声明没有运行 LoCoMo 评分。

## 13. 已完成五路的吞吐结论

上一组五路 `conv-26/41/42/43/44` 中：

- `conv-41/42/44` 完整完成。
- `conv-26` 完成所有 batch，在 final management 遇到一次 `ECONNRESET`。
- `conv-43` 在第 3 批遇到确定的脚注归一化错误。
- 没有 429、持续限流、普遍 timeout、PermissionError 或固定 lane 阻塞。
- 单路 `conv-30` 约 55.5 分钟；五路约在 45–70 分钟内完成 3 个样本并把另两个推进到最后阶段，粗略有效吞吐约为单路 4 倍、并行效率约 80%。样本规模不同，此数字用于容量判断，不作为严格性能基准。

因此六路值得试。`conv-43` 修复合并与 `posix-bash` 门禁现已通过；启动前只需复核新输出目录未占用、再次确认无旧 runner，并取得用户最终启动授权。

## 14. 交接后第一轮操作清单

1. 在 WSL 打开 `/mnt/e/Scriptorium`，阅读本文件和 `AGENTS.md`。
2. 检查 Python/SDK/rg、密钥文件存在性和旧 runner。
3. 审计当前 dirty diff；远端 `05ad7e3` 的相关代码已手工整合，禁止直接 pull 或重复套用。
4. 确认 conversation runner 和六路配置仍锁定 `posix-bash`。
5. 必要时重跑 Markdown、management、current-runtime 和 conversation-runner 回归；当前为 106 passed。
6. 四个新配置已经创建；启动前确认 `conv-47/48/49/50` 的 ext4 输出目录不存在，恢复配置仍指向原目录。
7. 向用户汇报 diff、测试和六路 lane 表，请求最终启动确认。
8. 获得确认后启动六路，并创建每 5 分钟的详细只读监控。
9. 全部结束后验收组件/引用、汇总成本与并行效率，并停止监控。

## 15. 相关文件

- `AGENTS.md`
- `docs/research/LOCOMO-RUNTIMEID-RECOVERY-QUEUE.md`
- `docs/research/LOCOMO-RUNTIMEID-SCALE-PLAN.md`
- `docs/research/LONGMEMEVAL-WSL-RUNBOOK.md`
- `code/src/management/topic_normalization.py`
- `code/src/management/agent.py`
- `code/src/build.py`
- `code/scripts/runners/conversation/config.py`
- `code/scripts/runners/conversation/runner.py`
- `code/tests/management/test_memory.py`
- `code/tests/scripts/test_current_runtime.py`

此文档不包含任何密钥。任何新环境若只看到远端仓库而看不到这里列出的 dirty files，都不是当前可直接续跑的实验环境。
