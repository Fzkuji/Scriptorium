# LongMemEval WSL 实验启动与巡检手册

本文档用于在 WSL2 Ubuntu 环境中启动、监控、恢复和验收 Scriptorium 的 LongMemEval-S 构建实验。命令默认仓库位于 Windows `E:\Scriptorium`，对应 WSL 路径 `/mnt/e/Scriptorium`。

## 1. 安全约束

- API key 只保存在仓库根目录 `provider-api-key.txt`，不得复制到配置、命令行明文、日志或 Git。
- 每次实验使用新的配置名和结果目录，绝不覆盖旧结果。
- 正式 smoke 使用 `build_only: true`，不运行 answerer 或 judge。
- Windows 失败结果与 WSL 结果分目录保存，不跨目录恢复。
- LoCoMo 评估受根目录 `AGENTS.md` 锁定；本手册不运行或修改 LoCoMo evaluator。
- 不修改 `.obsidian/workspace.json` 或 `.obsidian/graph.json`。

## 2. 进入环境

在 PowerShell 中进入 WSL：

```powershell
wsl.exe -d Ubuntu-24.04
```

随后在 WSL 中：

```bash
cd /mnt/e/Scriptorium
source .venv/bin/activate
python --version
python -c 'import claude_agent_sdk; print("claude-agent-sdk: ok")'
command -v rg
rg --version | head -1
```

`.venv` 应链接到 Linux 文件系统中的 `/home/qi2/.venvs/scriptorium`。Python SDK 自带 Linux Claude CLI；无需使用 PATH 中 `/mnt/c/.../claude` 的 Windows CLI。

`command -v rg` 必须返回 `/usr/bin/rg` 等 WSL 原生路径，不能返回 `/mnt/c/Program Files/WindowsApps/.../rg`。若缺失：

```bash
sudo apt-get update
sudo apt-get install ripgrep
```

## 3. 启动前门禁

先运行离线定向测试：

```bash
cd /mnt/e/Scriptorium
.venv/bin/python -m pytest -q \
  code/tests/management \
  code/tests/runtime/test_writer_capacity.py \
  code/tests/scripts/test_longmemeval_runner.py \
  code/tests/agent_runtime/test_claude_code.py \
  code/tests/retrieval/test_inspect.py
```

确认没有旧 runner：

```bash
pgrep -af 'run_longmemeval.py|run_writer_micro_smoke.py' || true
```

确认密钥存在，但不要显示内容：

```bash
test -s /mnt/e/Scriptorium/provider-api-key.txt
```

确认新配置满足以下条件：

- `start: 75`
- `limit: 1`
- `model: deepseek-v4-flash`
- `session_batch: 8`（仅作为未设置 token cap 时的后备值）
- `writer_input_token_cap: 15000`
- `max_turns: 120`
- `shell_backend: posix-bash`
- `verify_writes: true`
- `verify_every_sessions: 4`
- `final_manage: true`
- `build_only: true`
- `output_dir` 是从未使用的新目录

如果按 token 动态装箱，另外设置：

- `writer_input_token_cap: 15000`
- `session_batch` 仅作为无 token cap 时的后备值，不再决定实际批大小

token 模式会在完整 message 之间切分；一个很长的 session 可以跨 batch，单条 message 不会被拆开。checkpoint 的 `batch_plan[].input_tokens` 记录每批渲染后输入估算，`session_indices` 记录涉及的原始 session。`writer_tokenizer` 记录计数器身份。对于 `deepseek-v4-flash`，当前使用本地 `tiktoken` 的 `o200k_base` fallback；它是可复现的装箱预算，不是服务端精确计费 token。WSL 正式实验显式使用 `posix-bash`，避免模型生成 process substitution 等 Bash 语法时落到 `/bin/sh`。

当前推荐配置是 token-15k 动态装箱，并将活动结果写到 WSL ext4：

```text
code/scripts/configs/longmemeval_ms_smoke_token15k_t120_wsl_ext4_r1.json
```

其 `output_dir` 是 `/home/qi2/scriptorium-runs/...`。代码可以继续从 `/mnt/e/Scriptorium` 执行，但 Memory、stage、checkpoint、audit 和结果均留在 ext4，避免 DrvFs 的目录锁、原子替换和性能问题。实验完成并冻结后，再把结果复制回仓库的结果归档区。

旧的固定 S8 配置仅作为历史对照保留，不再是默认启动项：

```text
code/scripts/configs/longmemeval_ms_smoke_s8_t120_wsl_r1.json
```

复制配置开展新实验时，至少同时递增配置文件名和 `output_dir` 中的 run 编号。

### 3.1 启动前预览 token batch plan

该命令只读取数据并本地计数，不调用模型、不创建结果目录：

```bash
cd /mnt/e/Scriptorium/code
/mnt/e/Scriptorium/.venv/bin/python \
  scripts/diagnostics/plan_writer_batches.py \
  --data benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --item-index 75 \
  --model deepseek-v4-flash \
  --token-cap 15000
```

当前 index 75 的冻结计划应为 9 batches；前 8 批约 14.6k–15.0k estimated tokens，最后一批约 5.7k。若代码、prompt、tokenizer 或数据变化导致计划不同，应使用新的 run 编号，不得恢复旧 checkpoint。

## 4. 单-session diagnostic

正式 token-15k smoke 前先确认完整 Writer 事务链路：

```bash
cd /mnt/e/Scriptorium/code
/mnt/e/Scriptorium/.venv/bin/python \
  scripts/diagnostics/run_writer_micro_smoke.py \
  --data benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --output-dir results/development/one-session-wsl-micro-rN \
  --api-key-file ../provider-api-key.txt \
  --base-url https://www.packyapi.ai \
  --model deepseek-v4-flash \
  --item-index 75 \
  --max-turns 30 \
  --shell-backend posix-bash
```

将 `rN` 换成尚不存在的新编号。成功标准：

- `micro-result.json` 与 `build-checkpoint.json` 均为 `complete`；
- 1 session、6 source turns；
- 至少生成一个 Topic block；
- Topic 引用的 Source ID 能在 `memory/sources/` 中回查；
- Topic、Timeline、Core、Recent、Relations 分别保存；
- 无重复工具错误循环。

## 5. 启动完整 token-15k 单样本 smoke

推荐先在前台运行，便于直接看到启动错误：

```bash
cd /mnt/e/Scriptorium/code
/mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_longmemeval.py \
  --config scripts/configs/longmemeval_ms_smoke_token15k_t120_wsl_ext4_r1.json \
  --api-key-file ../provider-api-key.txt
```

固定 session 与 token 动态装箱会产生不同的 batch plan，二者不得使用 `--resume` 交叉恢复。

### 在当前批完成后安全停靠

不要在 writer 或 verification 进行中直接终止。需要降低并发时，在目标 item 目录创建 sentinel：

```bash
touch "$ITEM/.stop-after-current-batch"
```

runner 会先完成当前 token batch、verification 和必要的 topic 整理，原子提交
`build-checkpoint.json`，然后将 build 状态写为 `paused` 并以退出码 75 退出。此时
`completed_batches` 已包含刚完成的批次，可以安全恢复。

恢复前先移除 sentinel，再使用原配置、原输出目录和 `--resume`：

```bash
rm "$ITEM/.stop-after-current-batch"
cd /mnt/e/Scriptorium/code
/mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_longmemeval.py \
  --config scripts/configs/longmemeval_ms_smoke_token15k_t120_wsl_ext4_r1.json \
  --api-key-file ../provider-api-key.txt \
  --resume
```

每个 item 还有操作系统级单写者锁。两个 runner 同时写同一 item 时，后启动者会立即拒绝；不同输出目录可以并发。锁在进程退出时自动释放，磁盘上的 `.lock` 文件无需删除。

如需在 WSL 终端关闭后继续运行，可使用 `tmux`。若本机没有 `tmux`，也可使用 `nohup`，但必须使用新的日志文件名：

```bash
cd /mnt/e/Scriptorium/code
nohup /mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_longmemeval.py \
  --config scripts/configs/longmemeval_ms_smoke_token15k_t120_wsl_ext4_r1.json \
  --api-key-file ../provider-api-key.txt \
  > /home/qi2/scriptorium-runs/longmemeval-ms-item75-token15k-t120-wsl-ext4-r1.stdout.log \
  2> /home/qi2/scriptorium-runs/longmemeval-ms-item75-token15k-t120-wsl-ext4-r1.stderr.log &
echo $!
```

记录 PID，但不要把 API key 放入命令行。

## 6. 每 10 分钟巡检

设结果根目录：

```bash
RUN=/home/qi2/scriptorium-runs/longmemeval-ms-item75-token15k-t120-wsl-ext4-r1
ITEM="$RUN/items/0075_3a704032"
```

检查进程：

```bash
pgrep -af run_longmemeval.py || true
```

检查构建 checkpoint：

```bash
python -m json.tool "$ITEM/build-checkpoint.json"
```

重点字段：

- `status`
- `completed_batches` / `total_batches`
- `completed_sessions` / `total_sessions`
- `completed_source_turns` / `total_source_turns`
- `current_batch_number`
- `current_batch_input_tokens`
- `event_count`
- `writer_trajectories`
- `failure_audit`

查看当前实时 audit：

```bash
AUDIT=$(find "$ITEM" -maxdepth 1 -name 'writer-live-batch-*.jsonl' -printf '%T@ %p\n' \
  | sort -n | tail -1 | cut -d' ' -f2-)
test -n "$AUDIT" && tail -n 10 "$AUDIT"
test -n "$AUDIT" && cat "$AUDIT.progress.json"
```

查看失败快照与 stderr：

```bash
find "$ITEM" -maxdepth 1 -name 'writer-failure-*.json' -print
tail -n 50 "$RUN"*.stderr.log 2>/dev/null || true
```

正常信号：

- audit 或 progress 的更新时间持续前进；
- 工具调用大多为 `status: ok`；
- batch 完成后 `completed_batches`、sessions 和 source turns 一起增长；
- `windows_retries` 在 WSL 中应为 0；
- stderr 无 traceback。

需要立即检查或停止的信号：

- runner 进程消失但 checkpoint 不是 `complete`；
- checkpoint 为 `failed`；
- 同一种工具错误连续重复；
- audit/progress 长时间不更新；
- stderr 出现 traceback、认证错误或持续网络错误；
- batch 已产生有效 Writer 结果但 checkpoint 长时间未提交。

首次探查不存在的 `core.md` 返回一次非零状态通常可恢复；只有重复发生或阻止后续写入时才视为异常。

## 7. 停止实验

先确认 PID 对应目标 runner：

```bash
pgrep -af run_longmemeval.py
```

然后发送正常终止信号：

```bash
kill PID
```

不要使用宽泛的 `pkill python`。停止后保留整个结果目录、checkpoint、live audit、failure snapshot 和控制台日志，不要删除或覆盖。

## 8. 从已提交 batch 恢复

只有在以下条件均满足时才恢复：

- 配置、数据、模型、prompt 和代码身份未改变；
- `build-checkpoint.json` 中已有可靠的 `completed_batches`；
- memory transaction 与 checkpoint 一致；
- 不是明确标记为不可恢复的旧 Windows run。

使用相同配置、相同结果目录并增加 `--resume`：

```bash
cd /mnt/e/Scriptorium/code
/mnt/e/Scriptorium/.venv/bin/python \
  scripts/runners/run_longmemeval.py \
  --config scripts/configs/longmemeval_ms_smoke_token15k_t120_wsl_ext4_r1.json \
  --api-key-file ../provider-api-key.txt \
  --resume
```

恢复前务必备份或保留现有日志。runner 只应从最后一个已提交 batch 的下一个 batch 继续。

## 9. 完成后验收

完整 token-15k 单样本 smoke 应满足：

- checkpoint `status: complete`；
- `completed_batches: 9`；
- `completed_sessions: 48`；
- `completed_source_turns: 479`；
- 每个 batch 都有 Writer trajectory 和 live audit；
- Source、Topic、Timeline、Core、Recent、Relations 均存在；
- Topic 中所有 Source ID 可在 `memory/sources/` 回查；
- 没有 failure snapshot 或未解释的 stderr；
- `build_only` 运行没有 judge 结果或 judge 成本。

完成 token-15k smoke 后再决定是否运行其他 token cap 或固定-session 对照；不同 batch plan 不得复用彼此的 memory/checkpoint 作为构建结果。

## 10. 当前运行

截至 2026-08-07，当前运行是：

```text
配置：code/scripts/configs/longmemeval_ms_smoke_token15k_t120_wsl_ext4_r1.json
结果：/home/qi2/scriptorium-runs/longmemeval-ms-item75-token15k-t120-wsl-ext4-r1
样本：dataset index 75 / question 3a704032
规模：48 sessions / 479 source turns / 9 token-balanced batches
预算：writer_input_token_cap=15000 / tiktoken o200k_base fallback estimate
模式：build-only / no judge
```

## 11. 并发实验经验（2026-08-07）

### 11.1 推荐并发度

已在 WSL ext4、`deepseek-v4-flash`、token cap 15k 下完成实测：

| 并发度 | 样本数 | 墙钟时间 | 单样本平均 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 4 | 4 | 约 114.3 分钟 | 约 104.4 分钟 | 稳定，推荐默认值 |
| 5 | 5 | 约 134.9 分钟 | 约 124.2 分钟 | 总吞吐约比四路高 6%，但单样本约慢 19% |

五路首批的实测速度分别为 13.20、12.05、11.04、10.41、11.28
秒/工具轮，中位数 11.28，未触发 24 秒/轮的降并发阈值。四路与五路均未发现
429、持续 timeout、PermissionError、非零 `windows_retries` 或固定 lane 阻塞。

建议：

- 默认使用 4 路并发，延迟和稳定性更均衡。
- 追求最大总吞吐时可使用 5 路，但收益已接近平台边际。
- 不要一次启动 20 个 runner。处理 20 个 sample 时使用 5 个 worker，每个 worker
  顺序处理 4 个 sample；每个 sample 仍保留独立 item checkpoint。

### 11.2 20-sample / 5-worker 分片

index 85--104 的分片方式：

| Worker | start | limit | 覆盖 index |
| --- | ---: | ---: | --- |
| 1 | 85 | 4 | 85--88 |
| 2 | 89 | 4 | 89--92 |
| 3 | 93 | 4 | 93--96 |
| 4 | 97 | 4 | 97--100 |
| 5 | 101 | 4 | 101--104 |

对应配置：

```text
code/scripts/configs/longmemeval_ms_20_token15k_t120_wsl_ext4_worker1.json
code/scripts/configs/longmemeval_ms_20_token15k_t120_wsl_ext4_worker2.json
code/scripts/configs/longmemeval_ms_20_token15k_t120_wsl_ext4_worker3.json
code/scripts/configs/longmemeval_ms_20_token15k_t120_wsl_ext4_worker4.json
code/scripts/configs/longmemeval_ms_20_token15k_t120_wsl_ext4_worker5.json
```

后台启动前必须先创建输出目录，否则 shell 会在 runner 创建目录之前尝试打开
`launch.stdout` / `launch.stderr`，导致重定向失败且 runner 根本没有启动。

从 PowerShell 使用 `Start-Process` 时，必须把完整的 Bash 命令显式包成
`bash -lc` 的一个参数。启动后不能只检查 Windows PID；必须再确认：

1. worker 对应的 `wsl.exe` 父子进程存在；
2. `items/<item-id>/build-checkpoint.json` 已创建；
3. 首个 `writer-live-batch-000-*.progress.json` 开始更新；
4. `launch.stderr` 为空或没有 traceback。

### 11.3 Windows 侧巡检：不要反复启动 wsl.exe

高并发运行时，不要每个 lane 单独执行一次 `wsl.exe` 查询，也不要在查询超时后
立即重试。超时的客户端可能继续滞留；多轮巡检会堆积 `wsl.exe`，最终堵塞新的
Windows-to-WSL 命令入口，即使 Linux 内部的实验仍在正常运行。

优先从 Windows 直接只读访问 WSL ext4：

```powershell
$root = '\\wsl.localhost\Ubuntu-24.04\home\qi2\scriptorium-runs'
Get-ChildItem -LiteralPath $root
```

巡检建议：

- 使用一个 PowerShell 查询一次读取全部 worker checkpoint。
- 4/5 路 smoke 可每 8 分钟检查；20-sample 长任务可每 15 分钟检查。
- 只在 checkpoint 推进、sample 完成或出现新异常时汇报。
- 汇报每个 worker 的已完成 sample、当前 item/batch、sessions、source turns，
  以及相较上次的增量；不要重复汇报无变化的 wrapper 状态。
- 读取失败时不要循环重试。

若发现大量滞留 `wsl.exe`，先通过 `Win32_Process.CommandLine` 区分用途。
只终止已确认无用的旧测试、旧诊断或旧巡检客户端；不得使用宽泛的
`Stop-Process -Name wsl`，不得停止 `vmmemWSL`，也不得误杀
`run_longmemeval.py` worker。清理前应保留新 worker 的父进程及其同命令子进程。

### 11.4 长任务验收

每个 sample 完成后都应验证 `status=complete`、所有 batch/session/source turns
完成、`final_management=complete`，并检查 Source、Topic、Timeline、Core、
Recent、Relations 均存在且非空。`verification.jsonl` 的记录数应与 source
session 数对应，且 `supported=false` 数量应为 0；同时确认没有 failure snapshot。

## 12. Runtime-ID 记忆写入架构（2026-08-08）

远端提交 `8ad0354` 将段落和证据标识的所有权从二次 reconciliation 模型调用
移到 Runtime。合并后的实验分支遵循以下规则：

- Claude Agent SDK 同时暴露 `Read`、`Edit`、`Write`、`Grep`、`Glob` 和 MCP shell；
  `tools=[]` 会让弱模型看不到内置文件工具，不能恢复这种配置。
- 模型只写正文、语义时间和 source handle；Runtime 分配稳定 block/evidence ID，
  校验来源，重建 Timeline、Recent 和 Relations。
- shell 仅用于列举、移动或删除路径；在 Windows 和 WSL 下都使用 portable POSIX
  语法。已有文件优先用 Edit，新文件使用 Write。
- 内置文件工具的整轮修改通过 `baseline -> commit_edits -> install` 提交；失败时
  整轮回滚并允许一次 repair。repair 再次提交失败必须向 build 抛错，禁止把该
  batch 写成已完成 checkpoint。
- `commit` audit 与 `shell`/`save_memory` 一样计入 build event count，否则会错误
  跳过 final management。
- 合并段落携带的旧 block ID 是 alias。Timeline 和 Recent 只渲染 canonical
  paragraph 一次；Relations 的 `aliases` 映射保存旧 ID 到 canonical ID，避免
  同一段正文在派生视图重复。

本地实验能力必须继续保留：token-balanced batch、原子 checkpoint、writer live
audit/progress、failure snapshot、`.stop-after-current-batch`、item lock、WSL ext4
输出，以及 Windows 文件系统的短暂 PermissionError 重试。远端架构更新后先跑
单元测试，再做 one-session micro-smoke 和单 sample smoke；不要直接恢复旧的部分
checkpoint，以免一个 item 混用两套 writer protocol。

### 12.1 内置编辑与 shell 的事务边界验证

2026-08-08 的首次 Runtime-ID one-session smoke 在 30 turns 后失败。审计表现为
`find/ls/mkdir` 连续返回 `TopicFormatError: memory source links required`，但没有
PermissionError、429、timeout 或 Windows retry。根因不是 shell 或 WSL：内置
Write/Edit 写入的临时 source handle 只会在 commit 时展开，而旧的
`MemoryWorkspace.shell()` 在任何 shell 命令执行前就解析这个未提交中间态，使
只读 `ls/find` 也被严格 Topic 校验拦截，模型随后反复重试到达轮次上限。

修复后，shell 的事务基线一律取自已提交 memory：只读 shell 可以查看未提交
中间态且不会提前安装；移动或删除实际改变 stage 时，内置编辑与 shell 改动作为
一个事务规范化、校验、同步和安装。最终 trajectory commit 仍执行完整 Topic
合同校验，失败仍整轮回滚，因此没有放宽数据完整性要求。

回归验证结果：WSL 管理层、Markdown、派生视图和 Claude Code 适配测试
`126 passed`。相同 index 75 首 session（6 source turns）在新的 ext4 目录中完成：
1/1 batch、1/1 session、6/6 turns、9 agent turns、4 events、0 tool errors、
0 retries、0 failure snapshots，API 阶段约 59.9 秒、构建墙钟 63.4 秒。4 个
canonical Runtime IDs、4 条 evidence、4 个 Topic source links、4 个 Timeline
topic links、4 个 Timeline source links均可解析；Recent 无重复 ID，Relations
无悬空 ID，Core 引用可解析。若以后再次出现同类短错误循环，应先区分固定
TopicFormatError 与真正 shell return code/PermissionError，不要直接提高 max turns。

## 13. 组件消融与 index 105--124 扩展（2026-08-09）

Memory 继续完整构建并冻结。Source、Topic、Timeline、Core、Recent 和
Relations 的主消融在检索阶段通过只读 component mask 完成，不为每个 mask
重复支付构建成本。只有研究问题明确涉及 writer 行为、构建成本或上游依赖变化时，
才做单独的构建期消融。完整规则见：

`docs/experiments/protocols/longmemeval_component_ablation.md`

检索 runner 新增：

```text
--memory-components topics,timeline,sources,core,recent,relations
```

显式 component mask 会关闭可绕过视图的 Bash，并仅为可见 Topic/Source 建立
搜索索引。每个答案记录保存实际 component 列表；旧 `--condition` 组合继续兼容。

已有 index 75--104 共 30 个样本已核验为 9/9 batches、全部 sessions 完成、
`status=complete` 且 `final_management=complete`。这些旧构建按用户要求单独保留为
`excluded_holdout`，暂不进入当前汇总、消融或评分；后续审计后再决定兼容融合，
或按当前协议重新构建。不得删除或覆盖这些输出。机器可读 cohort 记录位于
`code/scripts/configs/longmemeval_build_cohorts.json`。

当前正式 cohort 从 index 105 开始，使用 5 worker 顺序处理 index 105--124：

| Worker | start | limit | 覆盖 index |
| --- | ---: | ---: | --- |
| 1 | 105 | 4 | 105--108 |
| 2 | 109 | 4 | 109--112 |
| 3 | 113 | 4 | 113--116 |
| 4 | 117 | 4 | 117--120 |
| 5 | 121 | 4 | 121--124 |

配置文件名统一为：

```text
code/scripts/configs/longmemeval_ms_20_index105_124_token15k_t120_wsl_ext4_worker{1..5}.json
```

构建仍使用 token cap 15k、`max_turns=120`、完整 final management、WSL ext4
输出和 build-only 模式。组件 mask 只用于之后复用 frozen memory 的 query 阶段，
不改变本轮构建协议。

## 14. Core 容量条件修复与 shard 级安全停靠（2026-08-09）

index 105--124 首轮构建发现多项中后期 Core 超限。旧行为只在提交时发现
`core.md > 2000`，随后让通用 Writer 盲修一次；Writer 的普通提示不包含上限，
修复过程也没有 Runtime 精确计数工具，因此多个样本在 batch 6--9 回滚失败。

新协议保持普通 Writer prompt 和 2000-token 硬上限不变。只有提交明确抛出
`CoreCapacityError` 时才启用条件修复：

- 目标 `core_repair_target_tokens=1800`；
- 单个专项 Agent 最多 `core_repair_max_checks=8` 次精确 token 检查；
- 最多 `core_repair_max_trajectories=2` 个完整专项 Agent；
- 连续 `core_repair_stagnation_limit=2` 次没有缩短时报告 stagnation；
- `core_token_count` 只在条件修复轨迹中可见，普通 Writer 看不到该工具；
- Topic 信息、既有 block ID 和来源引用不得删除；Runtime 仍做最终原子校验，
  绝不自动截断 Core。

`build-checkpoint.json` 的 `writer_trajectories[].core_repair` 记录每个修复轨迹的
触发长度、硬上限、目标、每次检查值、stagnation、提交状态和错误。恢复旧失败
checkpoint 时不得覆盖原输出；先复制到新的 `core-repair-v2` recovery 目录做
单样本 smoke，并在运行记录中标明协议升级。

本轮还确认旧 `.stop-after-current-batch` 只暂停 item 内部 build，外层多样本
shard 会继续下一个 item。runner 现已改为：item 返回 paused 后立即结束 shard；
若当前 batch 在提交前失败但该 item 的 sentinel 仍存在，也结束 shard，不再进入
下一样本。普通失败且没有 sentinel 时仍沿用继续下一个 item 的原策略。

恢复前必须检查并移除目标 item 中遗留的 sentinel；不要修改 index 75--104 的
`excluded_holdout`，也不要直接覆盖 index 105--124 的原始失败快照。

## 15. Core 3k 增量修复协议（2026-08-09）

首个 `core-repair-v2` 恢复 smoke 成功跨过 index117 原失败批次，但专项 Agent
把 2319-token Core 一度压缩到约 900 tokens。结构与引用保持有效，然而这种
回滚后重做整批的方式可能不必要地削弱始终可见的 Core Memory。

后续协议改为：

- `core_max_tokens=3000`；
- `core_repair_target_tokens=2700`，作为留出约 10% headroom 的软目标；
- 普通 Writer prompt 仍不暴露容量数字；
- 只有 Core 容量校验失败时保留完整 staged workspace，不安装也不丢弃；
- 专项 Agent 在这个 staged workspace 上做最小必要压缩，不再重做原始 batch；
- 每个专项 Agent 仍最多 8 次精确检查，最多 2 条轨迹；
- 非 Core 容量类格式、引用或 ID 错误仍整轮回滚，不能借此绕过事务校验。

旧 index75--104 和原 index105--124 配置继续保留为 2k 协议 provenance，不得
原地改写。3k 恢复必须复制原 checkpoint 到新目录，并标记
`core-3k-incremental-v1`。由于每个 worker/item 使用独立 memory workspace，
该调整不引入跨 lane 文件写竞争；主要代价是 Core 常驻检索上下文可能略增。
