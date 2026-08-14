# LongMemEval-S WSL 正式实验：问题与改进记录

> 建立时间：2026-08-11（Asia/Shanghai）
> 适用范围：WSL ext4 上的 LongMemEval-S 正式 build-only 实验，重点覆盖 index 125–174 与 275–424。
> 事实来源：原始 runner 配置、run manifest、`build-checkpoint.json`、Writer live audit/progress、stdout/stderr 和 failure snapshot。本文只作集中索引；原始运行文件不得删除、覆盖或改写。
> 运行期约束：当前正式实验运行期间，不修改 repair 次数、prompt、checkpoint 语义、核心写入逻辑或 LoCoMo evaluator/锁定文件。
> 合并状态：2026-08-14 已吸收 `qi202/Scriptorium` 的 `docs/research/LONGMEMEVAL-EXPERIMENT-FINDINGS.md`（可靠性更新提交 `801a75a`）及本地 WSL 记录。本文是跨 WSL/macOS 的当前权威索引；下方初始表保留当时状态，后续“跨机器合并校正表”优先级更高。

## 状态与分类口径

- **已实施—代码**：工作树中已经存在、且可由代码 diff 或测试证明的修改。是否已提交不在本文推断；当前工作树为 dirty。
- **已实施—配置/运行结构**：已经用于正式或恢复运行的配置、provider、分片、并发、安全停靠或目录结构调整。
- **仅运行期缓解**：没有改变核心语义，只通过充值、降并发、换 provider、`--resume` 或安全停靠减轻影响。
- **待实施**：只记录建议；实验结束或用户明确批准前不得实施。
- `failed` 不等于 `complete`。只有最终 checkpoint 明确为 `complete` 才进入正式成功集；stale/building/no-checkpoint 均不得算成功。
- 日志没有逐行时间戳时，本文记录日志文件 `LastWriteTime`，并把事件精确时刻标为“待确认”，不把文件时间伪装成行级时间。

## 当前正式运行配置基线

| 配置项 | 当前值 | 证据 |
|---|---:|---|
| 数据范围与分片 | worker1 275–304；worker2 305–334；worker3 335–364；worker4 365–394；worker5 395–424 | `code/scripts/configs/longmemeval_ms_150_index275_424_core3k_opencode_worker{1..5}.json` |
| 运行模式 | `build_only=true`，恢复由命令行 `--resume` 执行；不运行 query/评分 | 同上；`code/scripts/runners/run_longmemeval.py` |
| Provider/transport | `provider_name=opencode-go-cc-transport`；`base_url=http://127.0.0.1:8787`；`model=deepseek-v4-flash` | 同上；`code/results/development/longmemeval-150-index275-424-core3k-control/worker{1..5}.launch-008-opencode-transport.*` |
| Writer 容量 | `writer_input_token_cap=15000`；`session_batch=8`；`local_reorg_every_sessions=8` | 同上 |
| Core 容量 | hard limit 3000；repair target 2700；max checks 8；max trajectories 2；stagnation limit 2 | 同上；`code/src/management/config.py` |
| 写入与验证 | `verify_writes=true`；`verify_every_sessions=4`；`verify_sources=true`；`final_manage=true` | 同上 |
| WSL shell | `shell_backend=posix-bash`；正式输出位于 `/home/qi2/scriptorium-runs/...` 的 WSL ext4 | 同上；`code/src/shell_backend.py`；各 worker `run_manifest.json`（字段需最终复核） |

## 统一问题与改进表

| 问题或需求 | 观察到的现象/错误 | 已采取的解决措施 | 当前状态 | 后续修改或调优建议 | 原始证据路径 | 相关代码/配置 |
|---|---|---|---|---|---|---|
| **C01 已实施—代码：原子 batch checkpoint 与精确 resume**（index 275–424；worker1–5；基线审计 2026-08-11） | 中断后必须从最后已提交 batch 恢复，不能重写已提交 batch；错误文本不适用。 | `build_memory` 保存 batch plan/hash、completed batch/session/source turn、Writer trajectory，并从原子 checkpoint 续跑。 | 已实施；测试通过。失败轨迹保留，成功样本只取最终 complete。 | 实验结束后增加固定 index 的断电/异常恢复集成测试，验证文件内容 hash 与已提交 batch 不变。 | 各 item `build-checkpoint.json`、`writer-live-batch-*.jsonl.progress.json`；例如 worker1 index276 路径见 F03。 | `code/src/build.py`；`code/tests/runtime/test_writer_capacity.py`；测试：2026-08-11 `34 passed, 1 skipped`（三份定向测试合计）。 |
| **C02 已实施—代码：Writer live audit/progress 与 failure snapshot**（index 275–424；worker1–5；2026-08-11） | 过去只能在失败后看到汇总，难以判断 trajectory 是否仍推进；错误文本因 sample 而异。 | 每个 attempt 生成 live JSONL audit 与 `.progress.json`；repair rejection 生成 `writer-failure-batch-NNN.json`；Agent progress 记录 assistant message/turn/error。 | 已实施；原始失败 snapshot 未被清除。 | 跑完后增加只读汇总器，只输出错误类型/计数/路径，不读取记忆正文。 | 例如 `.../items/0276_*/writer-live-batch-008-attempt-00{1,2}.jsonl.progress.json` 与 `writer-failure-batch-008.json`。 | `code/src/agent_runtime/claude_code.py`；`code/src/management/tools.py`；`code/src/management/agent.py`；`code/src/build.py`。 |
| **C03 已实施—代码：单 item writer lock**（index 275–424；worker1–5；2026-08-11） | 恢复/重启时存在同一 item 双写风险；预期错误为 `item output is already locked by another writer`。 | 为每个 `index_questionId` 创建跨平台文件锁，runner 退出时释放。 | 已实施；未观察到正式 lane 双开污染；测试通过。 | 跑完后保留 lock 冲突为可操作错误，并在 run manifest 中记录冲突 PID/启动标识（不得记录密钥）。 | `<worker-output>/.locks/<index>_<question_id>.lock`；正式 PID/唯一性由启动控制日志记录。 | `code/scripts/runners/longmemeval/support.py`；`code/tests/scripts/test_longmemeval_support.py::test_item_lock_rejects_second_writer_and_releases`。 |
| **C04 已实施—代码：batch 边界安全停靠**（index286 worker1、index376 worker4；2026-08-10） | 运行期需要安全减少活动 lane 时，不能强杀或破坏当前 batch。 | `.stop-after-current-batch` 只在完整 batch commit 后触发 `BuildPaused`；runner 记录 paused 并停止 shard，随后经用户授权删除 sentinel 和 `--resume`。 | 已实施；属于运行结构能力。286/376 后续恢复是否最终 complete：待最终审计。未将 paused 当 failed。 | 增加 manifest 级 pause/resume 时间线测试；保持 sentinel 操作必须明确授权。 | worker1 `.../items/0286_*/.stop-after-current-batch`（历史，现已删除）；worker4 `.../items/0376_*/.stop-after-current-batch`（历史，现已删除）；对应 `build-checkpoint.json` 的 paused 字段。 | `code/src/build.py`；`code/scripts/runners/run_longmemeval.py`；`code/scripts/runners/longmemeval/support.py`；相关 runner/writer-capacity 测试。 |
| **C05 已实施—代码：WSL/DrvFS 权限与 POSIX shell 适配**（全体 WSL lane；2026-08-09 至 2026-08-11） | Windows ACL/DrvFS 的合成 mode bits 会误触发 key 文件必须0600；Windows/WSL shell 语义不一致可能引发 shell/commit 错误。 | 仅在有真实 POSIX 权限语义的文件系统检查0600；增加 `auto/posix-bash/native` shell backend，正式配置固定 `posix-bash`。 | 已实施；本轮未发现 launch-008 stderr 或 shell/commit 新错误。 | 跑完后在真实 WSL ext4 与 `/mnt/e` DrvFS 各跑一组权限/命令集成测试。 | 配置文件；launch-008 stderr（当前长度0）；早期 launch-001 stderr 的工作目录错误另见 R01。 | `code/scripts/runners/run_longmemeval.py`；`code/src/shell_backend.py`；`code/src/management/workspace.py`；`code/tests/scripts/test_longmemeval_runner.py`。 |
| **C06 已实施—代码：Core 容量 repair 与精确 token 检查**（index276 worker1；日志 mtime 2026-08-10 23:59:24） | `memory writer core-capacity repair was rejected: ValueError: unused Core footnote: e-…`；失败发生在 batch8。 | 已实现硬上限3000、软目标2700、最多8次检查/2条 repair trajectory/停滞阈值2，并提供只读 `core_token_count` 工具；index276 从原子 batch8 checkpoint 重跑。 | 代码已实施；index276 已恢复为 complete，checkpoint mtime 2026-08-11 03:11:17；旧 failure snapshot 保留，不污染最终成功集。 | 跑完后用 index276 做固定-index恢复回归；检查 repair 后引用/footnote 完整性，禁止简单吞掉验证错误。 | `code/results/.../worker1.launch-002.stdout:4`；`\\wsl.localhost\Ubuntu-24.04\home\qi2\scriptorium-runs\longmemeval-ms-150-index275-424-core3k-worker1-r1\items\0276_gpt4_af6db32f\writer-failure-batch-008.json`；同目录 checkpoint/progress。 | `code/src/management/config.py`；`code/src/management/tools.py`；`code/src/management/agent.py`；worker config core repair 值。 |
| **C07 已实施—代码：Writer 输入容量规划与审计元数据**（index369 worker4；日志 mtime 2026-08-10 23:51:11） | `one complete message exceeds the calibrated Writer input limit`；该失败没有 build checkpoint/snapshot。 | 已在 batch plan 记录 session indices/input tokens/tokenizer，并按完整 message 打包；单条 message 超限时明确失败而非截断。 | 容量检测已实施；index369 当前仍是 no-checkpoint 逻辑失败，尚未恢复，未计 complete。 | **待实施**：在不改变语义的前提下，为 preflight/no-checkpoint 失败写外层原子 failure record；是否允许单 session 分块需实验结束后单独设计和批准。 | `code/results/.../worker4.launch-002.stdout:10`；预期目录 `...worker4-r1/items/0369_852ce960/`，当前无 `build-checkpoint.json`。 | `code/src/build.py`；`code/src/runtime/capacity.py`；`code/tests/runtime/test_writer_capacity.py`；`writer_input_token_cap=15000`。 |
| **R01 已实施—配置/运行结构：正式输出迁入 WSL ext4**（index125–174、275–424；worker1–5；2026-08-09 起） | launch-001 出现工作目录/路径错误；Windows/DrvFS 路径兼容性和 I/O 语义不稳定。具体错误文本需从 stderr 复核。 | 正式 run output 改到 `/home/qi2/scriptorium-runs/...`；Windows 侧只保留 wrapper/control logs。 | 已实施；launch-001 属历史启动错误，不计 sample 失败。精确 stderr 文本待确认。 | 跑完后记录 ext4 与 DrvFS I/O/锁/权限差异，不迁回 DrvFS。 | `code/results/development/longmemeval-150-index275-424-core3k-control/worker*.launch-001.*`（当前批早期日志路径需确认）；`...longmemeval-50-index125-174-core3k-control/worker*.launch-001.stderr`，mtime 2026-08-09 23:14–23:15。 | 五路 config 的 `output_dir`；`docs/research/LONGMEMEVAL-WSL-RUNBOOK.md`。 |
| **R02 已实施—配置/运行结构：5×30 静态分片**（index275–424；worker1–5；2026-08-10） | 需要并行构建150个 sample，同时避免范围重叠。 | 固定分片275–304、305–334、335–364、365–394、395–424；每路独立 output_dir/config/wrapper。 | 已实施；未发现分片重叠。当前完成度以实时 checkpoint 为准。 | 下一批425以后启动前先核对数据集上界和既有分配清单。 | 五份 worker config；五个 WSL output roots；control launch logs。 | `code/scripts/configs/longmemeval_ms_150_index275_424_core3k_*worker*.json`。 |
| **R03 已实施—配置/运行结构：严格 build-only**（index275–424；worker1–5） | 要求构建与 query/评分隔离，避免失败恢复时混入评测调用。 | `build_only=true`；监控禁止启动 query/评分。 | 已实施；当前结果只代表 memory build。 | 全部 build 核验通过后再由用户单独授权 query/评分。 | 五份 OpenCode worker config；run manifest（最终需复核）。 | `code/scripts/runners/run_longmemeval.py`。 |
| **R06 已实施—配置/运行结构：专用 OpenCode Claude Code transport**（index275–424；worker2/3/5受旧路径影响；2026-08-11） | 通用 LiteLLM launch-007 共31个独立 sample 出现 HTTP400 `invalid_request_error: No tool output found for tool call ...`：worker2 index311–313、315–319（8个）；worker3 index335–337、343–344、346–350、352–354（13个）；worker5 index401、403–411（10个）。这是 assistant tool call 与后续 tool output 在适配/序列化链路中未正确配对的协议错误；现有证据不足以归因成“模型忘记调用工具”。 | 停用通用 LiteLLM 路径；先做 MCP tool round-trip 与真实 Writer smoke；成功后固定专用 transport `127.0.0.1:8787`、`deepseek-v4-flash`，五路使用 launch-008 从原 checkpoint `--resume`。 | 已实施；专用 launch-008 未再发现 `No tool output found for tool call`、tool_use/tool_result 或未知工具错误，stderr 当前为0。旧 launch-007 failed 不得算 launch-008 新失败，也不进入 complete。transport health 偶发3秒探针 false但 checkpoint 仍推进，暂判探针瞬时失败。 | 为专用 transport 增加多轮并行tool_use/tool_result配对、重复/缺失/乱序结果、reasoning block与错误返回的协议测试；health endpoint区分存活、忙碌和协议异常。 | `code/results/development/longmemeval-150-index275-424-core3k-control/worker2.launch-007-opencode.stdout:14–37`（mtime 2026-08-11 02:21:17）；`worker3.launch-007-opencode.stdout:2–56`（mtime 02:21:27）；`worker5.launch-007-opencode.stdout:14–43`（mtime 02:22:14）；对照 `worker{1..5}.launch-008-opencode-transport.{stdout,stderr}`。 | 五份 OpenCode config；专用 transport 实现/启动脚本路径仍待确认。 |
| **R07 仅运行期缓解：电脑重启后的 checkpoint 恢复**（index125–174；worker1–5；2026-08-10 02:26 launch-003） | launch-002 因电脑重启退出；重启前旧 wrapper 退出不是新异常。此前10个旧 ECONNRESET 失败的具体 checkpoint error 字段待复核。 | 经用户授权以 launch-003 从原 checkpoint `--resume`；按分片起点遍历、跳过 complete、回收 failed，再继续中断 item。 | 已实施运行恢复；launch-003 stderr 均为0。是否50个全部最终完成及 ECONNRESET 精确 index/错误文本：待确认。 | 跑完后补录最终成功/失败清单与原子恢复点；将“wrapper退出”和“sample失败”分离为不同事件类型。 | `code/results/development/longmemeval-50-index125-174-core3k-control/worker{1..5}.launch-002/003.*`；WSL `longmemeval-ms-50-index125-174-core3k-worker*-r1/items/*/build-checkpoint.json`。 | 对应50-sample worker configs；原子 resume 代码 C01。 |
| **F01 运行失败：dangling block link**（worker3 index335，Packy launch-002；worker3 index347，OpenCode launch-008；日志 mtime 2026-08-11 01:30 / 当前文件 mtime随运行更新） | `memory writer repair was rejected: ValueError: dangling block link: …`。 | 保留 `writer-failure-batch-*.json` 与 attempt progress；runner 继续其他 sample；后续通过 `--resume` 从原子 checkpoint 恢复。 | index335 已 complete（checkpoint mtime 2026-08-11 03:53:18，batch2有 attempt2）；index347 仍为 stale/building checkpoint 4/9，launch-008 独立失败 attempt 1/3，未计 complete。未回滚原始失败轨迹；partial checkpoint 不进入正式成功集。 | 待实施：针对 dangling relation 做 repair 前静态图校验和固定-index347恢复测试；不可静默删除关系。 | index335：`...worker3-r1/items/0335_gpt4_9a159967/writer-failure-batch-002.json`；index347：`.../0347_gpt4_93f6379c/writer-failure-batch-000.json`、progress、checkpoint；`worker3.launch-008-opencode-transport.stdout:26`。 | Writer verification/repair：`code/src/management/agent.py`、`workspace.py`、`errors.py`。 |
| **F02 运行失败：disallowed special token**（worker3 index351；launch-008 line34；2026-08-11，精确行时刻待确认） | `Encountered text corresponding to disallowed special token '&lt;&#124;endoftext&#124;&gt;'.` | 当次运行未热改且将3条日志合并为一次独立 attempt；后续代码已将模型文本按 ordinary text 编码并补回归测试，未修改 prompt。 | 历史 index351 attempt 1/3、no-checkpoint，未计 complete；**代码缺陷已修复，但固定-index正式恢复结果仍待确认**。旧失败不因修复落地而被覆盖。 | 运行 index351 固定恢复验证，确认 token 计数与普通文本语义；补 no-checkpoint 外层原子 failure record。 | `code/results/.../worker3.launch-008-opencode-transport.stdout:34–37`；`...worker3-r1/items/0351_gpt4_78cf46a3/`（历史无 checkpoint）；qi202提交`801a75a`。 | `code/src/runtime/tokenization.py`；`code/tests/runtime/test_writer_capacity.py`。 |
| **F03 运行失败：Core repair rejection 后恢复成功**（worker1 index276；见C06） | unused Core footnote。 | 从 batch8 原子 checkpoint重跑，生成 attempt2，旧 snapshot保留。 | 已恢复 complete；无正式结果污染。 | 固定-index回归与引用完整性核验。 | 同C06。 | 同C06。 |
| **F04 运行失败：单条 complete message 超 Writer cap**（worker4 index369；见C07） | 单条完整 message 超校准输入限制。 | runner记录 stdout failed 并继续；未截断输入。 | no-checkpoint，待恢复/待清理；未计 complete。 | 为 preflight 失败补原子 failure record；分块策略须另行批准。 | 同C07。 | 同C07。 |
| **F05 网络错误分类：401**（当前审计范围） | 未发现可确认的 HTTP401。早期监控曾因裸搜数字误把 sample index401 当HTTP401。 | 监控已要求带HTTP/error语境匹配。 | 已纠正监控口径；是否其他历史批存在401：待确认。 | 日志解析改为结构化状态字段，不对裸数字做错误匹配。 | 当前 launch-002/008 stdout/stderr；监控自动化历史。 | 待实施只读错误汇总器。 |
| **F06 网络错误分类：timeout / FailedToOpenSocket / connection reset / 5xx**（当前275–424） | launch-002/008 基线扫描未发现可确认新增 timeout、FailedToOpenSocket、connection reset 或5xx。index125–174 的旧 ECONNRESET 由此前恢复记录提及，但原 checkpoint 文本本次尚未复核。 | 当前仅监控；没有代码/配置热改。 | 275–424：未观察到；125–174：待确认具体 index、时间与证据字段。 | 跑完后用结构化脚本扫描所有 checkpoint/error/snapshot，分别计数，不合并为“网络错误”。 | 当前 control logs；125–174 WSL item checkpoints与launch-002/003 logs。 | 待实施只读审计脚本与测试。 |
| **F07 final management 长等待/进度可见性**（多个 index，例如289、410；worker1/5；2026-08-11） | 9/9 batch后可长时间 final pending；index289最终完成，未出现错误；index410排除旧中断后的有效耗时约1:36:58。 | 依靠 progress/checkpoint/activity监控，不把短暂无写入误报为失败；不干预 final。 | 运行期监控缓解；没有证据表明正式结果污染。 | 待实施 final-management phase heartbeat/开始结束时间字段，区分正常长调用、transport等待和真实停滞。 | 对应 item `build-checkpoint.json` 与 progress mtime；例如 worker5 `items/0410_59524333/`。 | `code/src/build.py`；Agent progress callback。 |
| **P01 待实施：结构化错误账本/只读审计器**（全体 index） | stdout同一失败可有摘要+细节；no-checkpoint失败易漏计。 | 仅在本文集中索引，未改代码。 | 待实施。 | 输出 `index/worker/attempt/error_type/final_status/evidence_path`，按独立运行边界去重；绝不读取正文。 | 本文所有 control logs/checkpoints/snapshots。 | 建议新增 diagnostics 脚本与单元测试；不得新建评分器。 |
| **P02 待实施：no-checkpoint failure 原子记录**（index351、369） | 失败发生在 checkpoint建立前，目录存在但缺少可恢复状态。 | 当前依赖 stdout，未热修。 | 待实施。 | 在外层 runner 写独立 atomic failure metadata，不改变 build checkpoint 语义；需用户批准后实施。 | index351/369目录与stdout。 | `code/scripts/runners/longmemeval/support.py`（候选位置）。 |
| **P04 已实施：special token 安全编码**（index351） | 模型文本触发 tokenizer disallowed special。 | tokenizer 对普通文本显式使用 `disallowed_special=()`；保留 literal sentinel 的回归测试，不修改 prompt。 | 已实施；原 index351 失败仍作为历史证据，不能把代码修复反写成当次成功。 | 在固定 index 回归中确认 ordinary-text 编码与 token 计数一致。 | index351 stdout；仓库可靠性更新提交 `801a75a`。 | `code/src/runtime/tokenization.py`；`code/tests/runtime/test_writer_capacity.py`。 |
| **P05 已实施：Writer/关系图 bounded repair 与诊断**（index335、347、440、473、486；Core容量另见276、460） | dangling link、unused footnote在repair提交阶段被拒绝；Mac item473曾耗尽三条 generic repair trajectory。 | 对 generic validation failure 增加有界修复、attempt 编号的 live audit/progress/failure snapshot，并向修复路径提供 dangling-link 局部诊断；最终完整性规则不放宽。 | 已实施并由定向测试覆盖；历史失败轨迹保留。是否每个固定 index 均已正式恢复成功仍以各自最终 checkpoint 为准。 | 继续保留固定-index canary；若有界修复耗尽，应终止并保留结构化快照，不能自动删关系或无限重试。 | 本地335/347 failure snapshot；Mac440/473/486时间线及 canary 证据见提交 `801a75a` 的原 findings。 | `code/src/management/agent.py`、`block_views.py`、`prompts.py`；`code/src/runtime/derived_views.py`；相关 management/runtime 测试。 |
| **P06 待实施：final phase 结构化心跳**（index289、410等） | final pending长时间无checkpoint写入，监控难区分忙碌与停滞。 | 目前用mtime与两轮阈值缓解。 | 待实施。 | 增加 final phase start/last-progress/end 字段与transport call状态，不改变final语义。 | item checkpoint/progress时间线。 | `code/src/build.py`、Agent progress。 |
| **P07 待实施：固定-index恢复验证与最终失败清理流程**（276、335、347、351、369及最终失败集） | 恢复成功、stale与no-checkpoint状态混杂。 | 本轮只记录，不清理。 | 待实施，须等待全批自然结束和用户授权。 | 先备份/校验原始证据；按固定index `--resume`；核验组件、unsupported=0、probe refs；仅在成功后更新正式状态，原失败轨迹永久保留。 | 各 index路径与最终run manifest。 | runner/support/build及相关测试。 |

## 2026-08-14 跨机器合并校正表

本节把 qi202 仓库中以 Mac 时间线记录的问题归并到上面的根因表。它是当前状态的权威补丁：若与建立于 2026-08-11 的初始行冲突，以本节为准。合并只吸收能够指导代码、恢复语义、Writer、transport、checkpoint、验证或可观测性优化的事项；额度、充值、余额和单纯供应商限流不进入调优问题计数。

| 合并根因 | 受影响 index / worker / 时间 | 观察到的错误或现象 | 合并后的措施与状态 | 回滚、污染与原子恢复点 | 原始证据与代码 |
|---|---|---|---|---|---|
| **M01 Writer 图一致性与有界修复**（合并 F01/P05） | WSL：335、347，worker3，2026-08-11；Mac：440、473、486，2026-08-11；精确时刻见各日志，缺失者待确认 | `dangling block link`；item473 曾耗尽三条 generic repair trajectory；item440 的原编辑与 generic repair 均被完整性校验拒绝。 | **已实施—代码**：有界 generic repair、attempt 编号的审计/进度/失败快照、dangling-link 局部诊断；最终验证规则保持严格。同步后的定向测试为 `106 passed`。 | 被拒绝的 staged edit 未提交，不污染最后原子 checkpoint；正式恢复必须从最后完整 batch 继续。各 index 最终成功状态仍以原 checkpoint 为准，不由文档反推。 | WSL `items/0335_*`、`items/0347_*` failure snapshot；Mac 原 findings 的440/473/486时间线；`code/src/management/{agent,block_views,prompts}.py`、`code/src/runtime/derived_views.py`。 |
| **M02 Writer/Core 容量边界**（合并 C06/C07） | WSL：276、369，worker1/4，2026-08-10；Mac：460、244、487，worker2及对应worker，2026-08-11；精确行时刻部分待确认 | 276/460 为 Core 容量或 footnote 修复失败；369 为单条完整 message 超 Writer cap；244/487 为 SDK 单条 JSON 超 reader buffer。 | **部分已实施**：Core 有硬上限、目标和有界修复；SDK buffer 经实际失败由1 MiB提高至8 MiB、再定为20 MiB，并增加 inactivity timeout。**待实施**：369 这类不可分单消息的 preflight 原子失败记录及通用分块策略。 | 容量修复不应截断原文；失败 staged edit 不提交。276 已从 batch8 恢复；其他项目的最终状态依原 checkpoint。buffer 调整是传输容量修复，不改变记忆语义。 | 对应 checkpoint/failure snapshot；`code/src/agent_runtime/claude_code.py`、`code/src/build.py`、`code/src/management/config.py`、`code/src/runtime/tokenization.py`。 |
| **M03 SDK 无输出/长调用可观测性**（合并 F07） | WSL：289、410 等；Mac：498、62，2026-08-12 | 进程仍存活但 checkpoint、progress 或 SDK 流长时间没有可见更新；没有足够证据把沉默直接判成模型错误。 | **已实施—代码/运行结构**：SDK inactivity timeout、batch 边界安全停靠、原子 resume；监控以多来源 mtime 和连续轮次判定。**待实施**：final phase 和 transport call 的结构化心跳。 | 存活但沉默的 runner 不视为空槽，不叠加任务、不强杀；498/62 的停靠与恢复从最后原子 checkpoint 进行。 | Mac498/62时间线；`code/src/agent_runtime/claude_code.py`、`code/src/build.py`、`code/scripts/runners/run_longmemeval.py`。 |
| **M04 tokenizer literal sentinel**（校正 F02/P04） | WSL index351，worker3，2026-08-11，行级时刻待确认 | `Encountered text corresponding to disallowed special token '&lt;&#124;endoftext&#124;&gt;'`。 | **已实施—代码**：普通文本编码显式允许 literal sentinel 作为普通字符串，并补回归测试；不修改 prompt。 | 原 no-checkpoint 失败轨迹不被删除，也不能因修复落地而追记为当次成功；固定-index恢复需另行核验。 | `worker3.launch-008-opencode-transport.stdout`；`code/src/runtime/tokenization.py`；writer-capacity/tokenization测试。 |
| **M05 审计化 resume drift** | Mac item258，2026-08-13 08:02 HKT；其他身份漂移样本待确认 | 严格 resume 会因配置/运行身份漂移拒绝继续；直接放宽会掩盖不兼容恢复。 | **已实施—代码**：仅通过显式 opt-in 允许受审计的 resume drift，记录差异；默认仍严格。 | 不覆盖旧 checkpoint；若差异不在允许范围则继续拒绝。是否污染由 manifest 中的 drift 记录和最终验证决定。 | Mac原 findings item258；`code/scripts/runners/conversation/{config,runner}.py`、`code/scripts/runners/longmemeval/support.py`、`code/scripts/runners/run_longmemeval.py`。 |
| **M06 冻结清单与评测输入边界** | LongMemEval 已完成构建集合，2026-08-13至14；具体 index 由生成的 inventory 决定 | 滚动构建、问答和 Judge 并行时，若没有冻结来源，容易把未完成、身份不匹配或后续变化的 checkpoint 混入评测。 | **已实施—代码/工具**：新增冻结 inventory，只收录符合状态和身份条件的元数据与 hash；原始运行文件继续作为事实来源。 | 工具只读，不改 checkpoint；清单冲突应停止而不是覆盖答案。 | `code/scripts/analysis/build_longmemeval_frozen_inventory.py`；`docs/research/LONGMEMEVAL-RELIABILITY-UPDATE-2026-08.md`。 |
| **M07 portable checkpoint 路径** | Mac 导入的142–174、302、303、334、362等，2026-08-13；逐项时间待确认 | checkpoint 保存相对 `paths.memory_dir`，跨机器导入后可能被判断为 missing source memory。 | **仅运行期缓解 + 待实施**：导入副本按实际位置修正并保留来源；建议把可迁移路径与本机绝对路径分离，并在加载时做身份/hash核验。 | 不改原始正式 checkpoint；只修正导入副本。污染风险需靠源 hash、dataset index、question identity 和 memory validity 联合排除。 | Mac原 findings portable checkpoint事件；`code/scripts/runners/longmemeval/support.py`、frozen inventory工具。 |
| **M08 macOS launchctl 生命周期与 shell 可移植性** | Mac 多个 worker，2026-08-11至13；具体 label/时间见原 findings | 已完成/失败 job 可能被 launchctl 再调度；BSD `sed` 与 GNU 参数不同，影响批量配置修改。 | **已实施—运行结构**：正式运行中移除已终结 label、使用明确的 GNU 工具链/命令；runbook 已补充。**待实施**：受测的 launch wrapper，在终态后自卸载并拒绝重复 label。 | 这是调度与配置风险；只有 checkpoint/进程证据确认未双写时才能判定未污染。 | qi202原 findings launchctl条目；`docs/research/MAC-LONGMEMEVAL-RUNBOOK.md`。 |
| **M09 transport tool-use/tool-result 配对**（沿用 R06） | WSL worker2/3/5 的31个独立 sample，2026-08-11 | 通用 LiteLLM 路径出现 HTTP400 `No tool output found for tool call`；不能据此归因为“模型不会调用工具”。 | **已实施—运行结构**：切到专用 OpenCode transport；当前同步未引入第二套协议。**待实施**：乱序、重复、缺失 tool result 和 reasoning block 的协议回归测试。 | 旧失败不计新 transport 的失败；恢复从原 checkpoint，正式结果只收 complete。 | launch-007/008对照日志；五份 OpenCode config；transport实现路径仍待补齐。 |
| **M10 WSL/macOS 统一平台接口** | 新实验；2026-08-14；不追记为历史正式 run 的既有配置 | 两个平台使用同一 runner，但 shell 默认、路径和主机身份原先分散在 runbook/启动命令中，容易把 Mac 配置误启动到 WSL，或产生不可审计的平台差异。 | **已实施—代码**：新增 `--platform-profile auto|wsl|macos`；显式 profile 与主机不匹配时拒绝；WSL/macOS 自动 shell 统一解析为 `posix-bash`；manifest/invocation 记录 requested/detected platform、resolved shell 与输出存储类别。聚焦测试 `26 passed`。 | 不改现有 checkpoint 和正式配置；不自动选择 output_dir，不管理 launchctl/nohup，不改变 prompt、Writer、repair、checkpoint 或评测语义。 | `code/src/execution_platform.py`；`code/scripts/runners/run_longmemeval.py`；`code/tests/runtime/test_execution_platform.py`；两份平台 runbook。 |

### 合并后的实施与待办口径

- **已实施代码能力：12项。** 原 C01–C07 七项，加上 SDK inactivity/buffer、special-token ordinary-text 编码、审计化 resume drift、冻结 inventory、WSL/macOS统一平台接口五项。M01 的 bounded repair 作为 C02/P05 的升级合并计算，不重复计数。
- **已实施配置/运行结构：7项。** 原 R01–R03、R06–R07 五项，加上 macOS launchctl 终态 label 管理和 GNU shell/toolchain 约束两项；临时额度、余额、充值不计。
- **仍待实施建议：8项，其中近期候选5项、用户明确延期3项。** 近期候选为结构化错误账本、不可分消息策略、portable checkpoint schema、launch wrapper 自卸载、transport 协议回归。2026-08-14 用户明确延期：no-checkpoint 原子失败记录、final/transport 心跳、固定-index恢复验收；除非用户重新授权，不进入近期实现。已经落地的 special-token 和 bounded repair 不再列为待实施。
- **测试结果：** 2026-08-14 在 WSL 运行可靠性合并聚焦测试，`106 passed in 10.06s`；新增平台接口另运行 runner/support/platform 聚焦测试，`26 passed in 6.07s`。LoCoMo evaluator 未修改，其锁定 SHA-256 已核验为 `17ef2179cd2781880649eed4a7d62988c069123b5044f007c194cdab8e63f88b`。
- **证据保留：** qi202 文档中的逐时事件仍可从提交 `801a75a` 读取；本文不复制问题、答案、记忆或 verification 正文，也不以 Markdown 替代原日志。

## 初始 WSL 基线计数（历史快照）

2026-08-11 的初始 WSL 盘点把同一根因的多文件修改合并为一个能力项，当时共 **7 项**：C01–C07。该计数已被上方 2026-08-14 合并口径取代，仅用于说明当时验证范围。定向测试结果为：

```text
34 passed, 1 skipped in 4.31s
```

测试命令使用 `.venv-win/Scripts/pytest.exe`，覆盖：

- `code/tests/runtime/test_writer_capacity.py`
- `code/tests/scripts/test_longmemeval_runner.py`
- `code/tests/scripts/test_longmemeval_support.py`

系统 Python 缺少 pytest（`No module named pytest`），因此改用项目虚拟环境；这不是产品代码失败。

## 初始配置/运行结构计数（历史快照）

2026-08-11 时共 **5 项**：R01–R03、R06–R07。其中 R07 属于**仅运行期恢复**，不代表核心代码语义改变。当前合并计数见上方；外部额度、充值和低余额并发限制始终不纳入改进项统计。

## 初始待实施建议计数（历史快照）

2026-08-11 时共 **6 项**：P01–P02、P04–P07。其中 P04、P05 后来已实施，不能再按当前待办计数；当前 **8 项**合并待办见上方。

## 仍缺少证据、需要后续确认

1. index125–174 旧 ECONNRESET 的精确 index、checkpoint error 文本、发生时间与最终恢复状态。
2. 当前批 launch-001 工作目录错误的具体 stderr 文本和对应启动时间；它只属于启动失败，不应计 sample failed。
3. 专用 OpenCode transport 实现与启动脚本的仓库路径，以及 Writer smoke 的输出目录/测试记录。
4. 每个 worker `run_manifest.json` 中 `provider_migrations`、最终 provider、launch id 与恢复次数的最终值。
5. index286、376 安全停靠后的最终 complete 状态，以及 sentinel 创建/删除的精确时间。
6. index347、351、369 的最终正式恢复状态；代码修复已经落地不等于这些历史运行已自动转成成功。
7. 本地同步修改尚未提交；其上游来源可追溯到 qi202 提交 `801a75a`，本地最终 commit SHA 仍待产生。
8. Mac 原 findings 中部分事件只有日志文件时间或人工巡检时间，缺少逐行时间戳；需要最终证据清单明确标为 mtime/待确认。

## 跑完后修改计划

> 2026-08-14 优先级决定：`no-checkpoint` 原子失败记录、final/transport 心跳、固定-index恢复验收暂缓。保留问题与证据，但不纳入下一轮代码修改。其余五项可在后续修改阶段实施。

| 优先级 | 计划 | 完成条件 |
|---:|---|---|
| P0 | 冻结并备份五路 run manifest、checkpoint、audit/progress、stdout/stderr、failure snapshots；生成只读错误账本。 | 原始证据路径与hash可追溯；不删除失败轨迹。 |
| P0 | 最终成功样本核验：Sources/Topics/Timeline、Core、Recent、Relations、verification非空；unsupported=0；Runtime-ID/probe refs完整。 | failed不计complete；每个成功index有核验记录。 |
| 暂缓 | 固定-index恢复验证：276、335、347、351、369以及最终失败集。 | 用户重新授权后再执行；历史证据和原子 checkpoint 保持原样。 |
| P1 | 实施P01：结构化错误账本并补测试；不包含已暂缓的 no-checkpoint 原子 failure record。 | 能区分401/403/429/quota/timeout/socket/reset/5xx及程序错误；同attempt去重。 |
| P1 | 验收已实施的 P04/P05：special token ordinary-text 编码与 Writer 有界图修复。 | 聚焦单元测试已通过；仍需 index351/347 固定恢复回归，且验证规则不放宽。 |
| P1 | 设计 portable checkpoint schema 与受测的 launchctl wrapper。 | 跨机器路径不再依赖导入后手工修正；终态 job 不会被重新调度。 |
| 暂缓 | 实施P06：final phase heartbeat与transport健康信号。 | 用户重新授权后再实施；当前继续沿用日志、progress和checkpoint活动时间判断。 |
| P2 | 正式失败项清理。 | 仅在备份、恢复、核验全部完成并获用户授权后更新正式汇总；原始失败文件永久保留。 |

## 持续维护规则

- 后续巡检发现新根因时，优先更新本表已有条目的“受影响 index/worker/时间/证据/状态”；同一根因不创建重复章节。
- 每次更新必须先读原始运行文件；不得仅依据旧 Markdown 或旧心跳数字。
- 若证据冲突，记录冲突与“待确认”，不选择性覆盖。
- 代码/配置若在实验期间未实际部署，状态必须保持“待实施”，不得写成“已修复”。
- 外部额度耗尽、充值、账户余额导致的并发限制属于不可抗力运营因素，不作为代码优化问题或待实施建议记录；只有它暴露出独立、可复现的代码缺陷时，才记录该代码缺陷本身。
- 不触碰 `code/scripts/evaluation/eval_full.py`、LoCoMo evaluator 或相关锁定文件。
