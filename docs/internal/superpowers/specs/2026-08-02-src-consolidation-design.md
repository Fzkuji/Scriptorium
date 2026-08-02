# NativeMem `src/` 定版设计

## 目标

将当前 V11 设为唯一 NativeMem 实现，取消版本选择层。核心模块直接位于
`code/src/`，不增加 `nativemem/` 外层目录。删除旧实现、旧版本专用测试、旧版本
专用实验脚本和冻结代码副本。现有 benchmark、结果、第三方仓库和 memory 产物不变。

清理前状态由 Git 提交 `31cc34c` 保存。

## 目标目录

```text
code/
├── src/                         # 仅放可复用的核心实现
│   ├── __init__.py
│   ├── build.py
│   ├── conversation.py
│   ├── management/
│   │   ├── api.py
│   │   ├── agent.py
│   │   ├── block_views.py
│   │   ├── config.py
│   │   ├── event_writing.py
│   │   ├── model_reconciliation.py
│   │   ├── prompts.py
│   │   ├── provider.py
│   │   ├── source_archive.py
│   │   ├── reconciliation.py
│   │   ├── topic_normalization.py
│   │   ├── topic_reconciliation.py
│   │   ├── verification.py
│   │   └── workspace.py
│   ├── markdown/
│   │   ├── models.py
│   │   ├── parser.py
│   │   ├── syntax.py
│   │   └── writer.py
│   ├── retrieval/
│   │   ├── agent.py
│   │   ├── config.py
│   │   ├── context.py
│   │   ├── prompts.py
│   │   ├── runtime.py
│   │   ├── schemas.py
│   │   ├── shell.py
│   │   ├── tools.py
│   │   ├── views.py
│   │   ├── bm25.py
│   │   └── embedding.py
│   ├── runtime/
│   │   ├── online.py
│   │   ├── state.py
│   │   ├── derived_views.py
│   │   └── tokenization.py
├── scripts/                     # 运行命令、benchmark 配置和分析代码
│   ├── adapters/
│   ├── analysis/
│   ├── configs/
│   ├── evaluation/
│   ├── gateways/
│   ├── model_capacity/
│   │   └── calibrate_writer.py
│   └── nativemem/
├── tests/                       # 与 src 并列
├── results/                     # 所有运行产物
│   ├── analysis/
│   └── model_capacity/
├── benchmarks/
└── third_party/
```

`management/` 负责记忆写入、整理、来源归档、事务与校验；`markdown/` 负责
Topic Markdown 解析和渲染；`retrieval/` 负责 grep、BM25、Embedding 和检索
Agent；`runtime/` 负责在线处理状态、token 计数和自动派生视图。`src/` 不包含
benchmark adapters、评估程序或网关服务；这些可执行程序分别归入
`scripts/adapters/`、`scripts/evaluation/` 和 `scripts/gateways/`。
`code/src/evaluation` 仅保留为指向 `../scripts/evaluation` 的相对链接，使哈希锁定且
不可修改的 `scripts/eval_full.py` 继续解析原导入路径；评测实现只维护在
`scripts/evaluation/`，该链接不属于 NativeMem 核心 API。

## 入口

`code/src/__init__.py` 只导出当前正式接口，包括 `BuildConfig`、`build_memory`、
`MemoryConfig`、`MemoryWorkspace`、`QueryConfig` 和 `collect_answer`。调用方不再
使用 `load_version()`、`nativemem_versions.v11` 或任何版本环境变量。

NativeMem 运行脚本统一放入：

```text
code/scripts/nativemem/
├── run_locomo.py
├── run_longmemeval.py
├── reanswer_longmemeval.py
├── common/
├── locomo/
├── longmemeval/
└── ablation/
```

测试统一放入 `code/tests/nativemem/`，按 build、management、markdown、
retrieval 和 runtime 职责拆分。删除只验证 v7、v8、v9、v10 和版本路由的测试。

## 模型长度校准脚本

`code/scripts/model_capacity/calibrate_writer.py` 是独立可执行入口。其核心函数接收
显式的 model client、模型名、两个校准样本和候选 token 长度；命令行入口通过
`--provider-config` 读取本地 provider 配置，不读取环境变量。校准过程
只在完整 session 边界组批，以 2K、4K、8K、16K 的顺序增加候选长度；每个候选
长度重新构建 Topic Memory，并检查事实保留、Markdown 合法性和输出截断。

校准结果写入：

```text
code/results/model_capacity/<provider>--<model>/<run-id>/calibration.json
```

结果至少包含 `model`、`prompt_hash`、`safe_writer_request_tokens`、实际测试的 session
边界、事实保留率、调用次数、token、耗时和失败原因。正式 build runner 通过显式的
`calibration_path` 参数读取 `safe_writer_request_tokens`；组批前计算完整 Writer 请求
长度，加入下一个完整 session 会超过限制时立即提交当前批次。单个 session 不拆分。

模型、Writer prompt 或工具 schema 改变时，使用同一脚本重新生成校准结果。脚本
逻辑由 `code/tests/model_capacity/` 中的本地测试覆盖，测试本身不写入 `src/`。
校准只输出 Writer 容量结果，不输出 LoCoMo benchmark 分数；任何 LoCoMo 对比仍然
只能使用仓库锁定的 `scripts/eval_full.py`。

## 取消 `experiments/` 目录

`code/experiments/` 不再保留。该目录现有内容按职责处理：

- 可执行分析代码移入 `code/scripts/analysis/`。
- `experiment_manifest.json` 和 `run_matrix.jsonl` 移入
  `code/scripts/configs/<study-name>/`。
- CSV、JSON、TeX、分析产物和结果说明移入
  `code/results/analysis/<study-name>/`。
- `code/experiments/*/code/` 中的冻结代码副本删除。
- `__pycache__`、`.pyc` 和 `.DS_Store` 删除。
- 仓库根目录的 `experiments -> code/experiments` 链接删除。

`docs/experiments/` 保存论文实验设计、协议和结果说明，不是可执行代码目录，因此
本次不改名。

## 删除范围

- `code/src/nativemem.py`
- `code/src/v8_memory.py`
- `code/src/v10_memory.py`
- `code/src/nativemem_versions/`
- `code/src/legacy/`
- `code/src/adapters/run_nativemem.py`
- 仅服务旧 NativeMem 版本的脚本、审计脚本和测试
- 完成内容迁移后的 `code/experiments/` 目录
- `docs/method/versions/` 中的旧版本说明

已有 `code/results/`、`code/benchmarks/`、`code/third_party/`、`paper/` 和任何
生成的 memory 目录均不删除、不改名。

## 兼容边界

仓库根目录的 `src -> code/src`、`scripts -> code/scripts`、`tests -> code/tests`
等仍有实际目标的相对链接继续保留；`experiments` 链接随目录删除。保留的链接只
解决仓库路径兼容，不再承担 NativeMem 版本兼容。
`code/src/evaluation -> ../scripts/evaluation` 是唯一的包内兼容链接，只服务于保持
`scripts/eval_full.py` 原文件字节不变。
历史结果中的代码哈希和旧路径不重写；需要复现旧版本时使用 Git 提交
`31cc34c` 或更早提交。

## 验证

重构完成必须满足：

1. 仓库活跃 Python 代码不再导入 `v8_memory`、`v10_memory`、
   `nativemem_versions` 或 `adapters.run_nativemem`。
2. 当前 build、Topic Markdown、派生视图、在线 runtime、BM25、Embedding、
   retrieval agent 和 LoCoMo/LongMemEval runner 测试通过。
3. `scripts/eval_full.py` 的 SHA-256 保持
   `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`。
4. `python scripts/verify_portable_layout.py`、导入检查和 `git diff --check` 通过。
5. 清理前已经位于 `results/` 的文件、benchmark 数据和 memory 产物保持不变；
   新迁移或生成的文件只进入 `results/analysis/` 和 `results/model_capacity/`。
6. 模型长度校准脚本可以使用本地伪 client 完成边界选择，并将结果写入指定的
   `results/model_capacity/` 目录；正式 builder 能通过函数参数读取该结果。
