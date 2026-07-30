# MemoryOS (EMNLP 2025, BAI-LAB)

- 仓库：https://github.com/BAI-LAB/MemoryOS
- 调研日期：2026-07-02（main 分支，depth-1 clone）
- 结论速览：**有自带 LoCoMo 评测代码**（`eval/` 目录）。**没有 LLM judge**——唯一的自动指标是词级 set-based F1。README/论文宣称的 BLEU-1 结果在仓库里**没有对应代码**。没有 LongMemEval 评测。

## 评测脚本位置

全部在仓库根目录 `eval/` 下（README "🎯Reproduce" 一节指向这里）：

| 文件 | 作用 |
|---|---|
| `eval/main_loco_parse.py` | 主脚本：读入 `locomo10.json` → 把对话灌进 MemoryOS 三层记忆 → 逐条 QA 检索+生成答案 → 输出 `all_loco_results.json` |
| `eval/evalution_loco.py` | 指标脚本：对结果按 category 算平均 F1（仅此一个指标） |
| `eval/locomo10.json` | 打包进仓库的 LoCoMo 数据集（10 个 sample，共 1986 条 QA） |
| `eval/short_term_memory.py` / `mid_term_memory.py` / `long_term_memory.py` / `dynamic_update.py` / `retrieval_and_answer.py` / `utils.py` | eval 专用的一份独立记忆系统实现（与 pypi 包代码分开） |

复现命令（README 原文）：

```bash
cd eval
# Configure API keys and other settings in the code
python3 main_loco_parse.py
python3 evalution_loco.py
```

`memoryos-pypi/`、`memoryos-chromadb/`、`memoryos-mcp/`、`memoryos-playground/` 里没有任何评测代码。

## Answer prompt（原文）

位置：`eval/main_loco_parse.py`，函数 `generate_system_response_with_meta()`（第 80–142 行）。回答模型 `gpt-4o-mini`，`temperature=0.7`，`max_tokens=2000`，API base_url 硬编码为第三方代理 `https://cn2us02.opapi.win/v1`。

**没有 "5-6 words" 这类显式字数限制**，但有更强的简洁约束："answer ... in an extremely concise manner" 出现两次，外加一个 one-shot 例子要求只回实体短语，以及严格的日期格式指令。

System prompt 原文：

```python
system_prompt = (
    f"You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}. "
    f"Here are some of your character traits and knowledge:\n{assistant_knowledge_text}\n"
    f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
    f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
    f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
)
```

User prompt 原文：

```python
user_prompt = (
    f"<CONTEXT>\n"
    f"Recent conversation between {speaker_a} and {speaker_b}:\n"
    f"{history_text}\n\n"
    f"<MEMORY>\n"
    f"Relevant past conversations:\n"
    f"{retrieval_text}\n\n"
    f"<CHARACTER TRAITS>\n"
    f"Characteristics of {speaker_a}:\n"
    f"{background}\n\n"
    f"the question is: {query}\n"
    f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
    f"Please only provide the content of the answer, without including 'answer:'\n"
    f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
    f"If the question is about the duration, answer in the form of several years, months, or days.\n"
    f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
)
```

值得注意的两点：

1. **one-shot 例子疑似取自 LoCoMo 数据本身**：system prompt 里的示例问题 "What did the charity race raise awareness for?"（答 "mental health"）对应 LoCoMo 里 Caroline 的 charity race 情节——数据集中甚至有一条 category 5 对抗问题 "What did Caroline realize after her charity race?"（adversarial_answer: "self-care is important"）。等于用 benchmark 内容做了 in-context 示范。
2. 日期指令强制 `"15 July 2023"` 格式并要求把相对时间（"last year"）换算成绝对年份——这是针对 LoCoMo temporal 类问题的词面 F1 优化，因为 F1 是纯词重叠，没有任何日期容差逻辑。

## Judge 配置

**该仓库没有 LLM judge。** 全仓库 `grep -i "judge\|bleu\|rouge"` 在 `.py` 文件里零命中（只有 README 提到 BLEU 数字）。没有 judge 模型、没有 judge prompt、没有 0/1 或连续打分、没有日期容差——唯一指标是下面的词级 F1。

```text
（无 judge 代码可引用）
```

## 指标计算

文件：`eval/evalution_loco.py`。

- `simple_tokenize()`：lowercase + `re.findall(r'\b\w+\b', text)` 分词。
- `calculate_f1()`：**set-based** token overlap F1（先 `set()` 去重，再算 precision/recall）。注意这不是 SQuAD 标准的 multiset F1，去重后对重复词不敏感，通常比标准 F1 略高。

```python
def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate F1 score for prediction against reference."""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1
```

- 没有做 SQuAD 式的 normalize（不去冠词、不去标点以外的处理）。
- 汇报方式：`main()` 按 category 分组打印 `statistics.mean(f1_scores)`，即**逐 category 平均，没有总平均、没有加权**。
- **Category 5 (adversarial) 参与评测**：`main_loco_parse.py` 第 254–255 行，`answer` 字段为空时用 `adversarial_answer` 当参考答案算 F1：

```python
if(original_answer == ""):
    original_answer = qa.get("adversarial_answer", "")
```

  也就是说 category 5 的"正确答案"是像 "self-care is important" 这样的对抗性错误答案，模型答案与它做词重叠——这与 LoCoMo 官方（要求模型识别不可回答并输出 "No information available" 类回答再由 judge 判断）的做法完全不同。
- **BLEU-1 缺失**：README 与论文宣称 "boosting F1 scores by 49.11% and BLEU-1 by 46.18% on the LoCoMo benchmark"，但仓库里没有任何 BLEU 计算代码。BLEU-1 数字无法用仓库代码复现，推测是作者本地另有脚本（或 LoCoMo 官方 eval 代码）算的。
- LLM-as-judge 分数（论文若有）同样无法从本仓库复现。

## 评测范围

- **Benchmark**：只有 LoCoMo（`locomo10.json`，10 个 sample 全跑，无抽样、无切片）。QA 分布：category 1=282, 2=321, 3=96, 4=841, 5=446，共 1986 条。
- **没有 LongMemEval**、没有 MSC、没有其他 benchmark。
- **Baseline**：仓库内没有任何 baseline 实现或对比脚本。README Todo 写着 "Integrated Benchmarks: Standardized benchmark suite with a cross-model comparison for Mem0, Zep, and OpenAI" 仍是 **Ongoing**。论文（仓库内附 `Paper-MemoryOS.pdf`）中的 baseline 对比数字来源无法从仓库代码验证。
- **回答模型**：`gpt-4o-mini`（记忆构建各环节的 LLM 调用也都是 gpt-4o-mini）。
- 检索配置：`queue_capacity=10`（检索队列 10 条 page），`segment/page/knowledge_threshold=0.1`，短期记忆 `max_capacity=1`（每对 QA 立即下沉到中期记忆），中期 `max_capacity=2000`，热度阈值 `H_THRESHOLD=5.0`。
