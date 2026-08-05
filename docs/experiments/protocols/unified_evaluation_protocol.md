# 统一评测协议（Unified Evaluation Protocol）

> NativeMem 论文项目评测协议。基于四份调研（LoCoMo / LongMemEval / BEAM / 各论文评测代码）整合而成。
> 本文档自包含：每个指标给出计算公式与代码出处，可直接照此执行。
> 日期：2026-07-02

---

## 1. Benchmark 数据总览

| 项 | LoCoMo | LongMemEval (oracle / S) | BEAM |
|---|---|---|---|
| 论文 | ACL 2024, Maharana et al. (2024.acl-long.747) | ICLR 2025, arXiv 2410.10813 | ICLR 2026, arXiv 2510.27246 |
| 本地数据 | `code/benchmarks/locomo/data/locomo10.json`（2.8 MB） | `code/benchmarks/longmemeval/data/longmemeval_oracle.json`（15 MB）；`longmemeval_s_cleaned.json`（277 MB） | HF 缓存于 `code/benchmarks/beam/hf_cache/`（100K/500K/1M 三档 arrow） |
| 会话数 | 10 conversations（conv-26/30/41-44/47-50） | 500 题各自带 haystack（oracle 1-6 session/题；S 38-62 session/题） | 100 conversations（100K:20 / 500K:35 / 1M:35 / 10M:10，**4 个 bucket 合计 100**） |
| session/turn | 272 sessions，5882 turns；每 conv 8k-16k 词（均值 13,377） | oracle 共 948 sessions，10,960 turns（每 session 2-32 turn，mean 11.56） | 每会话消息数：100K≈288 / 500K≈1088 / 1M≈2134 / 10M≈20,870（论文 Table 3） |
| 上下文规模 | 每 conv ~13k 词 | oracle mean 5,639 tokens；S mean ≈103k tokens（cl100k 实测） | 100K≈130k tokens/会话 … 10M≈10M tokens/会话 |
| 题数 | 1986 QA（cat1-4 共 1540 + cat5 对抗 446） | 500 题（含 30 个 `_abs` abstention 题，混在 4 个 type 里） | 2,000 题（100 会话 × 20 题） |
| Category | 5 类（int 编码）：1=multi-hop(282), 2=temporal(321), 3=open-domain(96), 4=single-hop(841), 5=adversarial(446) | 6 类：temporal-reasoning(133), multi-session(133), knowledge-update(78), single-session-user(70), single-session-assistant(56), single-session-preference(30) | 10 类能力 × 各 2 题/会话：abstention, contradiction_resolution, event_ordering, information_extraction, instruction_following, knowledge_update, multi_session_reasoning, preference_following, summarization, temporal_reasoning |
| 答案形式 | `answer` 短语（cat5 只有 `adversarial_answer`；cat3 含 "结论; 解释" 需取分号前段；多答案逗号分隔） | `answer` 短语/整数（preference 类是 rubric；abs 题 answer 是解释文本） | `ideal_response` + `rubric`（nugget 列表，mean 3.1 个/题） |
| evidence | `evidence` = dia_id 列表（`"D<session>:<turn>"`） | `answer_session_ids` + turn 级 `has_answer` 标记 | `plan_reference` |
| 时间戳 | session 级：`"1:56 pm on 8 May, 2023"`（格式 `%I:%M %p on %d %B, %Y`） | session 级：`"2023/04/10 (Mon) 23:07"`；另有 `question_date` | turn 级 `time_anchor`：`"March-15-2024"` |
| 官方主指标 | token-F1（stem+normalize）；cat5 关键词匹配 | LLM judge (gpt-4o-2024-08-06) 二元 accuracy，分类模板 | 逐 rubric-nugget LLM judge（gpt-4.1-mini）0/1 分（0.5 被 int 截断），event_ordering 用 tau-b×F1 |

### 结构速查

```
LoCoMo item: {sample_id, conversation:{speaker_a, speaker_b, session_N:[{speaker,dia_id,text,(blip_caption,img_url,query)}], session_N_date_time}, qa:[{question, answer|adversarial_answer, evidence, category}], observation, session_summary, event_summary}
LongMemEval item: {question_id(_abs?), question_type, question, answer, question_date, haystack_dates, haystack_session_ids, haystack_sessions:[[{role,content,has_answer}]], answer_session_ids}
BEAM item: {conversation_id, conversation_seed, narratives, user_profile, conversation_plan, chat(2D: session→turns, turn 含 role/content/time_anchor/index), probing_questions:{类别名→[题]}}
BEAM 题: {question, rubric:[nugget], difficulty, 标准答案字段随类别而异} —— 十类各 2 题；标准答案在 answer / ideal_answer(矛盾) / ideal_response(弃答) / ideal_summary(总结) / expected_compliance(指令、偏好遵循) 之一
```
注意：BEAM 的 `probing_questions` 在 HF 里是 Python repr 字符串，要 `ast.literal_eval`，得到的是**按类别分组的字典**而非题目列表（`code/scripts/runners/beam/convert.py`）。

BEAM 官方口径与我们的差异：官方把标准答案拆成 nugget，逐条判 0 / 0.5 / 1 再平均，顺序题用 Kendall tau-b 计部分正确；我们按每题非对即错。同一份答案在官方口径下分数更高，两者不可直接比较。

---

## 2. LoCoMo 评估方式详解

### 2.1 官方 F1（snap-research/locomo，`task_eval/evaluation.py`）

**normalize_answer(s)**（evaluation.py:75-92），顺序执行：
1. `s.replace(',', "")`（删逗号）
2. lower
3. 去 `string.punctuation`
4. 正则删 `\b(a|an|the|and)\b`（**比 SQuAD 标准多一个 and**）
5. 空白归一

**f1_score(pred, gt)**（evaluation.py:126-138）：token 级 F1，normalize 后每 token 过 `nltk.stem.PorterStemmer`，多重集交集：
```
same = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
P = same/len(pred_tokens); R = same/len(gt_tokens); F1 = 2PR/(P+R)   # 无重叠 → 0
```

**f1(pred, gt)**（evaluation.py:141-145，multi-answer 版）：先按逗号 split（在 normalize 之前，因 normalize 会删逗号），再：
```
F1 = mean_over_gt_subanswers( max_over_pred_subanswers( f1_score(p, g) ) )
```

**每 category 分支**（`eval_question_answering`，evaluation.py:189-241）：

| category | 处理 |
|---|---|
| 1 multi-hop | multi-answer `f1`（逗号 split 后 max-mean） |
| 2 temporal / 4 single-hop | 单答案 `f1_score` |
| 3 open-domain | 先 `answer = answer.split(';')[0].strip()`（去解释），再 `f1_score` |
| 5 adversarial | **关键词匹配**：输出含 `'no information available'` 或 `'not mentioned'`（lower 子串）→ 1，否则 0 |

汇总（`evaluation_stats.py:31-108`）：按 category 累加 per-question 分数取均值；**overall = 全部 1986 题均值，cat5 计入**——这解释了 Llama-3-70B overall 30.1 虚高（adversarial 80.0）。官方无 BLEU、无 LLM-judge；rouge/bertscore/EM 函数存在但主管线不用。

**官方论文 Table 2 关键数字（F1，overall）**：Human 87.9；gpt-4-turbo(128K) 51.6；claude-3-sonnet 42.8；gemini-1.0-pro 39.1；gpt-3.5-turbo(16K) 35.9；Mistral-7B 18.7。RAG（Table 3，gpt-3.5-turbo + DRAGON 检索）：无 RAG 22.4 → Observation top-5 最优 43.3。

### 2.2 Mem0 管线（`mem0-benchmarks/benchmarks/locomo/`）

流程（run.py）：
1. **Ingest**：session 按日期排序；`CHUNK_SIZE=1`（每 turn 一次 `mem0.add`，带 session epoch 时间戳）；speaker_a→user、speaker_b→assistant，content=`"{speaker}: {text}"`；图片 turn 拼 `[Sharing image - query: {query}. The image shows: {blip_caption}]`。
2. **Search**：每题 `mem0.search(question, user_id, top_k=200)`。
3. **Answer + Judge per cutoff**：`--top-k-cutoffs "10,20,50,200"`，每个 cutoff 取 score 最高前 c 条**单独生成答案并单独 judge**；答案从 `rsplit("ANSWER:", 1)[-1]` 提取。

Answerer prompt（prompts.py:40-98）：7 步推理（扫全记忆→实体归属→跨记忆合并→选最具体→时间锚定 reference_date=最后 session 日期→包含性检查→**强制作答，禁止说 not mentioned**）；记忆按 `created_at` 旧→新重排、带人类可读日期前缀、**不显示 rank/score**；answerer/judge 默认 gpt-5。

Judge（prompts.py:203-312）：JSON `{reasoning, label}`，label∈{CORRECT, WRONG} → 1/0。宽容规则：gold 列表命中≥1 项即对；同义/同极性情绪词算对；多余细节不扣分；日期差 14 天内算对、时长误差 50% 内算对；同指称即对。仅 "ZERO correct items" 或完全跑题才 WRONG。cat3 同官方取分号前段（`preprocess_answer`）。所有类别同一 prompt。

**范围**：`CATEGORIES_TO_EVALUATE = [1,2,3,4]`，**排除 cat5（446 题），只评 1540 题**（cat5 无 `answer` 键，代码上也会 KeyError）。

指标（run.py:622-656）：每 cutoff 报 overall 与 by-category 的 `accuracy = correct/total×100`（judge CORRECT 比例，即 J-score）。无 F1/BLEU。

### 2.3 官方 vs Mem0 差异表

| 维度 | 官方 | Mem0 |
|---|---|---|
| 指标 | token-F1（stem+normalize）+ cat5 关键词 | LLM judge 二元 J-score |
| 题目范围 | 1986 题（含 cat5） | 1540 题（cat1-4） |
| cat3 | 取 `;` 前段 | 同 |
| 检索评估 | RAG recall@k（evidence dia_id 命中） | cutoff 消融（top_10/20/50/200） |

### 2.4 各论文 quirk（详见 §5）
- cat5 三种互斥做法并存：A-Mem 参考答案=**adversarial_answer**（二选一 prompt，选中对抗答案得分）；SimpleMem 参考答案=**"Not mentioned in the conversation"**；LightMem/Mem0 直接**跳过 cat5**。三者的 cat5/overall 数字互不可比。
- A-Mem/Nemori/SimpleMem 的 F1 是同一份 **set-based（去重）token F1**（`simple_tokenize`：lower + 只替换 `.,!?`），与官方 SQuAD 式多重集 F1 不同，数字系统性不可比。

---

## 3. LongMemEval 评估方式详解

### 3.1 官方 judge（`src/evaluation/evaluate_qa.py`）

- **规范 judge 模型：`gpt-4o-2024-08-06`**（`print_qa_metrics.py` L20 有 assert）；调用参数 `n=1, temperature=0, max_tokens=10`。论文声称与人工专家一致率 >97%。
- **5 个分类 prompt 模板**（`get_anscheck_prompt` L24-43）：

| 触发 | 规则要点 |
|---|---|
| single-session-user / single-session-assistant / multi-session（共用） | 含正确答案或等价/含全部中间步骤→yes；**只含答案要求信息的子集→no** |
| temporal-reasoning | 同上 + **明确允许天数 off-by-one**（答 19 天、真值 18 天算对） |
| knowledge-update | 旧信息与更新值共存也算对，只要更新后的值在 |
| single-session-preference | answer 是 rubric；不必覆盖全部要点，正确回忆并利用了用户个人信息即对 |
| abstention（`question_id` 以 `_abs` 结尾，覆盖任意 type） | 正确识别不可回答即 yes（说信息不完整、或说有别的信息但问的没有，都算） |

统一结尾 "Is the model response correct? Answer yes or no only."
- **yes/no 解析**（L112-113）：`'yes' in response.lower()`——全文子串匹配，默认 False。

### 3.2 abstention 处理
- 30 个 `_abs` 题**没有独立 question_type**，混在 temporal-reasoning(6)/multi-session(12)/knowledge-update(6)/single-session-user(6) 里。
- 指标汇总（`print_qa_metrics.py`）：逐类 accuracy（abs 题计入所属类）+ `Task-averaged Accuracy`（6 类宏平均）+ `Overall Accuracy`（500 题微平均）+ `Abstention Accuracy`（30 题单独再算）。
- 检索指标（`print_retrieval_metrics.py`）：session/turn 级 `recall_all@{5,10}`、`ndcg_any@{5,10}`；**检索评估剔除 30 个 abs 题**（无 ground-truth 位置）。

### 3.3 三种 setting 与论文关键数字
- **oracle**：只含证据 session（~5.6k tokens/题）；**S**：~115k tokens（~40-50 session）；**M**：~500 session（~1.5M tokens）。
- GPT-4o：oracle 0.870 → S 0.606（降 30.3%）；Llama-3.1-8B 0.744 → 0.334（论文 Fig 4b）。

### 3.4 Mem0 变体差异（`mem0-benchmarks/benchmarks/longmemeval/`）
- **单一统一 judge prompt 取代官方 5 模板**（prompts.py L265-342，完全忽略 question_type）；显式放宽偏置（"When in doubt, lean toward 'yes'"）；off-by-one 泛化到所有题型；superset 判对、模糊单位换算、"0"↔"not enough information" 双向互判对。
- abstention 规则内嵌统一 prompt："response 拒绝作答即匹配 abstention gold，period"；但汇总**不单独输出 abstention accuracy**。
- yes/no 解析比官方严谨（`</judge_thinking>` 后倒序找整行 yes/no）。
- 默认 judge/answerer 均 gpt-5（官方 judge 是 gpt-4o-2024-08-06）；数据用 S cleaned；CHUNK_SIZE=2（user+assistant pair），top_k=200，cutoff 10/20/50/200。
- **答案生成 prompt 严重针对数据集调优**（"chandelier counts as jewelry" 等 hack）→ **Mem0 的 LongMemEval 数字与官方协议不可直接比**。

---

## 4. BEAM 评估方式详解

### 4.1 数据规模

| split | 会话数 | 题数 | 内存展开 | 每会话消息 |
|---|---|---|---|---|
| 100K | 20 | 400 | 14.1 MB | ~288 |
| 500K | 35 | 700 | 86.1 MB | ~1088 |
| 1M | 35 | 700 | 172.6 MB | ~2134 |
| 10M | 10 | 200 | 975.4 MB | ~20,870 |

HF 公开不 gated；19 个 domain；rubric nugget mean 3.1 个/题。

### 4.2 指标

**官方**（GitHub `mohammadtavakoli78/BEAM`，`src/evaluation/compute_metrics.py`）：
- Judge：**gpt-4.1-mini, temperature=0**。
- 9 种能力：逐 rubric nugget 调 unified judge prompt，JSON `{"score": 0/0.5/1}`；**官方代码 `int(score)` 把 0.5 截成 0**（只有 event_ordering 用 float）。
- event_ordering：LLM/MiniLM 对齐 → P/R/F1 → scipy `kendalltau(variant="b")` → `final_score = (tau+1)/2 × F1`（漏事件通过 F1 惩罚）。
- 论文 Table 1（0-1 均分）：最好成绩是 Llama-4-Maverick + LIGHT，100K/500K/1M/10M = 0.358/0.359/0.336/0.266；所有方法 contradiction_resolution ≈ 0。

**Mem0 复刻**（`mem0-benchmarks/benchmarks/beam/`）：
- 逐 nugget judge（gpt-5），分数 clamp（≥0.75→1.0, ≥0.25→0.5, else 0）→ **保留 0.5，比官方系统性偏松**；题分 = nugget 均分；`accuracy = 题分≥0.5 的题占比`（pass rate）+ `avg_score`。
- Kendall tau-b：手写实现（`benchmarks/common/metrics.py` L146-205），**只在交集上算、漏事件不惩罚**，且 **tau 结果只写 JSON 不进指标**——Mem0 报的 event_ordering 数字实际是 nugget judge 分。
- Mem0 公开结果（answerer/judge=gpt-5, top_200）：1M 全量 700 题 pass rate 70.1% / avg_score 0.641；10M 200 题 50.5% / 0.486。**与论文 Table 1 四重不可比**：judge 不同、0.5 处理不同、event_ordering 口径不同、answerer 不同。

### 4.3 可行性评估（我们能不能跑）

- **可跑**。数据公开；评估侧（answer+judge）与记忆系统解耦，只需替换 `mem0.add`/`mem0.search` 两个接口（`benchmarks/common/mem0_client.py`）即可复用整套 runner。
- 成本（CHUNK_SIZE=2，每 add ≈ 2 次 LLM 调用）：100K bucket ~5.8K 调用/~2M token；1M bucket ~75K 调用/~35M token；10M bucket ~209K 调用/~100M token，且 ingest 是顺序执行（10M 单会话按 2 add/s 也要 ~3 小时）。
- **推荐口径**：冒烟 `--chat-sizes 100K --conversations 0-9`（200 题，~1M token ingest）；正式对标用 **1M bucket 全量 700 题**（Mem0 公开基线口径）。10M 除非主打长度声明否则性价比低。
- 复现时**必须写明三项**：judge 模型、0.5 分处理（保留 vs int 截断）、event_ordering 口径（nugget 分 vs tau×F1），否则与论文/Mem0 都对不上。

---

## 5. 各论文评测代码对照

### 5.1 指标实现

| 系统 | F1 实现 | judge | 其他指标 |
|---|---|---|---|
| A-Mem（本地 `baselines/A-Mem/utils.py`） | **set-based token F1**：`simple_tokenize`=lower+替换`.,!?`+split，集合交集算 P/R（非 SQuAD Counter，无冠词归一化） | **无 LLM judge** | ROUGE-1/2/L(use_stemmer)、BLEU-1..4(nltk+smoothing1)、BERTScore(roberta-large)、METEOR、SBERT(all-MiniLM-L6-v2) 共 12 指标 |
| Nemori | 与 A-Mem `utils.py` **逐字相同**的 set-based F1 | Mem0 版 `ACCURACY_PROMPT`（CORRECT/WRONG, "be generous"），默认 gpt-4.1-mini, t=0 | BLEU-1；groupby(category).mean |
| LightMem | **无开源 F1 代码**（README 表的 F1/BLEU-1 无对应实现） | 同款 Mem0 `ACCURACY_PROMPT`，gpt-4o-mini, t=0；**明确跳过 cat5** | — |
| Memory-R1 | **代码未开源**；论文称 token-level F1 | 细节在附录，正文未给模型名 | BLEU-1 |
| SimpleMem | `test_ref/utils.py` 与 A-Mem **逐字节相同**（diff 确认） | 自定义 "Relevance & Accuracy Evaluator" 1.0/0.0，宽松条款（±1-2 天、子集可接受），gpt-4.1-mini, **t=0.3** | 同 A-Mem 12 指标 |

### 5.2 Answer prompt 对比（核心：全部要求简短回答）

| 系统 | 要求简短？ | 原文关键句 |
|---|---|---|
| Mem0 | 是 | ANSWER_PROMPT 第 8 条 "**The answer should be less than 5-6 words.**" |
| A-Mem | 是 | "write an answer in the form of a **short phrase**... Answer with exact words from the context whenever possible... **Short answer:**"；cat2 另加 "generate the **shortest possible answer**... avoid using any subjects" |
| Nemori | 是 | Mem0 原版 prompt，含 "less than 5-6 words" |
| LightMem | 是 | 同一 Mem0 prompt（4 个变体全带此句） |
| Memory-R1 | 是（论文） | "The answer should be less than 5-6 words."（adapted from Mem0） |
| SimpleMem | 是 | "provide a **very CONCISE answer (short phrase** about core information)"，JSON 输出 |

**结论**：F1 之间的可比性完全依赖 answer prompt 同为短答；五家全部约束简短输出，其中三家直接沿用 Mem0 原句。

### 5.3 cat5（adversarial）三种做法

| 做法 | 系统 | 说明 |
|---|---|---|
| 参考=adversarial_answer | A-Mem | 二选一 prompt（'Not mentioned...' vs adversarial_answer 随机序），**选中对抗答案反而得分** |
| 参考="Not mentioned in the conversation" | SimpleMem | 同样二选一，但 gold 固定为 Not mentioned |
| 跳过 cat5 | LightMem、Mem0 | judge 循环里 `if category==5: continue` |

### 5.4 已报数字备查
- Nemori v1: judge 0.744 / F1 49.5 / BLEU 38.5（gpt-4o-mini judge）；v4: gpt-4.1-mini judge 80.8 / F1 52.1。（网传 0.781 在 v1/v4/README 均不可考，勿引用。）
- LightMem(512,0.7): F1 47.75 / BLEU-1 36.16 / ACC 71.95-73.90。
- Memory-R1(GRPO): F1 45.0 / BLEU-1 37.5 / Judge 62.7；其 Mem0 baseline 30.4/22.2/45.7。
- SimpleMem: Avg F1 43.24（GPT-4.1-mini），vs Mem0 34.20、full-context 18.70。
- **注意**：以上 F1 全是 set-based 实现（或不可核），与 LoCoMo 官方 F1（gpt-4-turbo 51.6）**不同口径，不能放同一列**。

---

## 6. 我们的统一评测协议（建议）

### 6.1 Benchmark 与样本

| Benchmark | 范围 | 题数 | 理由 |
|---|---|---|---|
| **LoCoMo**（主） | 全 10 conv，**cat1-4 共 1540 题** | 1540 | 与 Mem0/LightMem 口径一致，baseline 数字最全 |
| **LongMemEval-S**（主） | `longmemeval_s_cleaned.json` 全 500 题（含 30 abs） | 500 | 官方标准 setting；oracle 只做上界 sanity check |
| **BEAM**（可选/扩展） | 冒烟：100K conv 0-9（200 题）；正式：1M 全量 700 题 | 200→700 | 对标 Mem0 公开基线口径；10M 不跑（ingest ~100M token 性价比低） |

开发迭代用小切片：LoCoMo 用已有 `create_locomo_split.py` 的子集；LongMemEval 用已有 `selected_30.json`（30 题索引）。**正式报数必须全量。**

### 6.2 构建 / 检索模型（2026-07-03 更新）
- **筛查轮：gpt-5.4-mini**（ChatGPT 订阅经 proxy，¥0）统一用于记忆构建与检索侧 LLM 调用（早期 run 混用过 deepseek-v4-flash，之后一律固定 5.4-mini）。
- **正式轮：gpt-4o-mini**（OpenRouter）——对齐文献 "backbone gpt-4o-mini" 口径，builder 与 answerer 必须同模型。
- 所有 baseline 的记忆构建 LLM 与我们同轮同模型，控制变量——否则比较的是构建模型不是记忆机制。
- 检索 top-k：LoCoMo/LongMemEval 报 **top_20 为主结果**，附 top_10/50 消融（沿用 Mem0 cutoff 机制，`benchmarks/common/utils.py:299-301`）；BEAM 沿用 top_200 + cutoff 100（Mem0 默认）。
- Embedding 检索器各系统用其默认配置，在论文中注明。

### 6.3 统一 answerer（必须统一）
- **需要统一**：§5.2 证明 F1/judge 可比性依赖 answer prompt；不统一则差异来自 prompt 而非记忆。
- **prompt 用 Mem0 原版 ANSWER_PROMPT**（Nemori/LightMem/Memory-R1 同款），关键句保留：
  - "The answer should be less than 5-6 words."
- **禁止使用** Mem0 benchmarks 里的 7 步长 prompt（LoCoMo prompts.py:40-98）和 LongMemEval 的数据集特调 prompt（含 "chandelier counts as jewelry" 等 hack）——这两个是 Mem0 刷榜用的，用了与所有论文都不可比。
- Answerer 模型：所有系统（包括 baseline）统一同一个——筛查轮 gpt-5.4-mini、正式轮 gpt-4o-mini，temperature=0。筛查轮绝对分属 4.1-mini 段位（Full-ctx 锚点 80.3），对文献 4o-mini 列看相对锚点差值。
- 记忆呈现：检索结果按 created_at 旧→新排序、带日期前缀、不显示 score（Mem0 做法，避免 anchoring）。
- BEAM 用 Mem0 的 BEAM answer prompt（`benchmarks/beam/prompts.py:104-158`，含 "矛盾取最新""不够信息就说 I don't have enough information..."），因为 BEAM 是 rubric 评分不是短答 F1。

### 6.4 Judge（2026-07-03 定稿：gpt-4o-mini）
- **模型：gpt-4o-mini via OpenRouter**（`llm_clients.py` 默认已改），temperature=0。成本可忽略（全量重判一个 s0 run ≈ $0.01，LoCoMo 全 1540 题 ≈ $0.1）。
- ~~GPT-5.5~~ 已弃用：对 paraphrase/相对日期答案系统性过严（LightMem s0 152 题重判翻正 20、反翻 3，overall 61.8→73.0），且 full-ctx 校准无法暴露此偏差（full-ctx 答案贴原文）。凡用 5.5 judge 的旧数字一律作废重判。
- **LoCoMo prompt：Mem0/Nemori/LightMem 同款 `ACCURACY_PROMPT`**（CORRECT/WRONG 二元，"be generous with your grading - as long as it touches on the same topic as the gold answer"），JSON `{"label": "CORRECT"|"WRONG"}` → 1/0。代码来源：LightMem `experiments/locomo/llm_judge.py` 或 Nemori `evaluation/locomo/metrics/llm_judge.py`（逐字相同）。cat3 先取 `answer.split(';')[0]`。
- **LongMemEval：用官方 5 分类模板**（`src/evaluation/evaluate_qa.py` `get_anscheck_prompt`），不用 Mem0 的放宽版——官方模板与人工一致率 >97%，且 abstention/temporal 的规则差异有语义意义。yes/no 解析改用严格版（整行匹配，不用官方的全文子串 `'yes' in`）。
- **BEAM：官方 unified nugget judge prompt**（Mem0 `benchmarks/beam/prompts.py:170-233` 的移植版语义一致，可直接用），**保留 0.5 分**（写明与官方 int 截断不同）；event_ordering 主表报 nugget 分，附录报 tau×F1（官方口径，用 scipy kendalltau + F1 惩罚，不用 Mem0 的交集版）。
- Judge 一致性：抽 100 题人工核一遍 gpt-4o-mini judge 的一致率并在论文报告。

### 6.5 指标

| 指标 | 定义 | 代码来源 |
|---|---|---|
| **LLM Judge (0/1)** 主指标 | `J = (1/N) Σ 1[label==CORRECT]`，按 category 和 overall 报 | LightMem `llm_judge.py`（prompt+解析），模型 gpt-4o-mini |
| **F1** 辅助 | set-based token F1：`simple_tokenize`（lower + 替换 `.,!?` 为空格 + split 去重集合），`P=|pred∩gt|/|pred|, R=|pred∩gt|/|gt|, F1=2PR/(P+R)` | A-Mem `utils.py:34-38,136-145`（与 Nemori/SimpleMem 逐字相同）——选它因为所有可对比论文都是这份实现；**表格脚注注明非 LoCoMo 官方 F1** |
| **BLEU-1** 辅助 | `sentence_bleu(weights=(1,0,0,0), SmoothingFunction().method1, nltk.word_tokenize(lower))` | A-Mem `utils.py:50-66` |
| LongMemEval 附加 | Task-averaged Acc（6 类宏平均）+ Overall Acc（微平均）+ Abstention Acc（30 题单独） | 官方 `print_qa_metrics.py` |
| BEAM | avg_score（题分=nugget 均分的均值）+ pass rate（题分≥0.5 占比），按 10 能力分组 | Mem0 `benchmarks/beam/run.py:810-863` |
| 检索质量（分析用） | LoCoMo: evidence dia_id recall；LongMemEval: session 级 recall@k（剔除 abs 题） | 官方 evaluation.py:228-235 / print_retrieval_metrics.py |

### 6.6 Adversarial（LoCoMo cat5）
- **主表不评 cat5**（1540 题口径）——与 Mem0、LightMem 一致，是当前多数论文做法，且避开 §5.3 三种互斥评法的混乱。
- 附录可选报一版 cat5：gold 固定为 abstention（SimpleMem 口径，"Not mentioned in the conversation"），judge 判是否正确拒答；**绝不用 A-Mem 的 adversarial_answer 口径**（奖励幻觉）。
- 表格与相关工作对比时**注明各家 cat5 口径**，不与含 cat5 的 overall（如 LoCoMo 官方 Table 2）直接比。

### 6.7 Baseline 列表

| Baseline | 来源 | 说明 |
|---|---|---|
| **Mem0** | 本地 `mem0-benchmarks/` | 主对标；ingest/search 接口即 `mem0.add/search`，替换构建模型跑 |
| **A-Mem** | 本地 `baselines/A-Mem/` | 用其记忆系统 + 我们统一的 answerer/judge（不用它的 test_advanced.py 评法） |
| **Nemori** | GitHub nemori-ai/nemori | `add.py → search.py` 复用，evals 换成统一协议 |
| **LightMem**（可选） | GitHub zjunlp/LightMem | 其 judge 与我们同款，主要换 judge 模型和 answerer |
| Full-context / RAG-BM25（下界参照） | 自建 | 无记忆系统的朴素对照 |

Memory-R1 不作可复现 baseline（代码未开源），只引用其论文数字并注明口径不可比。

### 6.8 执行清单
1. 固定随机种子；answerer/judge temperature=0；所有 LLM 调用记录 model id + 时间。
2. 每个系统每个 benchmark 落盘 per-question JSON（question_id, retrieved memories, answer, judge label/score），便于误差分析与 rejudge（复用 Mem0 `--predict-only` / `--evaluate-only --rejudge` 三段式）。
3. 论文实验设置一节必须写明：judge 模型与 prompt 来源、F1 实现口径（set-based，非官方）、cat5 处理、LongMemEval abs 题处理、BEAM 0.5 分与 event_ordering 口径、检索 top-k。
4. 报数格式：主表 = LLM Judge (0/1)；辅表 = F1 + BLEU-1；LoCoMo 分 4 类 + overall；LongMemEval 分 6 类 + task-avg + overall + abstention；BEAM 分 10 能力 + avg_score + pass rate。

### 6.9 关键代码路径
- LoCoMo 官方评估：`code/locomo/task_eval/{evaluation.py, evaluate_qa.py, evaluation_stats.py}`
- Mem0 管线：`code/mem0-benchmarks/benchmarks/{locomo,longmemeval,beam}/{run.py, prompts.py}`、`benchmarks/common/{mem0_client.py, metrics.py, llm_client.py, utils.py}`
- LongMemEval 官方：`code/longmemeval/src/evaluation/{evaluate_qa.py, print_qa_metrics.py, print_retrieval_metrics.py}`
- A-Mem 指标（统一 F1/BLEU 来源）：`code/baselines/A-Mem/utils.py`
- BEAM 官方评估副本：scratchpad `beam_compute_metrics.py` 等（正式使用时从 GitHub mohammadtavakoli78/BEAM 重新拉取）
- Builder/Answerer proxy（订阅模型）：`src/chatgpt_proxy.py`（gpt-5.4-mini/gpt-5.5, localhost:8199）；Judge 直连 OpenRouter（gpt-4o-mini），不走 proxy
