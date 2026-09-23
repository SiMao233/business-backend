# RAG 评测框架（Golden Dataset + 检索/答案指标 + 基线对比）

**用途**：改动 Chunking / Embedding / Retriever / Reranker / Query Rewrite 之后，
用同一套固定题目重新评测，回答一句话 —— **「这次改动是变好了还是变差了？」**

```bash
.venv/bin/pytest tests/rag/ -q                       # ① 数据集校验 + 检索层门禁（免 token，约 20s）
.venv/bin/python scripts/rag_eval.py --baseline latest   # ② 出报告 + 与上次对比（含逐题改善/回归）
RAG_EVAL_ANSWERS=1 .venv/bin/pytest tests/rag/ -q     # ③ 答案层评测（真实调用模型，花 token）
```

> ⚠️ 评测需要**真实环境**（MySQL + Qdrant + embedding；答案层另需模型网关）。
> 环境不可达时 pytest 用例会 **skip**（不是"通过"），CLI 会报可读错误并退出码 2。
> 离线机器上能跑的只有数据集校验与指标纯函数单测。

---

## 一、目录与产物

| 路径 | 说明 | 是否提交 |
| --- | --- | --- |
| `tests/rag/golden/rag_golden.toml` | **Golden Dataset**（40 题 / 2 个知识库） | ✅ 提交 |
| `tests/rag/test_golden_dataset.py` | 数据集结构校验（离线） | ✅ |
| `tests/rag/test_metrics_unit.py` | 指标/报告/复核表的纯函数单测（离线） | ✅ |
| `tests/rag/test_retrieval_eval.py` | 检索层评测 + 门禁（需环境） | ✅ |
| `tests/rag/test_answers_eval.py` | 答案层评测（需环境 + `RAG_EVAL_ANSWERS=1`） | ✅ |
| `tests/rag/conftest.py` | 环境探测、数据集夹具、报告缓存（会话内只跑一次） | ✅ |
| `evals/baselines/*.json` | **基线**（跨会话/跨机器对比用） | ✅ 提交 |
| `evals/runs/*.json` | 每次运行的完整报告（含逐题命中明细） | ❌ gitignore |

代码分工：`app/modules/ai/evaluation.py`（**纯函数**：数据集模型 / 指标 / 报告 / 对比 / 复核表）、
`app/modules/ai/eval_runner.py`（**唯一含 I/O**：开会话、调检索链路、可选调用模型）、
`scripts/rag_eval.py`（CLI）。**全程只读**：不写库、不改索引、不落会话/消息/用量。

---

## 二、Golden Dataset 格式与加题指引

TOML（标准库 `tomllib` 解析，**零新增依赖**）。每条题目：

```toml
[[datasets.questions]]
id = "fact-01"                        # 全局唯一
category = "simple_fact"              # 见下表七类
question = "员工手册规定的标准工作时间是几点到几点？"
expected_answer = "周一至周五上午九点到下午六点，午休一小时"
expected_answer_points = ["上午九点", "下午六点"]   # correctness 用的金标要点
expected_source = ["员工手册.txt"]                   # 期望来源**文档名**（稳定，不用 id）
expected_evidence = ["工作时间为周一至周五上午九点到下午六点"]  # 必须出现在命中片段里
expected_behavior = "answer"          # answer | abstain（无答案题）
history = []                          # 多轮/指代题必填：[[user, ...], [assistant, ...]]
notes = "出题依据（可追溯）"
```

**三条硬规矩**（否则数据集会静默失效）：

1. **`expected_evidence` 必须逐字来自原文档**（建议整句复制），且不跨 chunk 边界
   —— 判定按「内容包含」而非 chunk 主键，因此**重新切分后依然有效**（这是能评估 Chunking 改动的前提）；
2. **`expected_source` 用文档名**，不要用 `document_id`（后者重索引后会变）；
3. **无答案题不要填 evidence/source**：它们不参与 Recall/MRR，只判「是否拒答」。

**分类与题量**（`simple_fact` 8 / `keyword` 5 / `proper_noun` 6 / `multi_turn` 5 / `anaphora` 5 /
`cross_chunk` 5 / `no_answer` 6 = 40）：

| 分类 | 含义 | 评估目标 |
| --- | --- | --- |
| `simple_fact` | 单点事实 | 基本召回 |
| `keyword` | 错误码 / 标识符 / API 名 | 混合检索的**关键词路** |
| `proper_noun` | 专有名词 / 术语 | 语义召回 |
| `multi_turn` | 问题自足，但需历史消歧 | 多轮上下文 |
| `anaphora` | **纯指代**（"那这个怎么办？"） | **Query Rewrite**（改写前几乎不可能命中） |
| `cross_chunk` | 证据分散在 ≥2 个切块/文档 | 多命中能力、Chunking 粒度 |
| `no_answer` | 知识库里确实没有 | **拒答（ABSTAIN）** |

**加题流程**：① 从真实库取证据子串 → ② 写进 TOML → ③ `pytest tests/rag/test_golden_dataset.py -q`
（结构校验）→ ④ 跑一次完整评测，确认新题**真的能命中**（若恒不命中，说明证据子串或题目写法有问题）。
> 结构校验会拦下：重复 id、缺字段、分类非法、`multi_turn`/`anaphora` 缺 history、
> 无答案题误填 evidence、证据子串过短（<8 字符）或题内重复、跨 chunk 题证据不足两条、分类失衡。

---

## 三、指标定义与**局限**

### 检索层（macro：先每题算分再平均）

| 指标 | 定义 |
| --- | --- |
| `hit@K` | top-K 里是否命中**任一**证据（1/0） |
| `evidence_recall@K` | top-K 覆盖的**证据条目比例**（跨 chunk 题因此能看出"漏了几条"） |
| `source_recall@K` | top-K 覆盖的**期望文档比例** |
| `mrr` | 首个命中证据的排名倒数（无命中 = 0） |

另有**双口径**：`raw`（只用原始问题检索）与线上路径（按配置执行 Query Rewrite）。
两者差异只在有历史的题上出现 —— **这组对比就是 Query Rewrite 的回归门禁**。

### 答案层（`--answers` / `RAG_EVAL_ANSWERS=1`）

| 指标 | 定义 | 方向 |
| --- | --- | --- |
| `answer_correctness` | 金标要点在答案中的命中率 | 越高越好 |
| `points_supported_rate` | 答案命中的要点里，**同时出现在本轮 context** 的比例（疑似幻觉信号） | 越高越好 |
| `citation_valid_rate` | 无编造引用（`invalidCited` 为空）的题占比 | 越高越好 |
| `cited_present_rate` | 至少带一个 `[n]` 引用的回答占比 | 越高越好 |
| `abstain_accuracy` | **无答案题**通过率（宽松口径：ABSTAIN 或正文明确声明无法确认） | 越高越好 |
| `false_abstain_rate` | **有答案题**被误拒的比例（阈值过严的信号） | 越低越好 |
| `uncited_claims_total` | 无引用事实句总数 | 越低越好 |

### ⚠️ 局限（务必知情）

1. **都是确定性 proxy，不是语义级判断**。要点命中率靠子串匹配，判不了"答得对不对、全不全"——
   所以才有**人工复核表**兜底（`--review-md` 生成 → 人工填 → `--review-load` 汇总）。
2. **`points_supported_rate` 有定义域**：只有"答案命中了金标要点"时才有值；`None` 意味着
   **"答非所问 / 没答到要点"**，**不是**"没有幻觉"。
3. **答案层不设硬门禁**：大模型有随机性，把"必须答出某句话"写成断言会造出 flaky 用例。
   答案层只断言结构完整与链路正确（例如**拒答时不得调用模型**），质量判断交给人工 + 基线对比。
4. **不覆盖 Agent 工具模式**：答案层是自编排（检索 → 提示词 → 模型），不走 `create_agent`；
   与线上刻意差异：`temperature=0`、不复用 Agent 配置快照。
5. **环境依赖**：离线 skip ≠ 通过；要接 CI 必须先具备 MySQL + Qdrant + embedding。
6. **报告体积**：单次报告约 120KB（含逐题 top-10 命中明细）；基线文件因此较大。

---

## 四、门禁阈值与校准政策

阈值写在数据集的 `[thresholds]`（snake_case，便于手写；报告里是 `hit@5` 形式，由
`threshold_to_metric()` 映射）：

```toml
[thresholds]
hit_at_5 = 0.80            # 实测 0.882，留约 10% 余量
evidence_recall_at_5 = 0.78
mrr = 0.72
```

- 只要求**不低于门禁**，不要求指标必须提升（这是回归门禁，不是性能目标）；
- `hit@1`（实测 0.735）**不加入硬门禁**，但保留在报告与对比表里观察；
- ⚠️ **一旦 rerank / rewrite / embedding 模型 / 切分参数 / 数据集发生实质变化**，
  必须：重新实测（`--tag` 标注变化）→ 校准阈值 → 更新基线。否则门禁会误报（见下方实验数据）。

---

## 五、基线对比：怎么读 delta

```bash
.venv/bin/python scripts/rag_eval.py --tag "chunk=1000/200" --baseline default
.venv/bin/python scripts/rag_eval.py --save-baseline default        # 认可当前结果时固化基线
.venv/bin/python scripts/rag_eval.py --baseline latest --fail-on-regression   # 有回退则退出码 1
```

输出示例（节选真实运行）：

```
指标                  baseline  current  delta
hit@1               0.735     0.853    +0.118  ▲
mrr                 0.805     0.873    +0.067  ▲
回归题目(0):
改善题目(3): pn-03 hit@5 0→1, ...
```

读法：**先看整体指标方向，再看逐题清单**。逐题清单能直接告诉你"哪几道题变差了"，
配合 `evals/runs/*.json` 里的 `hits`（每题 top-10 的文档名与分数）就能定位"换回了哪个文档"。

---

## 六、真实基线（2026-09-21 首次测量）

默认配置：hybrid 开 / **rerank 关** / **rewrite 关**；面试八股文 68 切块 + 分块验证库 5 切块。

| 指标 | 值 |
| --- | --- |
| `hit@1 / @3 / @5 / @10` | 0.735 / 0.853 / **0.882** / 0.912 |
| `evidence_recall@1 / @3 / @5 / @10` | 0.691 / 0.824 / **0.868** / 0.897 |
| `mrr` | **0.805** |

分类 `hit@5`：`simple_fact` 1.000(8) · `keyword` 1.000(5) · `multi_turn` 1.000(5) ·
`cross_chunk` 1.000(5) · `proper_noun` 0.833(6) · **`anaphora` 0.400(5)** ← 唯一短板（改写关闭所致）。
无答案题 6 题：**零命中率 1.000**、低相关率 0.000。

答案层（40 题，`temperature=0`，约 2.9 分钟）：`answer_correctness` 0.838 ·
`points_supported_rate` 1.000 · `citation_valid_rate` 0.975 · `cited_present_rate` 0.775 ·
`abstain_accuracy` **1.000** · `false_abstain_rate` **0.000** · `uncited_claims_total` 14 · 平均延迟 4.04s。

---

## 七、框架有效性验证（两次真实对比实验）

### 实验 1：Reranker 开启 vs 关闭（基线 = 关闭）

| 指标 | off | on | delta |
| --- | --- | --- | --- |
| `hit@1` | 0.735 | 0.853 | **+0.118 ▲** |
| `evidence_recall@1` | 0.691 | 0.794 | +0.103 ▲ |
| `mrr` | 0.805 | 0.873 | **+0.067 ▲** |
| `hit@5` | 0.882 | 0.912 | +0.029 ▲ |
| `hit@10` | 0.912 | 0.912 | 0.000 = |

分类上 `proper_noun` 0.833 → **1.000**（此前那道"证据排在 6~10 名"的题被精排救了回来）；
`cross_chunk` 的 `evidence_recall@5` 0.900 → 0.800（略降，但 `mrr` 0.750 → 0.867 提升）。
**结论**：Reranker 让**排序更准**（hit@1 / mrr 明显提升），但不改变召回上限（hit@10 不变）——
与它的设计定位完全一致。

### 实验 2：Query Rewrite 开启 vs 关闭（基线 = 关闭）

| 指标 | off | on | delta |
| --- | --- | --- | --- |
| `anaphora hit@5` | 0.400 | **0.800** | **+0.400 ▲** |
| `anaphora mrr` | 0.300 | 0.700 | +0.400 ▲ |
| `hit@5` | 0.882 | 0.941 | +0.059 ▲ |
| `evidence_recall@5` | 0.868 | 0.926 | +0.059 ▲ |
| `mrr` | 0.805 | 0.864 | +0.059 ▲ |
| `source_recall@1` | 0.868 | 0.956 | +0.088 ▲ |

同时报告的"原始问题口径对照"显示：**原始问题 hit@5 0.882 vs 线上（改写后）0.941**
—— 这组差异正是 Query Rewrite 的贡献量。**无任何指标回退。**
（本次实验给足了超时：`RAG_QUERY_REWRITE_TIMEOUT=12.0`、`MAX_TOKENS=1024`；
用默认 6s 会偶发降级，那属于**配置问题**而不是框架问题。）

---

## 八、改 Chunking 的标准操作链

评测**只读、不负责改索引**，所以完整链条是：

```bash
# 1. 改切分参数（.env: CHUNK_SIZE / CHUNK_OVERLAP）或切分器代码
# 2. 重索引受影响文档（会重算切块，chunk.id 会变 —— 历史 sources 快照里的 chunk_id 失效，仅展示用）
.venv/bin/python scripts/rag_reindex.py --doc <document_id> --yes
# 3. 对账确认索引一致
.venv/bin/python scripts/rag_index_audit.py
# 4. 评测 + 与基线对比
.venv/bin/python scripts/rag_eval.py --tag "chunk=1200/200" --baseline default
```

> 切分参数属于"实质变化"：**先看 delta，再决定是否重新校准阈值并更新基线**。

---

## 九、已知无害噪音

- `aiomysql` 的 `Exception ignored in: Connection.__del__ ... RuntimeError: Event loop is closed`
  （进程退出阶段的清理噪音）；
- Qdrant 的 `UserWarning: Api key is used with an insecure connection`（远程 Qdrant 未启 TLS）；
- 远程 Qdrant 偶发 `ConnectTimeout`（瞬时网络抖动）→ CLI 会提示重试并给出排查命令，
  重跑即可；若频繁出现，先查网络/服务而不是怀疑代码。

---

## 十、明确不做的事（本阶段范围外）

- **LLM-as-judge**：本期不引入评测模型（答案层用确定性 proxy + 人工复核）；
- **CI 自动跑**：需要真实环境与 token，暂由人工触发；
- **人工评分的自动回归**：`--review-load` 只汇总通过率，不参与门禁；
- **多知识库矩阵评测**：一次跑一个数据集（可用 `--kb` 覆盖），多次运行即可横向比较；
- **Agent 工具模式评测**：见「局限 4」。
