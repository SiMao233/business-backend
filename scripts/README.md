# scripts/ —— RAG 运维与评测脚本

共四个脚本：前两个用于排查与修复 RAG 索引（MySQL ↔ Qdrant）一致性，后两个用于 Query Rewrite
真实 A/B 与 RAG 评测对比。除 `rag_reindex.py` 需显式加 `--yes` 才写数据外，其余均**只读**。

⚠️ **必须在项目根目录执行**：`.env` 与 `storage/` 都是相对路径（脚本内部已 `chdir` 到项目根，
但请仍从根目录调用，避免 `.env` 被漏读）。

## 1. `rag_index_audit.py` —— 索引对账（只读）

```bash
.venv/bin/python scripts/rag_index_audit.py            # 全部知识库
.venv/bin/python scripts/rag_index_audit.py --kb <id>  # 单个知识库
```

输出三类结论：

| 结论 | 含义 | 处理 |
|---|---|---|
| **真孤儿** | Qdrant 有点、MySQL 无对应切块 | 多为文档删除后向量未清掉 → 需清理 |
| **缺向量** | MySQL 有切块、Qdrant 无对应点 | 该切块检索不到 → 用 `rag_reindex.py --missing-vectors` 重索引 |
| **旧方案切块** | `chunk.id != vector_id` | 不影响对账，可选归一化 → `rag_reindex.py --stale-ids` |

退出码：`0` = 一致；`1` = 存在真孤儿或缺失向量（可接入巡检 / CI）。

### ⚠️ 关联键必须是 `vector_id`

**不要用 `chunk.id` 对账。** 切块与 Qdrant 点的关联键是 `sys_knowledge_chunk.vector_id`
（即 Qdrant 点 ID），不是 MySQL 主键：

- **旧方案**（确定性 ID `chunk_point_id` 之前索引的文档）：`chunk.id` 与点 ID 是**两个不同的
  uuid4**，靠 `vector_id` 列关联 —— 数据本身是自洽的；
- **新方案**（`chunk_point_id` 确定性 uuid5）：`chunk.id == vector_id`。

用 `chunk.id` 对账会把正常数据**误报**成「孤儿向量 + 缺向量」，得出方向完全相反的结论
（这个坑实际踩过一次）。

## 2. `rag_reindex.py` —— 重索引文档（默认 dry-run）

```bash
.venv/bin/python scripts/rag_reindex.py --stale-ids                    # 列出旧方案文档
.venv/bin/python scripts/rag_reindex.py --stale-ids --yes              # 执行
.venv/bin/python scripts/rag_reindex.py --missing-vectors --yes        # 修复缺向量的文档
.venv/bin/python scripts/rag_reindex.py --doc <id> --doc <id> --yes    # 指定文档
```

- 不加 `--yes` 只列出清单，**不动数据**；
- 执行时调用 `process_document_task()` —— ARQ worker 用的**同一个函数**，行为与线上一致
  （CAS 抢占、失败标 `failed`、确定性 ID 幂等、先清后写）；
- 三阶段设计保证「阶段 2（解析/切分/embedding）失败时旧索引完好」，最坏只是文档被标 `failed`；
- ⚠️ 会用**当前切分器**重算：切块数量与内容可能与旧版不同（预期行为，通常更好）；
- ⚠️ 会改变 `chunk.id` → 历史消息 `sources` 快照里的 `chunk_id` 会失效
  （该字段仅用于展示；来源卡片的预览/下载靠 `document_id`/`file_id`，不受影响）。

## 推荐工作流

```bash
# 1. 先对账，确认问题类型
.venv/bin/python scripts/rag_index_audit.py

# 2. 按需修复（先 dry-run 看清清单）
.venv/bin/python scripts/rag_reindex.py --missing-vectors
.venv/bin/python scripts/rag_reindex.py --missing-vectors --yes

# 3. 复核对账，直到退出码为 0
.venv/bin/python scripts/rag_index_audit.py
```

## 3. `rag_query_rewrite_ab.py` —— Query Rewrite 真实 A/B（只读）

```bash
.venv/bin/python scripts/rag_query_rewrite_ab.py --kb <kb_id>
.venv/bin/python scripts/rag_query_rewrite_ab.py --kb <kb_id> --scenario 1
```

调整 RAG 参数（改写开关 / 超时、`rag_top_k`、Reranker 开关…）后，验证「多轮指代问题」
的改写是否**真的**改善了检索，以及各降级路径是否正常：

- **PART 1/2** 多轮场景：真实改写 → 原 query 与改写 query 的检索对比，
  含**重复条目检查**（验证融合键口径：两路命中同一 chunk 必须合并成一条）；
- **PART 3** 短路对照：无上下文 / 自足问题 / 开关关闭 → 必须**零 LLM 调用**；
- **PART 4** 降级演练：开关关闭 / 极短超时（验证未触发 SDK 重试）/ 网关不可达。

⚠️ 会真实调用模型网关（产生少量 token 费用）与远程 Qdrant；**全程只读**。
内置场景需要知识库里确实有对应主题的文档，不匹配的场景会被自动跳过（不会误报）。

退出码：`0` = 全部通过；`1` = 有断言失败。

## 4. `rag_eval.py` —— RAG 评测与基线对比（只读）

改动 Chunking / Embedding / Retriever / Reranker / Query Rewrite 之后，用固定测试集（Golden Dataset）
判断「变好了还是变差了」。**数据集格式、指标定义与局限见 `tests/rag/README.md`。**

```bash
.venv/bin/python scripts/rag_eval.py                                  # 检索层评测（免 token，约 20s）
.venv/bin/python scripts/rag_eval.py --tag "chunk=1200/200"           # 给本次运行打标签
.venv/bin/python scripts/rag_eval.py --baseline latest                # 与上一次运行对比
.venv/bin/python scripts/rag_eval.py --baseline default --fail-on-regression   # 有回退则退出码 1
.venv/bin/python scripts/rag_eval.py --save-baseline default          # 认可当前结果 → 固化基线
.venv/bin/python scripts/rag_eval.py --answers --limit 10             # 答案层（真实调用模型）
.venv/bin/python scripts/rag_eval.py --answers --review-md evals/runs/review.md
.venv/bin/python scripts/rag_eval.py --review-load evals/runs/review.md
```

| 参数 | 说明 |
| --- | --- |
| `--dataset PATH` | Golden Dataset 路径（默认 `tests/rag/golden/rag_golden.toml`） |
| `--kb ID` | 覆盖数据集里的知识库（用于试跑新语料） |
| `--k 1,3,5,10` | Recall@K 的 K 列表 |
| `--top-k N` | 评测时临时使用的 top_k（默认 10：一次检索算出多个 K，排序与线上一致） |
| `--answers` | 跑答案层（**真实调用模型，花 token**） |
| `--limit N` / `--category X` | 快速子集调试（分类可重复指定） |
| `--tag TEXT` | 本次运行标签（写进报告 meta，便于对比时辨认） |
| `--out PATH` | 报告路径（默认 `evals/runs/eval-<时间戳>.json`，已 gitignore） |
| `--baseline latest\|名称\|路径` | 与基线对比（`名称` → `evals/baselines/<名称>.json`） |
| `--save-baseline NAME` | 固化基线（**该文件要提交**，这样跨会话/跨机器可比） |
| `--review-md PATH` | 生成人工复核表（填完用 `--review-load` 汇总） |
| `--review-load PATH` | 回读复核表并汇总通过率（未填题不计入分母） |
| `--fail-on-regression` | 存在指标回退时退出码 1（便于以后接 CI） |

输出：指标摘要（整体 + 分类 + 无答案题观测 + 原始问题对照）、阈值门禁结论，以及（带 `--baseline` 时）
逐指标 delta 表与逐题改善/回归清单。

退出码：`0` 正常；`1` 阈值未达标或存在回退；`2` 环境/参数错误（打印可读原因与排查命令）。
⚠️ **全程只读**：不写数据库、不改索引；答案层也不落会话/消息/用量记录。
⚠️ 单次报告约 120KB（含逐题 top-10 命中明细），因此 `evals/runs/` 不入库。

## 已知无害噪音

脚本结束时可能出现 `aiomysql` 的 `Exception ignored in: Connection.__del__ ...
RuntimeError: Event loop is closed`，以及 Qdrant 的
`UserWarning: Api key is used with an insecure connection`（远程 Qdrant 未启用 TLS）——
均为退出阶段的清理噪音，不影响执行结果。
