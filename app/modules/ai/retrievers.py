"""AI 检索器：向量检索 / 关键词检索 / RRF 融合（Hybrid Search）。

一次检索的流程：

    query
     ├─ VectorRetriever    Qdrant 语义召回（过取 fetch_k，按知识库过滤）
     └─ KeywordRetriever   MySQL LIKE 精确召回（错误码 / 版本号 / API 名 / 专有名词）
            ↓
        rrf_fuse()  仅按**排名**融合，不需要两路分数可比
            ↓
        top_k（启用 Reranker 时为候选池 rag_candidate_k，否则为 rag_top_k）
            ↓
        精排 → 见 rerankers.py（Reranker 负责 precision，可关闭 / 可替换 / 失败降级）
            ↓
        由 retrieval.py 富化文档名 / 来源文件

本模块只负责**召回**（宁多勿漏）；「排序准」由 `rerankers.py` 承担。
两者都只接在 `retrieval.retrieve_hits()` 内部，Chat / SSE 层无感。

为什么不引入 BM25 / Elasticsearch：
- MySQL InnoDB FULLTEXT 的 relevance 是 TF-IDF 变体，不是 BM25（缺 k1/b 饱和项与字段
  长度归一化）。但 **RRF 只用排名、不用分数**，所以「关键词路给出合理排序」即已足够，
  没有为此新增搜索基础设施的必要；
- 目标场景（错误码、版本号、API 名、专有名词）本质是**精确子串**匹配，LIKE 与之天然对齐，
  ngram 分词反而会把标识符切成 bigram 变成近似匹配；
- 知识库边界（knowledge_base_id）在**两路都强制过滤**且保持一致，不存在越界检索。

扩展点：新增检索器只需实现 `Retriever` 协议并在 `build_retriever()` 注册；
`HybridRetriever` 天然支持多路融合。精排（Reranker）不在本模块，见 `rerankers.py`。
"""

import re
from typing import Any, Protocol

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings

# hit["retrievers"] 里的来源标记（仅供观测与测试断言，不会进入 AiSourceOut 出参）
VECTOR = "vector"
KEYWORD = "keyword"
HYBRID = "hybrid"

# ① 含分隔符的标识符：错误码 / 版本号 / 常量名 / 路径片段
#    例：ERR_4012 / v1.2.3 / get_user_info / user/list / HTTP-500
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[._\-/][A-Za-z0-9]+)+")
# ② 纯拉丁词（≥2 字符）：HTTP / OpenAI
_WORD = re.compile(r"[A-Za-z]{2,}")
# ③ 纯数字（≥2 位）：4012
_NUMBER = re.compile(r"\d{2,}")
# ④ CJK 连续片段（用二元滑窗近似切词，不引入分词依赖）
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def extract_keywords(text: str, max_terms: int | None = None) -> list[str]:
    """从查询中抽取关键词（纯函数，顺序即优先级）。

    优先级：标识符 → 纯拉丁词 → 纯数字 → CJK 二元滑窗。
    高信号 token 排在前面，超出上限时优先保留它们（标识符是最能提升精确召回的部分）。

    细节：拉丁词/数字若落在已识别的标识符区间内则跳过 —— 否则 `ERR_4012` 会额外产出
    `ERR`，而 `LIKE '%ERR%'` 会误匹配 `error` 之类的无关内容。
    """
    if not text:
        return []
    limit = get_settings().rag_keyword_max_terms if max_terms is None else max_terms
    if limit <= 0:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    occupied: list[tuple[int, int]] = []

    def add(term: str) -> None:
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            terms.append(term)

    def inside_identifier(pos: int) -> bool:
        return any(start <= pos < end for start, end in occupied)

    for match in _IDENTIFIER.finditer(text):
        add(match.group())
        occupied.append(match.span())
    for pattern in (_WORD, _NUMBER):
        for match in pattern.finditer(text):
            if not inside_identifier(match.start()):
                add(match.group())
    for run in _CJK_RUN.findall(text):
        for i in range(len(run) - 1):
            add(run[i : i + 2])

    return terms[:limit]


def rrf_fuse(
    ranked_lists: list[list[dict]],
    weights: list[float] | None = None,
    rrf_k: int | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion：按**排名**融合多路检索结果（纯函数）。

    score(d) = Σ_r w_r / (k + rank_r(d))，rank 从 1 起，k 默认 60（RRF 原论文取值）。

    同一 chunk 被多路命中时**只保留一条**且分数累加 —— 这正是 Hybrid 的核心收益：
    既被语义召回、又被关键词精确命中的片段会排到最前。
    合并时优先保留带 `score`（向量相似度）的那份，以维持 `sources[].score` 的既有语义。
    """
    k = get_settings().rag_rrf_k if rrf_k is None else rrf_k
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    merged: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for hits, weight in zip(ranked_lists, weights, strict=True):
        for rank, hit in enumerate(hits, start=1):
            key = hit.get("chunk_id")
            if not key:
                # 缺少 chunk_id 无法作为融合键（防御性跳过，正常数据不会出现）
                continue
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            current = merged.get(key)
            if current is None:
                merged[key] = dict(hit)
                continue
            # 重复命中：保留向量相似度 + 合并来源标记
            if current.get("score") is None and hit.get("score") is not None:
                current["score"] = hit["score"]
            sources = set(current.get("retrievers") or []) | set(hit.get("retrievers") or [])
            current["retrievers"] = sorted(sources)

    fused = sorted(merged.values(), key=lambda item: scores[item["chunk_id"]], reverse=True)
    limit = get_settings().rag_top_k if top_k is None else top_k
    return fused[:limit] if limit > 0 else fused


class Retriever(Protocol):
    """检索器协议：实现 `retrieve()` 即可接入 `HybridRetriever`。

    约定：
    - `kbs` 是已过滤（status==1）的知识库 ORM 列表，**必须**作为检索边界；
    - 返回的 hit dict 至少包含 chunk_id / document_id / knowledge_base_id / seq_no /
      content 与 `retrievers` 来源标记；`score` 为向量相似度或 None；
    - `optional=True` 表示该路失败时允许降级跳过（不阻断对话）。
    """

    name: str
    optional: bool

    async def retrieve(
        self, db: AsyncSession, kbs: list[Any], query: str
    ) -> list[dict]: ...


class VectorRetriever:
    """向量检索：Qdrant 语义召回（复用 vector.py，不直接依赖 qdrant-client）。"""

    name = VECTOR
    # 向量路是主召回：失败必须向上抛（保持既有失败语义），不做静默降级
    optional = False

    def __init__(self, fetch_k: int | None = None) -> None:
        self.fetch_k = fetch_k

    async def retrieve(self, db: AsyncSession, kbs: list[Any], query: str) -> list[dict]:
        if not kbs:
            return []
        # 延迟导入：避免与 vector / knowledge 模块产生循环依赖
        from app.modules.ai import vector

        limit = self.fetch_k if self.fetch_k is not None else get_settings().rag_vector_fetch_k
        embeddings = await vector.build_embeddings(db, kbs[0].embedding_model_id)
        query_vector = await embeddings.aembed_query(query)
        hits = await vector.search_chunks(
            query_vector, [kb.id.hex for kb in kbs], max(1, limit)
        )
        for hit in hits:
            hit["retrievers"] = [self.name]
        return hits


def fusion_key(chunk: Any) -> str:
    """取切块的**融合键**（纯函数）：统一用「Qdrant 点 ID」的 hex 形式。

    为什么不能用 `chunk.id`：`rrf_fuse` 靠 chunk_id 判断「两路命中的是不是同一个切块」，
    而两路的 chunk_id 来源不同 —— 向量路取 Qdrant `payload.chunk_id`（即点 ID），
    关键词路取 MySQL 列。在确定性 ID（`chunk_point_id`）之前的存量切块，
    `chunk.id`（MySQL 主键）与点 ID 是**两个不同的 uuid4**，用 `chunk.id` 当键会让同一个
    切块在两路里 key 不同 → RRF 合不掉 → `sources` 出现重复条目、白占 `rag_top_k` 名额。
    故这里取 `vector_id`（点 ID 列），与 Qdrant payload 严格同值；为空时回退 `chunk.id`。

    格式统一为无连字符小写 hex（Qdrant payload 的既有形态）。
    """
    value = getattr(chunk, "vector_id", None) or chunk.id.hex
    return str(value).replace("-", "").lower()


class KeywordRetriever:
    """关键词检索：MySQL LIKE 精确召回（错误码 / 版本号 / API 名 / 专有名词）。

    未抽到关键词时直接返回空（不打库）：纯自然语言查询交给向量路更合适。
    命中项的 `score` 为 None —— 关键词命中没有与余弦相似度可比的分数，
    且 RRF 只用排名，因此 `sources[].score` 的既有语义（向量相似度）得以保持不变。
    """

    name = KEYWORD
    # 关键词路是增强能力：失败仅告警并退回向量结果（RAG 能力应允许降级）
    optional = True

    def __init__(self, top_k: int | None = None, max_terms: int | None = None) -> None:
        self.top_k = top_k
        self.max_terms = max_terms

    async def retrieve(self, db: AsyncSession, kbs: list[Any], query: str) -> list[dict]:
        if not kbs:
            return []
        terms = extract_keywords(query, self.max_terms)
        if not terms:
            return []
        # 延迟导入：避免 knowledge → ai 的模块级依赖
        from app.modules.knowledge.repository import ChunkRepository

        limit = self.top_k if self.top_k is not None else get_settings().rag_keyword_top_k
        rows = await ChunkRepository(db).search_by_keywords(
            [kb.id for kb in kbs], terms, max(1, limit)
        )
        return [
            {
                # 融合键必须与 Qdrant payload.chunk_id 取**同一个值**（不只是同一种格式），
                # 详见 fusion_key() 的说明
                "chunk_id": fusion_key(chunk),
                "document_id": chunk.document_id.hex,
                "knowledge_base_id": chunk.knowledge_base_id.hex,
                "seq_no": chunk.seq_no,
                "content": chunk.content,
                "score": None,
                "retrievers": [self.name],
            }
            for chunk, _doc_name in rows
        ]


class HybridRetriever:
    """混合检索：多路召回 + RRF 融合。

    降级策略：只有 `optional=True` 的检索器失败时才跳过（记录 warning）；
    主召回（向量路）失败仍然向上抛，保持既有失败语义不变。
    """

    name = HYBRID
    optional = False

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: list[float] | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.retrievers = retrievers
        self.weights = weights
        self.top_k = top_k
        self.rrf_k = rrf_k

    async def retrieve(self, db: AsyncSession, kbs: list[Any], query: str) -> list[dict]:
        ranked: list[list[dict]] = []
        for retriever in self.retrievers:
            try:
                ranked.append(await retriever.retrieve(db, kbs, query))
            except Exception as exc:  # noqa: BLE001 - 单路失败降级，不影响整体检索
                if not getattr(retriever, "optional", False):
                    raise
                logger.warning("检索器失败已降级 name={} err={}", retriever.name, exc)
                ranked.append([])
        return rrf_fuse(ranked, self.weights, self.rrf_k, self.top_k)


def build_retriever() -> Retriever:
    """按配置构造检索器（新增检索能力时的唯一注册点）。

    `rag_hybrid_enabled=False` → 纯向量检索，便于 A/B 对比与故障降级。
    """
    settings = get_settings()
    vector_retriever = VectorRetriever(fetch_k=settings.rag_vector_fetch_k)
    if not settings.rag_hybrid_enabled:
        return vector_retriever
    # 融合上限：启用 Reranker 时放大为候选池（召回宁多勿漏，精排负责收窄），
    # 未启用时维持 rag_top_k —— 关闭态行为与改造前完全一致。
    limit = settings.rag_candidate_k if settings.rag_rerank_enabled else settings.rag_top_k
    return HybridRetriever(
        retrievers=[
            vector_retriever,
            KeywordRetriever(
                top_k=settings.rag_keyword_top_k,
                max_terms=settings.rag_keyword_max_terms,
            ),
        ],
        weights=[settings.rag_vector_weight, settings.rag_keyword_weight],
        top_k=max(1, limit),
        rrf_k=settings.rag_rrf_k,
    )
