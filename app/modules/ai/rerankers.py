"""AI 重排序器（Reranker）：Hybrid 检索之后的精排环节。

职责边界（本模块只做一件事）：
- `retrievers.py` 负责**召回**（Vector + Keyword + RRF 融合），目标是「宁多勿漏」；
- 本模块负责**精排**（query + 候选片段 → relevance score），目标是「排序准」。
两者都只接在 `retrieval.retrieve_hits()` 内部，因此 Chat / SSE 协议与
`AiChatService` 完全不需要改动（不与 LLM Chat Service 强耦合）。

为什么走「OpenAI 兼容 rerank 端点」（`POST {base_url}/reranks`，模型如 qwen3-rerank）：
- 与 `vector.OpenAICompatibleEmbeddings` 同一套范式：直接用 `AsyncOpenAI` 客户端，
  不依赖 langchain（langchain 对第三方兼容服务的请求格式适配很差）；
- ⚠️ rerank 的 base_url 与 embedding **不同**（embedding 是 `compatible-mode/v1`，
  rerank 是 `compatible-api/v1`），所以不走知识库的 embedding_model_id，
  而是由配置指定的**供应商**承载专用 base_url / api_key，代码里不拼 URL。

分数语义：rerank 分数写入 `hit["rerank_score"]`，**不覆盖 `hit["score"]`**。
`sources[].score` 的既有语义是向量余弦相似度（纯关键词命中为 null），
覆盖它会破坏前端契约，故两者并存。

降级设计（AGENTS.md 第 14 条：RAG 能力必须允许关闭或降级）：
- 总开关关闭 → `NoopReranker`，且**零外部调用**；
- 配置缺失 / 供应商或实例查不到 / 实例停用 / 类型不符 → `NoopReranker` + warning（不抛异常）；
- 调用失败或超时 → warning 后退回融合结果，**绝不让检索链路 500**。

扩展点：新增 Reranker 只需实现 `Reranker` 协议并在 `build_reranker()` 注册
（例如换成本地 cross-encoder 时，只需要新增一个类，下游零改动）。
"""

from typing import Any, Protocol

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import ModelType
from app.core.config import get_settings

# Reranker 名称标记（供日志与测试断言）
NOOP = "noop"
MODEL = "model"


class Reranker(Protocol):
    """重排序器协议：实现 `rerank()` 即可替换实现（与 `Retriever` 协议同构）。

    约定：
    - `hits` 是召回阶段的候选片段（含 content / chunk_id 等），**不得原地修改**；
    - 返回按相关性降序的片段列表，可在 `hit["rerank_score"]` 上附加分数；
    - `optional=True` 表示该实现失败时允许降级（退回未重排的候选），不阻断对话。
    """

    name: str
    optional: bool

    async def rerank(self, query: str, hits: list[dict], top_n: int) -> list[dict]: ...


class NoopReranker:
    """空实现：恒等返回候选（不排序、不截断、不发任何请求）。

    用于「总开关关闭」与「配置不可用」两种降级场景。
    截断到 `rag_top_k` 由 `apply_rerank()` 统一负责，这里刻意不做，
    以保证开启/关闭 Reranker 时的收尾逻辑只有一处。
    """

    name = NOOP
    # 恒等实现不可能失败，故不参与 optional 降级判断
    optional = False

    async def rerank(self, query: str, hits: list[dict], top_n: int) -> list[dict]:
        return hits


def apply_scores(hits: list[dict], results: Any, threshold: float = 0.0) -> list[dict]:
    """把 rerank 响应映射回候选片段（纯函数）。

    `results` 每项形如 `{"index": 0, "relevance_score": 0.90}`，`index` 是
    **请求里 documents 数组的下标**（等于 hits 的下标），按位置映射即可，
    不需要用 chunk_id 回查（省一次比对，也不受 ID 形态差异影响）。

    防御（脏数据不得打断检索链路）：非 dict 项、index 缺失/越界/重复、
    relevance_score 非数值的项一律丢弃。

    返回值为**新的 dict**（浅拷贝），不修改调用方的 hit；按分数降序。
    """
    scored: list[tuple[float, dict]] = []
    seen: set[int] = set()
    for item in results or []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        if not isinstance(score, int | float) or isinstance(score, bool):
            continue
        if index < 0 or index >= len(hits) or index in seen:
            continue
        seen.add(index)
        if score < threshold:
            continue
        hit = dict(hits[index])
        hit["rerank_score"] = float(score)
        scored.append((hit["rerank_score"], hit))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _score, hit in scored]


def _extract_results(data: Any) -> list:
    """从 rerank 响应体里取出结果列表。

    兼容两种形状：OpenAI 兼容端点返回扁平的 `{"results": [...]}`，
    原生 DashScope 端点返回嵌套的 `{"output": {"results": [...]}}`。
    多做一次兼容的成本极低，却能避免上游改结构后整条检索链路直接失效。
    """
    if not isinstance(data, dict):
        return []
    results = data.get("results")
    if isinstance(results, list):
        return results
    output = data.get("output")
    if isinstance(output, dict) and isinstance(output.get("results"), list):
        return output["results"]
    return []


class OpenAICompatibleReranker:
    """OpenAI 兼容协议的 Reranker（`POST {base_url}/reranks`，如百炼 qwen3-rerank）。

    请求体刻意只传 `content`：`ChunkDraft.embedding_text` 带 `section_path` 前缀，
    是给向量召回补语义用的；精排面对的是「用户问题 → 用户能看到的正文」，
    传前缀反而会给标题类内容虚高的相关性。
    """

    name = MODEL
    # Reranker 是增强能力：失败仅告警并退回融合结果（RAG 能力必须允许降级）
    optional = True

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None,
        *,
        timeout: float,
        instruct: str = "",
        score_threshold: float = 0.0,
    ) -> None:
        # 延迟导入：与 vector.py 保持一致，不在模块导入期拉起 openai SDK
        from openai import AsyncOpenAI

        self.model = model
        self.instruct = instruct
        self.score_threshold = score_threshold
        # timeout 交给 SDK：超时抛 APITimeoutError，由 apply_rerank() 统一降级。
        # ⚠️ 必须显式 max_retries=0：SDK 默认重试 2 次，会让最坏延迟变成
        # timeout × 3 + 退避（实测 timeout=1s 时降级实际耗时 4.3s）。
        # 精排是串行阻塞首 token 的增强环节，宁可立刻降级也不要重试拖慢对话。
        self._client = AsyncOpenAI(
            api_key=api_key or "not-set",
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    async def rerank(self, query: str, hits: list[dict], top_n: int) -> list[dict]:
        if not hits:
            return []
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [hit.get("content") or "" for hit in hits],
            # top_n 不得为 0（会被上游当成非法参数），也不该超过候选数
            "top_n": max(1, min(top_n, len(hits))),
        }
        if self.instruct:
            body["instruct"] = self.instruct
        # cast_to=object → 返回原始 JSON dict（SDK 不做响应模型校验，兼容网关差异）
        data = await self._client.post("/reranks", cast_to=object, body=body)
        return apply_scores(hits, _extract_results(data), self.score_threshold)


async def build_reranker(db: AsyncSession) -> Reranker:
    """按配置构造 Reranker（工厂，也是**唯一的降级判定点**）。

    本函数**保证不抛异常**：无论「总开关关闭 / 配置缺失 / 供应商或实例查不到 / 已停用 /
    类型不符」还是「解析过程本身出错（如 DB 抖动）」都返回 `NoopReranker` 并记录 warning。
    Reranker 是检索链路上的增强环节，配置或依赖异常不应该让整个 Chat 不可用。

    `rag_rerank_enabled=False` 时不查库，零开销。
    """
    try:
        return await _resolve_reranker(db)
    except Exception as exc:  # noqa: BLE001 - 解析失败必须降级为不重排
        logger.warning("Reranker 解析失败，降级为不重排 err={!r}", exc)
        return NoopReranker()


async def _resolve_reranker(db: AsyncSession) -> Reranker:
    """按配置解析出 Reranker 实现（可能抛异常，由 `build_reranker()` 兜底）。"""
    settings = get_settings()
    if not settings.rag_rerank_enabled:
        return NoopReranker()

    if not settings.rerank_provider_code or not settings.rerank_model_code:
        logger.warning(
            "Reranker 已启用但未配置模型，降级为不重排 provider_code={} model_code={}",
            settings.rerank_provider_code,
            settings.rerank_model_code,
        )
        return NoopReranker()

    # 延迟导入：避免与 agent / knowledge 模块产生循环依赖
    from app.modules.agent.model.repository import (
        ModelInstanceRepository,
        ModelProviderRepository,
    )

    provider = await ModelProviderRepository(db).get_by_code(settings.rerank_provider_code)
    if provider is None:
        logger.warning("Reranker 供应商不存在，降级为不重排 code={}", settings.rerank_provider_code)
        return NoopReranker()
    if provider.status != 1:
        logger.warning("Reranker 供应商已停用，降级为不重排 code={}", provider.code)
        return NoopReranker()

    instance = await ModelInstanceRepository(db).get_by_provider_and_code(
        provider.id, settings.rerank_model_code
    )
    if instance is None:
        logger.warning(
            "Reranker 模型实例不存在，降级为不重排 provider={} model_code={}",
            provider.code,
            settings.rerank_model_code,
        )
        return NoopReranker()
    if instance.status != 1:
        logger.warning("Reranker 模型实例已停用，降级为不重排 code={}", instance.code)
        return NoopReranker()
    # 类型校验：避免把 chat / embedding 实例误配成 Reranker（报错会很难定位）
    if instance.model_type != ModelType.RERANK.value:
        logger.warning(
            "Reranker 模型实例类型不是 rerank，降级为不重排 code={} model_type={}",
            instance.code,
            instance.model_type,
        )
        return NoopReranker()

    return OpenAICompatibleReranker(
        model=instance.code,
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=settings.rag_rerank_timeout,
        instruct=settings.rag_rerank_instruct,
        score_threshold=settings.rag_rerank_score_threshold,
    )


async def apply_rerank(db: AsyncSession, query: str, hits: list[dict]) -> list[dict]:
    """对召回候选执行精排，返回最终交给 Context Builder 的片段（含降级与截断）。

    截断到 `rag_top_k` 统一在这里做：Noop 恒等返回候选池，截断后与
    「未启用 Reranker」时的结果完全一致（关闭态行为零变化）。
    """
    if not hits:
        return []
    settings = get_settings()
    reranker = await build_reranker(db)

    ranked = hits
    try:
        ranked = await reranker.rerank(query, hits, max(1, settings.rag_candidate_k))
    except Exception as exc:  # noqa: BLE001 - 精排失败不得影响对话主流程
        if not reranker.optional:
            raise
        logger.warning("Reranker 调用失败，回退为融合结果 name={} err={!r}", reranker.name, exc)
        ranked = hits

    # 分数区间日志：用于观察真实分布以决定 rag_rerank_score_threshold
    scores = [h["rerank_score"] for h in ranked if h.get("rerank_score") is not None]
    if scores:
        logger.info(
            "Reranker 完成 name={} 候选={} 保留={} 分数区间=[{:.4f}, {:.4f}]",
            reranker.name,
            len(hits),
            len(ranked),
            scores[-1],
            scores[0],  # ranked 已按分数降序
        )
    else:
        logger.debug("Reranker 未生效 name={} 候选={}", reranker.name, len(hits))

    return ranked[: max(1, settings.rag_top_k)]
