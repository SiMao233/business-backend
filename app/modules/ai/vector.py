"""向量检索基础设施：Qdrant 封装 + embedding 客户端构造。

AI 能力边界：所有 Qdrant 读写与 embedding 调用收敛在本文件，
业务模块（knowledge / ai）通过本模块访问向量数据，不直接依赖 qdrant-client。
未来抽独立 ai-service 时整体迁移本文件。

设计：
- 单一 collection（配置 qdrant_collection），payload 按 knowledge_base_id / document_id 过滤；
- collection 首次写入时按 embedding 实际维度创建；
- embedding 客户端基于知识库绑定的模型实例（OpenAI 兼容协议）构造。
"""

from functools import lru_cache
from uuid import UUID

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.model.repository import ModelInstanceRepository, ModelProviderRepository


class OpenAICompatibleEmbeddings:
    """OpenAI 兼容 embedding 客户端（基于 openai AsyncOpenAI）。

    直接使用 openai 客户端而非 langchain-openai 的 OpenAIEmbeddings：
    部分 OpenAI 兼容服务（如阿里云百炼）对 langchain 的请求格式兼容性差
    （langchain 可能带额外参数或 input 格式不符），openai 客户端更稳定。
    input 统一传字符串数组。
    """

    def __init__(self, model: str, api_key: str | None, base_url: str | None) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(api_key=api_key or "not-set", base_url=base_url)

    async def aembed_query(self, text: str) -> list[float]:
        """单条文本向量化（内部按数组发送）。"""
        resp = await self._client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化。"""
        resp = await self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    """获取 Qdrant 异步客户端单例（进程内复用连接）。"""
    settings = get_settings()
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        https=settings.qdrant_https,
        api_key=settings.qdrant_api_key,
    )


async def ensure_collection(dim: int) -> None:
    """确保 collection 存在；不存在则按 embedding 维度创建（余弦距离）。"""
    settings = get_settings()
    client = get_qdrant_client()
    collections = await client.get_collections()
    names = {c.name for c in collections.collections}
    if settings.qdrant_collection not in names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qdrant_models.VectorParams(
                size=dim, distance=qdrant_models.Distance.COSINE
            ),
        )
        logger.info("Qdrant collection created: {} dim={}", settings.qdrant_collection, dim)


async def upsert_chunks(points: list[dict]) -> None:
    """批量写入向量点。

    point 结构：{"id": str, "vector": list[float], "payload": dict}
    payload 约定：knowledge_base_id / document_id / chunk_id / seq_no / content
    """
    settings = get_settings()
    client = get_qdrant_client()
    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qdrant_models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ],
    )


async def delete_by_filter(**filters: str) -> None:
    """按 payload 字段等值过滤删除向量（如 knowledge_base_id / document_id）。

    容错设计：Qdrant 清理是辅助操作（MySQL 删除为主），连接失败仅记录 warning，
    不阻断业务删除流程。
    """
    settings = get_settings()
    client = get_qdrant_client()
    try:
        await client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=k, match=qdrant_models.MatchValue(value=v)
                        )
                        for k, v in filters.items()
                    ]
                )
            ),
        )
    except Exception as exc:  # noqa: BLE001 - Qdrant 不可用时降级，不阻断删除
        logger.warning("Qdrant 向量清理失败 filters={} err={}", filters, exc)


async def search_chunks(
    query_vector: list[float], knowledge_base_ids: list[str], top_k: int
) -> list[dict]:
    """检索相似块，返回按分数降序的 [{chunk_id, document_id, content, score}]。

    仅返回分数 >= rag_score_threshold 的结果。
    """
    settings = get_settings()
    client = get_qdrant_client()
    # qdrant-client 1.19 已移除 `search`（1.12 起废弃、1.16 起删除），统一用 query_points
    result = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="knowledge_base_id",
                    match=qdrant_models.MatchAny(any=knowledge_base_ids),
                )
            ]
        ),
        with_payload=True,
    )
    return [
        {
            "chunk_id": hit.payload.get("chunk_id"),
            "document_id": hit.payload.get("document_id"),
            "content": hit.payload.get("content", ""),
            "score": hit.score,
        }
        for hit in result.points
        if hit.score >= settings.rag_score_threshold
    ]


async def build_embeddings(
    db: AsyncSession, embedding_model_id: UUID | None
) -> OpenAICompatibleEmbeddings:
    """根据知识库绑定的 embedding 模型实例构造 embedding 客户端（OpenAI 兼容协议）。

    校验：实例存在、启用；供应商提供 base_url / api_key。
    """
    if embedding_model_id is None:
        raise BizError("知识库未绑定 embedding 模型实例")
    instance_repo = ModelInstanceRepository(db)
    provider_repo = ModelProviderRepository(db)
    instance = await instance_repo.get_by_id(embedding_model_id)
    if instance is None:
        raise NotFoundError("绑定模型实例不存在")
    if instance.status != 1:
        raise BizError("绑定模型实例已停用")
    provider = await provider_repo.get_by_id(instance.provider_id)
    return OpenAICompatibleEmbeddings(
        model=instance.code,
        api_key=provider.api_key if provider else None,
        base_url=provider.base_url if provider else None,
    )
