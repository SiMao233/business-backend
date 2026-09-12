"""AI 检索能力：RAG 知识库召回（Agent 绑定知识库 → 向量检索 → 参考资料）。

收敛所有「读知识库配置 + 向量检索」的逻辑，供两处复用：
- 纯 RAG 对话（chat/service.py）：无条件检索后注入参考资料；
- `knowledge_retrieval` 工具（tools.py）：由模型自主决定是否检索。

约束：Agent 绑定的多个知识库应使用同一 embedding 模型（用第一个知识库的模型构造查询向量）。
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings


async def retrieve_hits(db: AsyncSession, config: dict, query: str) -> list[dict]:
    """检索命中的片段列表。

    无绑定知识库 / 知识库未启用 / 无命中时返回空列表。
    """
    knowledge_ids = config.get("knowledge_ids") or []
    if not knowledge_ids:
        return []

    # 延迟导入：避免与 knowledge / vector 模块产生循环依赖
    from app.modules.ai import vector
    from app.modules.knowledge.repository import KnowledgeBaseRepository

    kb_repo = KnowledgeBaseRepository(db)
    kbs = []
    for kid in knowledge_ids:
        kb = await kb_repo.get_by_id(UUID(kid))
        if kb is not None and kb.status == 1:
            kbs.append(kb)
    if not kbs:
        return []

    embeddings = await vector.build_embeddings(db, kbs[0].embedding_model_id)
    query_vector = await embeddings.aembed_query(query)
    return await vector.search_chunks(
        query_vector, [kb.id.hex for kb in kbs], get_settings().rag_top_k
    )


async def retrieve_context(db: AsyncSession, config: dict, query: str) -> str:
    """检索并拼装「参考资料」文本；无命中返回空串。"""
    return format_context(await retrieve_hits(db, config, query))


def format_context(hits: list[dict]) -> str:
    """把命中片段拼装成「参考资料」文本（编号 + 内容）；无命中返回空串。"""
    if not hits:
        return ""
    parts = [f"[{i + 1}] {h['content']}" for i, h in enumerate(hits)]
    return "\n\n".join(parts)
