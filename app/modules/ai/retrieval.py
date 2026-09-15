"""AI 检索能力：RAG 知识库召回（Agent 绑定知识库 → 向量检索 → 参考资料 + 结构化来源）。

收敛所有「读知识库配置 + 向量检索 + 来源富化」的逻辑，供两处复用：
- 纯 RAG 对话（chat/service.py）：无条件检索后注入参考资料；
- `knowledge_retrieval` 工具（tools.py）：由模型自主决定是否检索。

约束：Agent 绑定的多个知识库应使用同一 embedding 模型（用第一个知识库的模型构造查询向量）。

`format_context()` 与 `build_sources()` 必须基于**同一个 hits 列表**、传**相同的 start 偏移**调用，
提示词里的 `[n]` 与出参 `sources[].index` 才能严格一一对应（前端据此把正文角标换成可点击来源卡片）。
Qdrant payload 不存文档名，故 `retrieve_hits()` 会批量回查 MySQL 富化（单条 IN 查询，不随命中数增长）。
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.modules.ai.steps import SourceType

# 来源条目里正文片段的截断长度（避免 steps JSON 体积随消息数量膨胀）
SOURCE_SNIPPET_LIMIT = 200


def _as_uuid(value: str | None) -> UUID | None:
    """把 Qdrant payload 里的 ID 字符串（hex 或标准 UUID）归一化为 UUID；非法/缺失返回 None。"""
    if not value:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


async def retrieve_hits(db: AsyncSession, config: dict, query: str) -> list[dict]:
    """检索命中的片段列表（已富化文档名 / 源文件 ID / 知识库名）。

    无绑定知识库 / 知识库未启用 / 无命中时返回空列表。
    返回项：{chunk_id, document_id, document_name, file_id, knowledge_base_id,
    knowledge_base_name, seq_no, content, score}
    """
    knowledge_ids = config.get("knowledge_ids") or []
    if not knowledge_ids:
        return []

    # 延迟导入：避免与 knowledge / vector 模块产生循环依赖
    from app.modules.ai import vector
    from app.modules.knowledge.repository import DocumentRepository, KnowledgeBaseRepository

    # 批量加载后按配置顺序还原（等价于逐条 get_by_id，但只有 1 次查询）
    # 注意：config 里的 id 可能是带连字符的 UUID 字符串，而 Qdrant payload 存的是 hex，
    # 统一用 UUID 对象归一化后再比对，避免两种写法互相查不到。
    kb_ids = [UUID(kid) for kid in knowledge_ids]
    kb_repo = KnowledgeBaseRepository(db)
    kb_map = {kb.id: kb for kb in await kb_repo.list_by_ids(kb_ids)}
    kbs = []
    for kb_id in kb_ids:
        kb = kb_map.get(kb_id)
        if kb is not None and kb.status == 1:
            kbs.append(kb)
    if not kbs:
        return []

    embeddings = await vector.build_embeddings(db, kbs[0].embedding_model_id)
    query_vector = await embeddings.aembed_query(query)
    hits = await vector.search_chunks(
        query_vector, [kb.id.hex for kb in kbs], get_settings().rag_top_k
    )
    if not hits:
        return []

    # 富化：批量回查文档（拿文档名与源文件 ID）；知识库名直接复用上面已加载的对象
    doc_ids = [UUID(str(hit["document_id"])) for hit in hits if hit.get("document_id")]
    docs = await DocumentRepository(db).list_by_ids(doc_ids) if doc_ids else []
    doc_map = {doc.id: doc for doc in docs}
    for hit in hits:
        doc = doc_map.get(_as_uuid(hit.get("document_id")))
        kb = kb_map.get(_as_uuid(hit.get("knowledge_base_id")))
        # 文档被删除时只保留原 ID，文档名退化为空串（前端展示占位）
        hit["document_name"] = doc.name if doc is not None else ""
        hit["file_id"] = doc.file_id.hex if doc is not None and doc.file_id is not None else None
        hit["knowledge_base_name"] = kb.name if kb is not None else ""
    return hits


async def retrieve_context(db: AsyncSession, config: dict, query: str) -> str:
    """检索并拼装「参考资料」文本；无命中返回空串。"""
    return format_context(await retrieve_hits(db, config, query))


def format_context(hits: list[dict], start: int = 0) -> str:
    """把命中片段拼装成「参考资料」文本（编号 + 文档名 + 内容）；无命中返回空串。

    `start` 是本块在**整个请求**里的编号起点：预检索传 0，模型主动调用的检索工具传
    「此前已呈现给模型的资料条数」，保证 `[n]` 在单次请求内全局唯一、不会与前面的块撞号。
    带文档名是为了让模型能按名引用，也是正文角标能落到具体来源的前提。
    """
    if not hits:
        return ""
    parts = [
        f"[{start + i + 1}]（来源：{h.get('document_name') or '未知文档'}）{h['content']}"
        for i, h in enumerate(hits)
    ]
    return "\n\n".join(parts)


def build_sources(hits: list[dict], start: int = 0) -> list[dict]:
    """把命中片段映射成 `AiSourceOut` 形状的 dict 列表（纯函数，供落库与出参共用）。

    必须与同一次 `format_context(hits, start)` 传相同的 `start`：
    `index` 即提示词与正文角标里的 `[n]`（单次请求内全局唯一）。
    键为 snake_case（出参由 Pydantic 转驼峰）；`content` 已按 `SOURCE_SNIPPET_LIMIT` 截断。
    """
    return [
        {
            "index": start + i + 1,
            "type": SourceType.KNOWLEDGE.value,
            "chunk_id": hit.get("chunk_id"),
            "document_id": hit.get("document_id"),
            "document_name": hit.get("document_name") or "",
            "knowledge_base_id": hit.get("knowledge_base_id"),
            "knowledge_base_name": hit.get("knowledge_base_name") or "",
            "seq_no": hit.get("seq_no"),
            "score": hit.get("score"),
            "file_id": hit.get("file_id"),
            "content": (hit.get("content") or "")[:SOURCE_SNIPPET_LIMIT],
        }
        for i, hit in enumerate(hits)
    ]
