"""Knowledge Service 层：业务规则、权限判断、流程编排。

知识库 CRUD + 文档管理（上传/删除/列表）+ 文档向量化后台任务编排。
文档处理流水线：上传 → 入队 ARQ → 解析 → 切分 → embedding → 写 Qdrant → 双写 MySQL chunk。
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid5

import anyio
from fastapi import UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import DocumentStatus, FileBizType, ModelType
from app.common.pagination import PageParams, PageResult
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError, ValidateError
from app.modules.agent.model.repository import ModelInstanceRepository
from app.modules.file.service import FileService
from app.modules.knowledge.model import KnowledgeBase, KnowledgeChunk, KnowledgeDocument
from app.modules.knowledge.repository import (
    ChunkRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
)
from app.modules.knowledge.schema import (
    ChunkOut,
    ChunkSearchOut,
    ChunkSearchQuery,
    DocumentOut,
    DocumentQuery,
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    KnowledgeBaseQuery,
    KnowledgeBaseUpdate,
)


class KnowledgeService:
    """知识库管理服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = KnowledgeBaseRepository(db)
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.instance_repo = ModelInstanceRepository(db)

    # 分页查询知识库列表
    async def list_page(self, query: KnowledgeBaseQuery) -> PageResult[KnowledgeBaseOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params,
            keyword=query.keyword,
            organization_id=query.organization_id,
            status=query.status,
        )
        return PageResult(
            list=[KnowledgeBaseOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询单个知识库详情
    async def get(self, knowledge_base_id: UUID) -> KnowledgeBaseOut:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        return KnowledgeBaseOut.model_validate(kb)

    # 创建知识库
    async def create(self, req: KnowledgeBaseCreate, creator_id: UUID | None) -> KnowledgeBaseOut:
        if await self.repo.get_by_code(req.code):
            raise BizError("知识库编码已存在")
        await self._validate_embedding_model(req.embedding_model_id)
        kb = KnowledgeBase(
            name=req.name,
            code=req.code,
            description=req.description,
            icon_id=req.icon_id,
            organization_id=req.organization_id,
            embedding_model_id=req.embedding_model_id,
            status=1,
            creator_id=creator_id,
        )
        return KnowledgeBaseOut.model_validate(await self.repo.create(kb))

    # 更新知识库（code 不可改）
    async def update(self, knowledge_base_id: UUID, req: KnowledgeBaseUpdate) -> KnowledgeBaseOut:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        if req.name is not None:
            kb.name = req.name
        if req.description is not None:
            kb.description = req.description
        if req.icon_id is not None:
            kb.icon_id = req.icon_id
        if req.organization_id is not None:
            kb.organization_id = req.organization_id
        if req.embedding_model_id is not None:
            await self._validate_embedding_model(req.embedding_model_id)
            kb.embedding_model_id = req.embedding_model_id
        if req.status is not None:
            kb.status = req.status
        return KnowledgeBaseOut.model_validate(await self.repo.update(kb))

    # 删除知识库（文档/切分块由 DB CASCADE 删除；同时清理 Qdrant 向量）
    # 返回被删除对象，供 API 层写入操作日志摘要
    async def delete(self, knowledge_base_id: UUID) -> KnowledgeBase:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        # 清理 Qdrant 向量（按 knowledge_base_id payload 过滤）
        from app.modules.ai import vector

        await vector.delete_by_filter(knowledge_base_id=kb.id.hex)
        await self.repo.delete(kb)
        return kb

    # 上传文档：复用文件上传 → 建文档记录(pending) → 入队 ARQ 后台向量化
    async def upload_document(
        self, file: UploadFile, knowledge_base_id: UUID, uploader_id: UUID
    ) -> DocumentOut:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        if kb.status != 1:
            raise BizError("知识库已禁用，无法上传文档")
        # 复用通用文件上传（biz_type=knowledge_doc）
        file_out = await FileService(self.db).upload(file, uploader_id, FileBizType.KNOWLEDGE_DOC)
        doc = KnowledgeDocument(
            knowledge_base_id=knowledge_base_id,
            file_id=file_out.id,
            name=file_out.name,
            status=DocumentStatus.PENDING.value,
            chunk_count=0,
            uploader_id=uploader_id,
        )
        doc = await self.doc_repo.create(doc)
        # 入队后台向量化（失败不影响上传，任务内会标记 failed）
        # 不传 job_id：ARQ 的 _job_id 去重窗口 = 执行期 + keep_result（默认 1 小时），
        # 会静默丢弃“上次失败后立刻重试”的入队，并发保护改由任务的 CAS 抢占承担。
        from app.tasks.worker import enqueue_job

        try:
            await enqueue_job("process_document", doc.id.hex)
        except Exception as exc:  # noqa: BLE001 - 入队失败仅记录，不阻断上传
            logger.warning("文档入队失败 document_id={} err={}", doc.id, exc)
        return DocumentOut.model_validate(doc)

    # 删除文档：清理 Qdrant 向量 + 删除记录（切分块由 DB CASCADE 删除）
    # 返回被删除对象，供 API 层写入操作日志摘要
    async def delete_document(self, document_id: UUID) -> KnowledgeDocument:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundError("文档不存在")
        from app.modules.ai import vector

        await vector.delete_by_filter(document_id=doc.id.hex)
        await self.doc_repo.delete(doc)
        return doc

    # 重试文档向量化：复用原文件重新入队（failed / pending 可重试；
    # parsing 超过 index_stale_minutes 视为僵尸任务，同样允许重试）
    async def reprocess_document(self, document_id: UUID) -> DocumentOut:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundError("文档不存在")
        if doc.status == DocumentStatus.PARSED.value:
            raise BizError("文档已处理完成，无需重试")
        if doc.status == DocumentStatus.PARSING.value and not self._is_stale_parsing(doc):
            raise BizError("文档正在处理中，请勿重复操作")
        # 重置状态并重新入队（复用原文件，不重新上传）
        doc.status = DocumentStatus.PENDING.value
        doc.error_message = None
        doc.chunk_count = 0
        await self.doc_repo.update(doc)
        # 不传 job_id（原因见 upload_document）：并发保护由任务内 CAS 抢占承担
        from app.tasks.worker import enqueue_job

        try:
            await enqueue_job("process_document", doc.id.hex)
        except Exception as exc:  # noqa: BLE001 - 入队失败仅记录，不阻断重试
            logger.warning("文档重试入队失败 document_id={} err={}", doc.id, exc)
        return DocumentOut.model_validate(doc)

    @staticmethod
    def _is_stale_parsing(doc: KnowledgeDocument) -> bool:
        """parsing 状态超过 index_stale_minutes 视为僵尸任务（worker 崩溃/被杀）。

        ARQ 任务超时/取消已被任务内兜底标记 failed，此判定覆盖的是
        worker 进程被强杀（SIGKILL）等无法执行兜底逻辑的场景，
        否则文档会永久卡在 parsing 且无法通过任何 API 恢复。
        """
        updated = doc.update_time
        if updated is None:
            return True
        if updated.tzinfo is None:  # MySQL 读回为 naive，按 UTC 还原（存取全链路 UTC）
            updated = updated.replace(tzinfo=UTC)
        age = datetime.now(UTC) - updated
        return age > timedelta(minutes=get_settings().index_stale_minutes)

    # 分页查询某知识库的文档列表（knowledge_base_id 从查询入参 body 传入）
    async def list_documents(self, query: DocumentQuery) -> PageResult[DocumentOut]:
        kb = await self.repo.get_by_id(query.knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.doc_repo.list_page(
            query.knowledge_base_id,
            page_params,
            keyword=query.keyword,
            status=query.status.value if query.status else None,
        )
        return PageResult(
            list=[DocumentOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询某文档的全部分块（按序号升序）
    async def list_document_chunks(self, document_id: UUID) -> list[ChunkOut]:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundError("文档不存在")
        chunks = await self.chunk_repo.list_by_document(document_id)
        return [ChunkOut.model_validate(c) for c in chunks]

    # 按关键字搜索某知识库内的分块（含来源文档名，分页）
    async def search_chunks(self, query: ChunkSearchQuery) -> PageResult[ChunkSearchOut]:
        kb = await self.repo.get_by_id(query.knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.chunk_repo.search_by_keyword(
            query.knowledge_base_id, query.keyword, page_params
        )
        items: list[ChunkSearchOut] = []
        for chunk, doc_name in page.list:
            out = ChunkSearchOut.model_validate(chunk)
            out.document_name = doc_name
            items.append(out)
        return PageResult(
            list=items,
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 校验 embedding 模型实例：存在、启用且类型为 embedding
    async def _validate_embedding_model(self, instance_id: UUID | None) -> None:
        if instance_id is None:
            return
        instance = await self.instance_repo.get_by_id(instance_id)
        if instance is None:
            raise NotFoundError("绑定模型实例不存在")
        if instance.model_type != ModelType.EMBEDDING.value:
            raise ValidateError("绑定模型实例必须为 embedding 类型")
        if instance.status != 1:
            raise BizError("绑定模型实例已停用")


# Chunk 确定性 ID 命名空间（固定值，保证跨进程/重启后 ID 稳定）
_CHUNK_ID_NAMESPACE = UUID("5f3a9c1e-7b2d-4e8f-9a6c-1d0b4e7f2a83")


def chunk_point_id(document_id: UUID, seq_no: int) -> UUID:
    """生成 chunk 的确定性 ID（同时作为 Qdrant point ID）。

    由 (document_id, seq_no) 唯一决定：同一文档同一序号重复索引得到相同 ID——
    MySQL 主键天然幂等（重复插入报 PK 冲突而非静默新增重复行），
    Qdrant 同 ID upsert 即覆盖。重复执行索引任务不会产生重复数据。
    """
    return uuid5(_CHUNK_ID_NAMESPACE, f"kbchunk:{document_id.hex}:{seq_no}")


def build_index_payload(
    document_id: UUID,
    knowledge_base_id: UUID,
    chunks: list[str],
    vectors: list[list[float]],
) -> tuple[list[dict], list[KnowledgeChunk]]:
    """把切分文本与向量组装为 Qdrant points 与 MySQL chunk 记录（纯函数）。

    - point / chunk ID 用 chunk_point_id() 确定性生成（幂等根基）；
    - chunks 与 vectors 数量必须一致（strict 校验，杜绝静默截断丢数据）。
    payload 约定与存量数据一致：ID 均为 .hex（无连字符）。
    """
    if len(chunks) != len(vectors):
        raise BizError(
            f"向量数量与切分块数量不一致 chunks={len(chunks)} vectors={len(vectors)}"
        )
    points: list[dict] = []
    chunk_records: list[KnowledgeChunk] = []
    for i, (chunk_text, vec) in enumerate(zip(chunks, vectors, strict=True), start=1):
        chunk_id = chunk_point_id(document_id, i)
        point_id = str(chunk_id)
        points.append(
            {
                "id": point_id,
                "vector": vec,
                "payload": {
                    "knowledge_base_id": knowledge_base_id.hex,
                    "document_id": document_id.hex,
                    "chunk_id": chunk_id.hex,
                    "seq_no": i,
                    "content": chunk_text,
                },
            }
        )
        chunk_records.append(
            KnowledgeChunk(
                id=chunk_id,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                seq_no=i,
                content=chunk_text,
                token_count=max(1, len(chunk_text) // 2),  # 粗略估算（中文为主）
                vector_id=point_id,
                char_count=len(chunk_text),
            )
        )
    return points, chunk_records


async def embed_in_batches(embeddings: object, chunks: list[str]) -> list[list[float]]:
    """分批向量化（顺序执行，避免单次请求超过兼容接口的 input 数组上限）。

    上游（阿里云百炼等）对单次 embeddings 调用的 input 条数有硬上限，
    大文档一次性提交会 400/413，故按 embedding_batch_size 顺序分批。
    """
    batch_size = max(1, get_settings().embedding_batch_size)
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(await embeddings.aembed_documents(batch))  # type: ignore[attr-defined]
    return vectors


async def process_document_task(document_id: str) -> dict:
    """ARQ 后台任务：解析 → 切分 → embedding → 清旧索引 → 写新索引。

    独立 AsyncSessionLocal 会话（worker 进程内执行，不依赖请求上下文）。
    状态流转：pending → parsing → parsed / failed。

    三阶段索引生命周期（保证重新索引一致性）：
    1. 抢占：CAS 置 status=parsing（并发任务被数据库层挡住，不依赖队列去重）；
    2. 只读计算：解析/切分/分批 embedding/维度校验/组装 payload——全程不碰旧索引，
       失败时旧索引完好，文档标 failed，重试即可；
    3. 替换：清旧向量 → 清旧 chunk → 写新向量 → 写新 chunk → 标 parsed。
       chunk/point ID 确定性生成，重复执行天然幂等。
    """
    from app.core.database import AsyncSessionLocal
    from app.modules.ai import vector
    from app.modules.knowledge.parser import parse_document, split_text

    async with AsyncSessionLocal() as db:
        doc_repo = DocumentRepository(db)
        kb_repo = KnowledgeBaseRepository(db)
        chunk_repo = ChunkRepository(db)
        file_service = FileService(db)

        doc = await doc_repo.get_by_id(UUID(document_id))
        if not doc:
            return {"ok": False, "error": "文档不存在"}

        # 阶段 1 抢占：CAS 置 parsing。抢不到说明已有任务在处理该文档（连点重试 /
        # 重复入队），直接跳过——并发跑两个索引任务会撞 chunk 主键冲突并把文档误标 failed。
        if not await doc_repo.mark_parsing(doc.id):
            logger.info("文档已在处理中，跳过本次索引 document_id={}", document_id)
            return {"ok": True, "skipped": "already parsing"}
        # 同步内存对象（CAS 走 Core UPDATE，ORM 实例不会自动刷新）
        doc.status = DocumentStatus.PARSING.value
        stage = "prepare"
        try:
            kb = await kb_repo.get_by_id(doc.knowledge_base_id)
            if not kb:
                raise BizError("知识库不存在")

            # ---- 阶段 2：只读计算（失败不影响旧索引）----
            # 读取文件内容（本地路径或字节）
            content = await file_service.get_content(doc.file_id)
            if content.data is not None:
                data = content.data
            elif content.path is not None:
                data = content.path.read_bytes()
            else:
                raise BizError("文件内容不可读")

            stage = "parse"
            # 解析 + 切分
            ext = doc.name.rsplit(".", 1)[-1].lower() if "." in doc.name else None
            text = parse_document(data, ext)
            chunks = split_text(text)
            if not chunks:
                raise BizError("文档内容为空或无法切分")

            stage = "embedding"
            # embedding（读知识库绑定的 embedding 模型实例，分批提交）
            embeddings = await vector.build_embeddings(db, kb.embedding_model_id)
            vectors = await embed_in_batches(embeddings, chunks)
            await vector.ensure_collection(len(vectors[0]))
            points, chunk_records = build_index_payload(
                doc.id, doc.knowledge_base_id, chunks, vectors
            )

            # ---- 阶段 3：替换旧索引（先清后写，确定性 ID 保证幂等）----
            stage = "index"
            await vector.delete_by_filter(document_id=doc.id.hex)
            await chunk_repo.delete_by_document(doc.id)
            await vector.upsert_chunks(points)
            await chunk_repo.create_many(chunk_records)

            doc.status = DocumentStatus.PARSED.value
            doc.chunk_count = len(chunk_records)
            await doc_repo.update(doc)
            logger.info("文档处理完成 document_id={} chunks={}", doc.id, len(chunk_records))
            return {"ok": True, "chunk_count": len(chunk_records)}
        except BaseException as exc:  # noqa: BLE001 - 任务内兜底标记 failed（含取消）
            # ARQ job_timeout 会取消任务（CancelledError 是 BaseException，
            # except Exception 捕不到 → 文档会永久卡在 parsing 且无法重试）。
            # 标记 failed 的写库放在屏蔽取消的作用域内完成，随后恢复取消语义。
            #
            # 文案口径：error_message 是**面向用户的业务文案**（前端直接展示），
            # 失败阶段（stage）只进日志与任务返回值，不污染该字段。
            if isinstance(exc, asyncio.CancelledError):
                # CancelledError 的 str() 为空，必须给兜底文案，否则 error_message 写成空串
                message = "任务被中断（可能超时或服务重启），请稍后重试"
            else:
                message = str(exc) or "文档处理失败"
            with anyio.CancelScope(shield=True):
                try:
                    doc.status = DocumentStatus.FAILED.value
                    doc.error_message = message[:500]
                    await doc_repo.update(doc)
                except Exception:  # noqa: BLE001 - 状态落库失败不能掩盖原始异常
                    logger.exception("文档失败状态落库失败 document_id={}", document_id)
            if isinstance(exc, asyncio.CancelledError):
                logger.warning("文档处理被取消 document_id={} stage={}", document_id, stage)
                raise
            logger.exception("文档处理失败 document_id={} stage={}", document_id, stage)
            return {"ok": False, "error": str(exc), "stage": stage}

